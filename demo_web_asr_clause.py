# -*- coding: utf-8 -*-
"""
实时语音翻译 Demo · Web 版：中文语音输入 → 英文语音同传（滑动窗口·安全切点）
====================================================================
在 demo_web_asr_clause.py 基础上的架构级重构：

    滑动窗口    会话期间不切断喂给 ASR 的上下文。每次取完整未翻译音频
                (最多 15s) 跑 ASR，拿到带标点的长文本。因上下文完整，
                ASR 绝不会崩成“我。”。
    安全切点    寻找文本中的终结标点（逗号/句号），用 ASR 逐字时间戳取该
                标点在音频中的精确位置。将前半段音频切出送翻译，后半段
                音频和未完结文本留在 Session 里继续接新音频。
    稳定切点    切分不等 VAD 静音，改为比对相邻两轮识别的标点时间戳：同一
                处标点连续出现即认定稳定，立刻切出。半截音频上 ASR 随手补
                的伪标点会被下一轮改写掉，因而不会误切。
    TTS 流式    子句即合成单元：合成完一段立刻出声；过短碎片不再并入。
    其余机制    与 demo_web_asr_clause.py 一致。

运行：
    conda activate demo
    python demo_web_asr_clause.py         # 端口 8327，就绪后自动开麦
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import asyncio
import json
from collections import deque
from contextlib import asynccontextmanager
import queue
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import sounddevice as sd

# ---------------- 可调参数 ----------------
SAMPLE_RATE = 16000
SR_TTS = 44100
MIN_UTT_S = 0.5
NOISE_BUF_CLEAR_S = 30.0
VAD_POLL_S = 0.3
STREAM_POLL_S = 0.5          # 探测间隔降为 0.5s，配合滑动窗口更快响应
SEG_MIN_S = 1.0
OVERLONG_S = 10.0
LEVEL_POLL_S = 0.1
STREAM_GAP_S = 0.12
MIN_FRAG_WORDS = 2
MIN_FRAG_CJK = 4
TERMINAL_PUNCT = re.compile(r"[。？！；，、!?;,…]")
MIN_COMMA_CHARS = 8
CUT_PAUSE_MS = 250          # 已废弃：切分改由「标点时间戳稳定」触发，仅留作回放工具覆盖入口
OVERLONG_PAUSE_MS = 100     # 已废弃：同上
STABLE_TOL_MS = 200         # 相邻两轮识别的同一标点毫秒差在此内即视为同一处标点
PUNCT_CHARS = "。！？；，、!?;:,…"
EMOJI_RE = re.compile(r"[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF️]+")
ASR_DEVICE = "cuda:0"
VAD_DEVICE = "cpu"
MT_MODEL = "tencent/Hy-MT2-1.8B"
MT_MAX_NEW_TOKENS = 512
TTS_LANG = "en"
OUTPUT_ROOT = "demo_output"
LOG_PATH = os.path.join("logs", "service_log.txt")
MAX_ASR_AUDIO_S = 15.0       # 滑动窗口最大长度，防止爆显存

CFG = {
    "tail_silence": 1.5,
    "speech_rms": 0.001,
    "tts_steps": 6,
    "voice": "M1",
    "device": None,
}

_log_lock = threading.Lock()

def plog(tag, msg):
    line = f"[{tag}] [{datetime.now():%Y-%m-%d %H:%M:%S.%f}] {msg}"
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with _log_lock:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    broadcast({"type": "log", "line": line})

record_cost_stats = {}

def record_cost(stage, t0):
    cost = time.perf_counter() - t0
    acc = record_cost_stats.setdefault(stage, [0.0, 0])
    acc[0] += cost
    acc[1] += 1
    return cost

_clients = set()
_loop = None

def broadcast(event: dict):
    if _loop is None or not _clients:
        return
    try:
        asyncio.run_coroutine_threadsafe(
            _broadcast(json.dumps(event, ensure_ascii=False)), _loop)
    except RuntimeError:
        pass

async def _broadcast(data: str):
    dead = []
    for ws in list(_clients):
        try:
            await ws.send_text(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _clients.discard(ws)

def build_funasr(**kw):
    from funasr import AutoModel
    try:
        return AutoModel(**kw, disable_pbar=True, disable_update=True, log_level="ERROR")
    except TypeError:
        return AutoModel(**kw)

def _param_device(m):
    try:
        core = m.model if hasattr(m, "model") else m
        return str(next(core.parameters()).device)
    except Exception:
        return "unknown"

model_info = {"phase": "loading", "mock": False}

def _load_asr_and_vad():
    plog("model_jiazai", "开始加载 ASR SenseVoiceSmall + VAD fsmn-vad")
    t0 = time.perf_counter()
    from modelscope import snapshot_download
    asr_dir = snapshot_download("iic/SenseVoiceSmall")
    asr = build_funasr(model=asr_dir, device=ASR_DEVICE)
    vad = build_funasr(model="fsmn-vad", device=VAD_DEVICE)
    model_info["asr_dev"] = _param_device(asr)
    model_info["vad_dev"] = _param_device(vad)
    plog("model_jiazai", f"ASR({model_info['asr_dev']}) + VAD({model_info['vad_dev']}) 就绪, 耗时 {time.perf_counter() - t0:.1f}s")
    return asr, vad

def _load_mt():
    plog("model_jiazai", f"开始加载翻译模型 {MT_MODEL}")
    t0 = time.perf_counter()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mt_tok = AutoTokenizer.from_pretrained(MT_MODEL, trust_remote_code=True)
    try:
        mt_model = AutoModelForCausalLM.from_pretrained(
            MT_MODEL, dtype=torch.bfloat16, device_map={"": "cuda:0"}, trust_remote_code=True).eval()
    except (Exception,):
        mt_model = AutoModelForCausalLM.from_pretrained(
            MT_MODEL, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).eval()
    n_cpu = sum(1 for p in mt_model.parameters() if not p.is_cuda)
    model_info["mt_dev"] = (f"部分CPU! ({n_cpu} 个参数张量不在 GPU)" if n_cpu else "cuda:0 (bf16)")
    plog("model_jiazai", f"翻译模型就绪, 设备: {model_info['mt_dev']}, 耗时 {time.perf_counter() - t0:.1f}s")
    return mt_model, mt_tok

def _tts_gpu_bootstrap():
    try:
        import torch
        lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(lib):
            os.add_dll_directory(lib)
            os.environ["PATH"] = lib + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass
    try:
        import supertonic.config as _sc, supertonic.loader as _sl
        for mod in (_sc, _sl):
            if hasattr(mod, "DEFAULT_ONNX_PROVIDERS"):
                mod.DEFAULT_ONNX_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    except Exception:
        pass

def _load_tts():
    plog("model_jiazai", "开始加载语音合成 Supertonic")
    t0 = time.perf_counter()
    _tts_gpu_bootstrap()
    from supertonic import TTS
    tts = TTS(auto_download=True)
    try:
        provs = tts.model.dp_ort.get_providers()
        prov = "/".join(p for p in provs if p != "CPUExecutionProvider") or provs[0]
    except Exception:
        prov = "未知"
    model_info["tts_dev"] = prov
    plog("model_jiazai", f"语音合成就绪, 设备 {prov}, 耗时 {time.perf_counter() - t0:.1f}s")
    return tts

def load_models():
    plog("model_jiazai", f"开始并行加载三个模型 (python/{sys.version.split()[0]})")
    import torch
    model_info["gpu"] = torch.cuda.is_available()
    if model_info["gpu"]:
        model_info["gpu_name"] = torch.cuda.get_device_name(0)
        model_info["vram_total"] = torch.cuda.get_device_properties(0).total_memory / 2**30
        plog("model_jiazai", f"GPU: {model_info['gpu_name']} (总显存 {model_info['vram_total']:.1f} GiB)")
    else:
        plog("model_jiazai", "[警告] torch.cuda 不可用, 模型将全部跑在 CPU 上!")
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_av = ex.submit(_load_asr_and_vad)
        f_mt = ex.submit(_load_mt)
        f_tts = ex.submit(_load_tts)
        asr, vad = f_av.result()
        mt_model, mt_tok = f_mt.result()
        tts = f_tts.result()
    model_info["wall_s"] = time.perf_counter() - t0
    if model_info["gpu"]:
        model_info["vram_alloc"] = torch.cuda.memory_allocated() / 2**30
        plog("model_jiazai", f"三模型加载完成, 耗时 {model_info['wall_s']:.1f}s, 显存 {model_info['vram_alloc']:.2f}/{model_info['vram_total']:.1f} GiB")
    else:
        plog("model_jiazai", f"三模型加载完成(CPU), 耗时 {model_info['wall_s']:.1f}s")
    return asr, vad, mt_model, mt_tok, tts

def transcribe(asr, speech, tag="asr"):
    """中文识别。返回 (文本, 耗时, 标点列表)。

    标点列表为 [(文本末下标, 结束毫秒), ...]，按出现顺序排列。末下标用于切
    文本，结束毫秒用于按 ASR 逐字时间戳精确切音频。毫秒是相对本次输入音频
    起点的偏移，调用方需自行加上窗口偏移。
    """
    if model_info["mock"]:
        time.sleep(0.1)
        chars = int(len(speech) / SAMPLE_RATE * 5)
        text = "你好，这是模拟模式下的同声传译测试，边说边翻译，说完这句会自动收尾存档。"
        return text[:chars], 0.1, []
    from funasr.utils.postprocess_utils import rich_transcription_postprocess
    t0 = time.perf_counter()
    res = asr.generate(input=speech, language="zh", use_itn=True, batch_size_s=60,
                       output_timestamp=True)
    text = rich_transcription_postprocess(res[0]["text"]).strip()
    text = EMOJI_RE.sub("", text).strip()
    # 文本侧按顺序取每个标点的末下标；音频侧取带标点 token 的结束毫秒。
    # 两侧标点数量一致，按顺序 zip 即成 (文本切点, 音频切点) 配对。
    # token 侧用「末字符是标点」判断，兼容「。独立成词」与「不错。绑在一起」两种约定。
    text_end = [m.end() for m in re.finditer(f"[{PUNCT_CHARS}]", text)]
    words = res[0].get("words") or []
    ts = res[0].get("timestamp") or []
    ms_end = [t[1] for w, t in zip(words, ts) if w and w[-1] in PUNCT_CHARS]
    return text, record_cost(tag, t0), list(zip(text_end, ms_end))

def translate(mt_model, mt_tok, text):
    if model_info["mock"]:
        time.sleep(0.2)
        return "Hello, this is a mock streaming translation result.", 0.2
    import torch
    t0 = time.perf_counter()
    prompt = f"将以下文本翻译成英语,注意只需要输出翻译后的结果,不要额外解释:\n\n{text}"
    messages = [{"role": "user", "content": prompt}]
    inputs = mt_tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(mt_model.device)
    with torch.no_grad():
        out = mt_model.generate(**inputs, max_new_tokens=MT_MAX_NEW_TOKENS, do_sample=False)
    en = mt_tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    return EMOJI_RE.sub("", en).strip(), record_cost("mt", t0)

def synthesize(tts, style, text):
    t0 = time.perf_counter()
    if model_info["mock"]:
        time.sleep(0.1)
        n = int(max(0.8, min(3.0, 0.15 * len(text))) * SR_TTS)
        wav = (0.05 * np.sin(2 * np.pi * 220 * np.arange(n) / SR_TTS)).astype(np.float32).reshape(1, -1)
        return wav, n / SR_TTS, record_cost("tts", t0)
    wav, duration = tts.synthesize(text=text, lang=TTS_LANG, voice_style=style, total_steps=CFG["tts_steps"], speed=1.0)
    return wav, float(duration[0]), record_cost("tts", t0)

def split_clauses(text):
    return [p.strip() for p in re.split(r"(?<=[,.!?;:])\s+|(?<=[，。！？；：、])", text.strip()) if p.strip()]

def resample_to_16k(audio, orig_sr):
    if orig_sr == SAMPLE_RATE:
        return audio.astype(np.float32)
    n_out = int(round(len(audio) * SAMPLE_RATE / orig_sr))
    x_old = np.linspace(0.0, 1.0, len(audio), endpoint=False)
    x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)

class MicRecorder:
    def __init__(self, device=None):
        self.blocks = []
        self.level = 0.0
        self.rms_hist = deque(maxlen=300)
        self._stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=SAMPLE_RATE // 10, device=device, callback=self._on_audio)

    def _on_audio(self, indata, frames, time_info, status):
        self.level = float(np.sqrt(np.mean(indata[:, 0] ** 2)))
        self.rms_hist.append(self.level)
        self.blocks.append(indata[:, 0].copy())

    def start(self): self._stream.start()
    def stop(self):
        try:
            self._stream.stop()
            self._stream.close()
        except Exception: pass

    def audio(self):
        return np.concatenate(self.blocks) if self.blocks else np.empty(0, np.float32)

    def trim(self, keep_s):
        total = sum(len(b) for b in self.blocks)
        limit = int(keep_s * SAMPLE_RATE)
        while self.blocks and total > limit:
            total -= len(self.blocks[0])
            self.blocks.pop(0)

    def clear(self): self.blocks.clear()

def effective_thr(recorder):
    if len(recorder.rms_hist) >= 30:
        floor = sorted(recorder.rms_hist)[int(len(recorder.rms_hist) * 0.2)]
    else:
        floor = 0.0
    return max(CFG["speech_rms"], floor * 1.5)

def is_speech(seg, recorder):
    bs = SAMPLE_RATE // 10
    n = len(seg) // bs
    if n >= 1:
        peak = float(np.percentile(np.sqrt(np.mean(seg[:n * bs].reshape(n, bs) ** 2, axis=1)), 90))
    else:
        peak = float(np.sqrt(np.mean(np.asarray(seg) ** 2)))
    thr = effective_thr(recorder)
    return (len(seg) >= MIN_UTT_S * SAMPLE_RATE and peak >= thr), peak, thr

class TtsStreamer:
    def __init__(self, engine):
        self.engine = engine
        self.q = queue.Queue()
        self._stream = sd.OutputStream(samplerate=SR_TTS, channels=1, dtype="float32")
        self._started = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while True:
                item = self.q.get()
                try:
                    if item is None: break
                    if item[0] != "chunk": continue
                    if not self._started:
                        self._stream.start()
                        self._started = True
                        broadcast({"type": "speaking", "on": True})
                    self._stream.write(np.asarray(item[1], dtype=np.float32).reshape(-1, 1))
                    if self.q.empty():
                        broadcast({"type": "speaking", "on": False})
                finally:
                    self.q.task_done()
        except Exception as e:
            plog("service", f"[警告] 播放线程异常退出: {e}")

    def put_chunk(self, wav): self.q.put(("chunk", wav))

    def close(self, drain=True):
        if drain and self.thread.is_alive(): self.q.join()
        self.q.put(None)
        self.thread.join(timeout=5)
        try:
            self._stream.stop()
            self._stream.close()
        except Exception: pass

# ---------------- 同传会话（滑动窗口版） ----------------
class Session:
    """不再切断喂给 ASR 的上下文，保留完整音频，仅在找到切点时弹出已完结部分。"""
    def __init__(self, engine, first_seg):
        self.engine = engine
        self.lock = threading.Lock()
        self.blocks = [first_seg]       # 始终保留未完结的音频上下文
        self.k = 0                      # 本会话已切出的语义段数
        self.closing = False
        self.last_detect = time.perf_counter()   # test_replay 直接驱动 detect_once 时用
        self.prev_punct_ms = []         # 上一轮识别的标点绝对毫秒（稳定判定用）
        self.t0 = time.perf_counter()

    def absorb(self, rec):
        if rec is not None and rec.blocks:
            with self.lock:
                self.blocks.extend(rec.blocks)
                rec.blocks = []

    def seconds(self):
        with self.lock:
            return sum(len(b) for b in self.blocks) / SAMPLE_RATE

    def audio(self):
        with self.lock:
            return np.concatenate(self.blocks) if self.blocks else np.empty(0, np.float32)

    def pop_front_samples(self, samples):
        """从缓冲区头部弹出指定采样点（切分后送翻译的部分）。"""
        with self.lock:
            popped = 0
            while self.blocks and popped + len(self.blocks[0]) <= samples:
                popped += len(self.blocks.pop(0))
            if popped < samples and self.blocks:
                self.blocks[0] = self.blocks[0][samples - popped:]

    def trailing_silence_ms(self, vad):
        audio = self.audio()
        if len(audio) == 0: return 0
        res = vad.generate(input=audio)
        segs = (res or [{}])[0].get("value") or []
        if not segs:
            return int(len(audio) * 1000 / SAMPLE_RATE)
        return max(0, int(len(audio) * 1000 / SAMPLE_RATE - segs[-1][1]))

    def take_all(self):
        with self.lock:
            blocks, self.blocks = self.blocks, []
            return blocks

INCOMPLETE_ENDINGS = [
    "因为", "虽然", "但是", "不过", "而且", "并且", "或者", "如果", "假如",
    "即使", "尽管", "无论", "不管", "只要", "只有", "除了", "关于", "对于",
    "比如", "例如", "像", "以及", "还有", "另外", "此外", "同时",
    "正在", "准备", "打算", "计划", "想要", "需要", "可能", "应该",
    "第一", "第二", "第三", "首先", "其次", "最后", "然后", "接着", "于是",
    "总之", "总的来说", "换句话说", "也就是说", "就是说",
    "我觉得", "我认为", "他说", "她说", "他们说", "意思是"
]

class Engine:
    def __init__(self):
        self.asr = self.vad = self.mt_model = self.mt_tok = self.tts = None
        self.recorder = None
        self.streamer = None
        self.running = False
        self.session_dir = None
        self.sess = None
        self.n = 0
        self.n_seg = 0
        self._drops = 0                 # 连续丢弃的噪声段数（限频日志用）
        self._mt_q = queue.Queue()
        self._tts_q = queue.Queue()
        self._styles = {}

    def get_style(self):
        if model_info["mock"]: return None
        if CFG["voice"] not in self._styles:
            self._styles[CFG["voice"]] = self.tts.get_voice_style(voice_name=CFG["voice"])
        return self._styles[CFG["voice"]]

    def ensure_session(self):
        if self.session_dir: return
        self.session_dir = os.path.join(OUTPUT_ROOT, datetime.now().strftime("web_%Y%m%d_%H%M%S"))
        os.makedirs(self.session_dir, exist_ok=True)
        with open(os.path.join(self.session_dir, "transcript.txt"), "w", encoding="utf-8") as f:
            f.write(f"===== web session {datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
        plog("service", f"存档目录: {self.session_dir}")

    def try_open_session(self, rec):
        audio = rec.audio()
        if len(audio) < int(0.6 * SAMPLE_RATE): return
        if len(audio) > NOISE_BUF_CLEAR_S * SAMPLE_RATE and not model_info["mock"]:
            rec.trim(NOISE_BUF_CLEAR_S)
            audio = rec.audio()
        res = self.vad.generate(input=audio)
        segs = (res or [{}])[0].get("value") or []
        if not segs: return
        a = max(0, int(segs[0][0] * SAMPLE_RATE / 1000) - int(0.3 * SAMPLE_RATE))
        b = min(len(audio), int(segs[-1][1] * SAMPLE_RATE / 1000))
        seg = audio[a:b]
        if model_info["mock"]:
            ok, rms, thr = True, 0.0, 0.0
        else:
            ok, rms, thr = is_speech(seg, rec)
        if not ok:
            self._drops += 1
            if self._drops == 1 or self._drops % 10 == 0:
                plog("service", f"丢弃噪声段 ({len(seg) / SAMPLE_RATE:.1f}s, "
                                f"峰值 {rms:.4f} < 门限 {thr:.4f}), 已丢弃 {self._drops} 段")
            return
        if self._drops:
            plog("service", f"有效语音恢复 (峰值 {rms:.4f}), 此前丢弃 {self._drops} 段")
            self._drops = 0
        self.ensure_session()
        self.n += 1
        self.sess = Session(self, seg)
        rec.clear()
        plog("record_log", f"句#{self.n} | 会话开始 ({len(seg) / SAMPLE_RATE:.1f}s 语音, "
                           f"峰值 {rms:.4f} >= 门限 {thr:.4f}), 滑动窗口同传中")
        broadcast({"type": "sess_open", "n": self.n})

    # def maybe_close_session(self):
    #     sess = self.sess
    #     if sess.closing or sess.seconds() < 0.1: return
    #     silence_ms = sess.trailing_silence_ms(self.vad)
    #     if silence_ms >= CFG["tail_silence"] * 1000:
    #         sess.closing = True
    #         blocks = sess.take_all()
    #         plog("service", f"句#{self.n} 语音结束 (剩余 {sess.seconds():.1f}s, 尾音 {silence_ms}ms), 冲洗最后一段")
    #         self._mt_q.put({"type": "flush", "n": self.n, "k": sess.k + 1,
    #                         "audio": np.concatenate(blocks) if blocks else np.empty(0, np.float32)})
    #         self.sess = None
    def maybe_close_session(self):
        """说完（尾部静音/超长）→ 排队收尾。加入短句动态加速逻辑。"""
        sess = self.sess
        if sess.closing or sess.seconds() < 0.1:
            return
        
        silence_ms = sess.trailing_silence_ms(self.vad)
        
        # === 核心优化：动态尾音阈值 ===
        # 如果当前会话从未切分过（sess.k == 0），说明是单口短句，等 0.6s 即可结束。
        # 如果已经切过分了（sess.k > 0），说明是长篇大论，保持 1.5s 防误切。
        if sess.k == 0:
            dynamic_tail_silence = min(CFG["tail_silence"], 0.6)
        else:
            dynamic_tail_silence = CFG["tail_silence"]

        # 判定是否触发结束
        if silence_ms >= dynamic_tail_silence * 1000:
            sess.closing = True
            blocks = sess.take_all()
            plog("service", f"句#{self.n} 语音结束 (剩余 {sess.seconds():.1f}s, 尾音 {silence_ms}ms, 阈值 {dynamic_tail_silence}s), 冲洗最后一段")
            self._mt_q.put({"type": "flush", "n": self.n, "k": sess.k + 1,
                            "audio": np.concatenate(blocks) if blocks else np.empty(0, np.float32)})
            self.sess = None

    # ---- 语义切分（滑动窗口版） ----
    def detect_once(self, sess):
        """取完整未翻译音频跑 ASR，在「连续两轮都出现的标点」上切分。

        切分不再等 VAD 静音，而是比对相邻两轮识别的标点时间戳：同一处标点
        连续出现（毫秒差在 STABLE_TOL_MS 内）即认定稳定，立即切出。半截音频
        上 ASR 随手补的伪标点会被下一轮全窗识别改写掉，因此不会被误切。
        """
        if sess.seconds() < SEG_MIN_S:
            return

        # 1. 获取完整音频（限制最后 15s 防爆显存）跑 ASR
        audio = sess.audio()
        if len(audio) > int(MAX_ASR_AUDIO_S * SAMPLE_RATE):
            audio_win = audio[-int(MAX_ASR_AUDIO_S * SAMPLE_RATE):]
        else:
            audio_win = audio
        offset_ms = (len(audio) - len(audio_win)) * 1000.0 / SAMPLE_RATE

        zh, cost, puncts = transcribe(self.asr, audio_win, tag="asr_detect")
        valid_chars = len(re.findall(r'[一-鿿a-zA-Z0-9]', zh))
        if valid_chars < 2:
            return

        overlong = sess.seconds() > OVERLONG_S
        # 2. 标点换算成 (文本末下标, 绝对毫秒)：时间戳相对 audio_win，需平移窗口偏移
        cands = [(idx, offset_ms + ms) for idx, ms in puncts]

        # 3. 稳定判定：取本轮与上一轮都出现过的最后一个标点（cands 有序，取到即最新）
        confirmed = None
        for idx, ms in cands:
            if any(abs(ms - prev) <= STABLE_TOL_MS for prev in sess.prev_punct_ms):
                confirmed = (idx, ms)
        sess.prev_punct_ms = [ms for _, ms in cands]

        # 4. 无稳定标点：超长句且完全无标点时兜底硬切（按 80% 比例）
        if confirmed is None:
            # if overlong and not cands and valid_chars > 20:
            #     cut_sample = int(len(audio) * 0.8)
            #     seg_audio = audio[:cut_sample]
            #     sess.pop_front_samples(cut_sample)
            #     sess.prev_punct_ms = []
            #     self.n_seg += 1
            #     sess.k += 1
            #     plog("service", f"句#{self.n} 超长无标点, 强制切分: {zh}")
            #     self._mt_q.put({"type": "seg", "n": self.n, "k": sess.k, "zh": zh, "audio": seg_audio})

            # 修改后：按词级时间戳软切
            if overlong and not cands and valid_chars > 20:
                # 1. 目标切点：总音频的 80% 处对应的毫秒数
                target_ms = len(audio) * 0.8 * 1000 / SAMPLE_RATE
                
                # 2. 取出 ASR 返回的所有词的时间戳 (即使没有标点也有词时间戳)
                words = res[0].get("words") or []
                ts = res[0].get("timestamp") or []
                
                # 3. 找到结束时间刚好不超过 target_ms 的最后一个词
                valid_word_end_ms = [t[1] for w, t in zip(words, ts) if t[1] <= target_ms]
                
                if valid_word_end_ms:
                    # 找到了合适的词边界，按这个词的结尾精确切分
                    punct_ms = valid_word_end_ms[-1]
                    cut_sample = int((offset_ms + punct_ms) * SAMPLE_RATE / 1000)
                    final_text = zh[:words[valid_word_end_ms.index(valid_word_end_ms[-1])][0] if False else len(zh)] # 这里需处理文本切分
                    # 为了简化，这里依然把整段 zh 送翻译，但音频按词边界精确切
                else:
                    # 极端情况：80%处连一个完整的词都没有（比如还在念一个极长的化学名词），退回硬切
                    cut_sample = int(len(audio) * 0.8)
                    
                seg_audio = audio[:cut_sample]
                sess.pop_front_samples(cut_sample)
                sess.prev_punct_ms = []
                self.n_seg += 1
                sess.k += 1
                plog("service", f"句#{self.n} 超长无标点, 按词边界软切: {zh}")
                self._mt_q.put({"type": "seg", "n": self.n, "k": sess.k, "zh": zh, "audio": seg_audio})

            return

        # 5. 提取已完结文本并进行保护检查
        punct_idx, punct_ms = confirmed
        final_text = zh[:punct_idx].strip()

        zh_clean = final_text.rstrip("。！？.!?；;…，、 ")
        if any(zh_clean.endswith(w) for w in INCOMPLETE_ENDINGS):
            return

        if len(final_text) < MIN_COMMA_CHARS:
            return

        # 6. 用已确认标点的绝对毫秒定位切点，精确对齐词边界
        cut_sample = min(int(punct_ms * SAMPLE_RATE / 1000), len(audio))

        # 7. 物理切分：弹出已完结音频，剩余留作上下文；稳定状态随缓冲区位移复位
        seg_audio = audio[:cut_sample]
        sess.pop_front_samples(cut_sample)
        sess.prev_punct_ms = []

        self.n_seg += 1
        sess.k += 1
        plog("record_log", f'句#{self.n} | 段{sess.k} 稳定切分 ({cost:.2f}s 检测, 本轮 {len(cands)} 标点, 切点 {cut_sample/SAMPLE_RATE:.1f}s): "{final_text}"')
        self._mt_q.put({"type": "seg", "n": self.n, "k": sess.k, "zh": final_text, "audio": seg_audio})

    def mt_handle(self, job):
        if job["type"] == "seg":
            en, cost = translate(self.mt_model, self.mt_tok, job["zh"])
            plog("record_log", f'句#{job["n"]} | 段{job["k"]} 整段翻译 {cost:.2f}s: "{en}"')
            self._tts_q.put({"type": "tts", "n": job["n"], "k": job["k"], "zh": job["zh"], "en": en, "audio": job["audio"]})
        else:
            audio = job["audio"]
            if len(audio) < int(0.3 * SAMPLE_RATE):
                plog("service", f"句#{job['n']} 最后一段过短, 忽略")
                if job["k"] == 1:
                    self.n -= 1
                    broadcast({"type": "sess_cancel", "n": self.n + 1})
                return
            zh, cost, _ = transcribe(self.asr, audio)
            valid_chars = len(re.findall(r'[一-鿿a-zA-Z0-9]', zh))
            if valid_chars < 2:
                plog("record_log", f"句#{job['n']} | 最后一段有效字符不足, 忽略防幻觉")
                if job["k"] == 1:
                    self.n -= 1
                    broadcast({"type": "sess_cancel", "n": self.n + 1})
                return
            plog("record_log", f'句#{job["n"]} | 段{job["k"]} 收尾识别 {cost:.2f}s: "{zh}"')
            en, cost_mt = translate(self.mt_model, self.mt_tok, zh)
            plog("record_log", f'句#{job["n"]} | 段{job["k"]} 整段翻译 {cost_mt:.2f}s: "{en}"')
            self._tts_q.put({"type": "tts", "n": job["n"], "k": job["k"], "zh": zh, "en": en, "audio": audio})

    def tts_handle(self, job):
        frags = split_clauses(job["en"])
        gap = np.zeros((int(STREAM_GAP_S * SR_TTS), 1), dtype=np.float32)
        wavs = []
        cost_tts = 0.0
        for i, frag in enumerate(frags):
            wav, dur, cost = synthesize(self.tts, self.get_style(), frag)
            cost_tts += cost
            wavs.append(wav)
            piece = np.concatenate([wav.squeeze(axis=0).reshape(-1, 1), gap if i < len(frags) - 1 else np.zeros((0, 1), np.float32)])
            if self.streamer:
                self.streamer.put_chunk(piece)
        fname = f"utt_{job['n']:04d}s{job['k']:02d}.wav"
        import soundfile as sf
        sf.write(os.path.join(self.session_dir, f"utt_{job['n']:04d}s{job['k']:02d}_in.wav"), job["audio"], SAMPLE_RATE)
        combined = np.concatenate(wavs, axis=1) if wavs else np.zeros((1, 0), np.float32)
        self.tts.save_audio(combined, os.path.join(self.session_dir, fname))
        with open(os.path.join(self.session_dir, "transcript.txt"), "a", encoding="utf-8") as f:
            f.write(f"[句#{job['n']}-段{job['k']}] 你说: {job['zh']}\n[句#{job['n']}-段{job['k']}] 译文: {job['en']}\n[句#{job['n']}-段{job['k']}] {fname}\n\n")
        plog("record_log", f"句#{job['n']} | 段{job['k']} 完成: 合成 {cost_tts:.2f}s, 播放 {combined.shape[1] / SR_TTS:.1f}s, 存档 {fname} (+{len(frags)} 段)")
        broadcast({"type": "seg", "n": job["n"], "k": job["k"], "zh": job["zh"], "en": job["en"], "mt": 0, "tts": round(cost_tts, 2), "dur": round(combined.shape[1] / SR_TTS, 1), "frags": len(frags), "wav": f"{os.path.basename(self.session_dir)}/{fname}"})

    def start_capture(self):
        if self.running: return
        if model_info["phase"] != "ready": raise RuntimeError("模型尚未就绪")
        self.ensure_session()
        self.recorder = MicRecorder(device=CFG["device"])
        self.recorder.start()
        self.streamer = TtsStreamer(self)
        self.running = True
        plog("service", f"麦克风采集开始 (设备={'默认' if CFG['device'] is None else CFG['device']}), 检测 {STREAM_POLL_S}s, 断句 {CFG['tail_silence']}s")
        threading.Thread(target=self._vad_loop, daemon=True).start()
        threading.Thread(target=self._detect_loop, daemon=True).start()
        threading.Thread(target=self._mt_worker, daemon=True).start()
        threading.Thread(target=self._tts_worker, daemon=True).start()
        threading.Thread(target=self._level_loop, daemon=True).start()
        broadcast_state()

    def stop_capture(self):
        if not self.running: return
        self.running = False
        if self.recorder: self.recorder.stop()
        plog("service", f"麦克风采集停止, 共 {self.n} 个会话 / {self.n_seg} 个语义段")
        threading.Thread(target=self._shutdown, daemon=True).start()
        broadcast_state()

    def _shutdown(self):
        if self.sess and not self.sess.closing:
            self.sess.closing = True
            blocks = self.sess.take_all()
            self._mt_q.put({"type": "flush", "n": self.n, "k": self.sess.k + 1, "audio": np.concatenate(blocks) if blocks else np.empty(0, np.float32)})
            self.sess = None
        self._mt_q.put(None)
        self._mt_q.join()
        self._tts_q.put(None)
        self._tts_q.join()
        if self.streamer: self.streamer.close(drain=True)
        self.recorder = self.streamer = None

    def _vad_loop(self):
        while self.running:
            time.sleep(VAD_POLL_S)
            rec = self.recorder
            if rec is None: break
            try:
                if self.sess is None:
                    self.try_open_session(rec)
                else:
                    self.sess.absorb(rec)
                    self.maybe_close_session()
            except Exception as e:
                plog("service", f"[警告] 会话监测出错: {e!r}")

    def _detect_loop(self):
        while self.running:
            time.sleep(STREAM_POLL_S)
            sess = self.sess
            if sess is None or sess.closing: continue
            try:
                self.detect_once(sess)
            except Exception as e:
                plog("service", f"[警告] 滑动窗口切分出错: {e!r}")

    def _mt_worker(self):
        while True:
            job = self._mt_q.get()
            try:
                if job is None: break
                self.mt_handle(job)
            except Exception as e:
                plog("record_log", f"翻译任务出错: {e!r}")
                broadcast({"type": "utt_error", "msg": str(e)})
            finally:
                self._mt_q.task_done()

    def _tts_worker(self):
        while True:
            job = self._tts_q.get()
            try:
                if job is None: break
                self.tts_handle(job)
            except Exception as e:
                plog("record_log", f"合成任务出错: {e!r}")
                broadcast({"type": "utt_error", "msg": str(e)})
            finally:
                self._tts_q.task_done()

    def _level_loop(self):
        while self.running and self.recorder is not None:
            time.sleep(LEVEL_POLL_S)
            if self.recorder is not None:
                thr = effective_thr(self.recorder)
                broadcast({"type": "level", "rms": round(self.recorder.level, 4), "speech": self.recorder.level > thr})

engine = Engine()

def status_event():
    st = {k: v for k, v in model_info.items()}
    st.update({"type": "state", "running": engine.running, "config": dict(CFG), "session": os.path.basename(engine.session_dir) if engine.session_dir else None, "n_utts": engine.n_seg, "stats": {k: [round(v[0], 2), v[1]] for k, v in record_cost_stats.items()}})
    return st

def broadcast_state(): broadcast(status_event())

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _loop
    _loop = asyncio.get_running_loop()
    broadcast_state()
    yield

app = FastAPI(title="实时语音翻译 Demo · 滑动窗口", lifespan=_lifespan)
HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_web_clause.html")

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    _clients.add(ws)
    try:
        await ws.send_text(json.dumps(status_event(), ensure_ascii=False))
        while True: await ws.receive_text()
    except (WebSocketDisconnect, Exception): pass
    finally: _clients.discard(ws)

@app.get("/")
def index(): return FileResponse(HTML_PATH)

@app.get("/api/status")
def api_status(): return status_event()

@app.post("/api/settings")
async def api_settings(body: dict):
    if "tail_silence" in body: CFG["tail_silence"] = min(2.0, max(0.2, float(body["tail_silence"])))
    if "speech_rms" in body: CFG["speech_rms"] = min(0.08, max(0.003, float(body["speech_rms"])))
    plog("service", f"设置更新: 断句停顿={CFG['tail_silence']}s, 噪声门限={CFG['speech_rms']}")
    return {"ok": True, "config": dict(CFG)}

@app.post("/api/mic/start")
def api_mic_start():
    try:
        engine.start_capture()
        return {"ok": True}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

@app.post("/api/mic/stop")
def api_mic_stop():
    engine.stop_capture()
    return {"ok": True}

app.mount("/audio", StaticFiles(directory=OUTPUT_ROOT), name="audio")

class MockVad:
    def generate(self, input=None, **kw):
        n, ms = len(input), int(len(input) * 1000 / SAMPLE_RATE)
        if n < int(3.0 * SAMPLE_RATE): return [{"value": []}]
        if n < int(6.0 * SAMPLE_RATE): return [{"value": [[0, ms]]}]
        return [{"value": [[0, ms - 800]]}]

class MockTTS:
    def save_audio(self, wav, path):
        import soundfile as sf
        sf.write(path, np.asarray(wav, dtype=np.float32).reshape(-1, 1), SR_TTS)
    def get_voice_style(self, voice_name=None): return None

def install_mock_models():
    model_info["mock"] = True
    model_info.update({"gpu": False, "asr_dev": "mock", "vad_dev": "mock", "mt_dev": "mock", "tts_dev": "mock", "wall_s": 0.0, "warmup_s": 0.0})
    engine.asr, engine.vad = object(), MockVad()
    engine.mt_model = engine.mt_tok = None
    engine.tts = MockTTS()
    model_info["phase"] = "ready"
    plog("model_jiazai", "MOCK 模式：跳过真实模型加载，使用假数据流水线")

def parse_args():
    p = argparse.ArgumentParser(description="中文语音 → 英文语音 滑动窗口同传 Web 服务")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8327)
    p.add_argument("--mic", type=int, default=None)
    p.add_argument("--tts-steps", type=int, default=CFG["tts_steps"])
    p.add_argument("--voice", default=CFG["voice"])
    p.add_argument("--pause-ms", type=int, default=None)
    p.add_argument("--no-auto-start", action="store_true")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--list-devices", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    if args.list_devices:
        print(sd.query_devices())
        return
    CFG.update(tts_steps=args.tts_steps, voice=args.voice, device=args.mic)
    if args.pause_ms is not None:
        global CUT_PAUSE_MS
        CUT_PAUSE_MS = args.pause_ms

    print(f">>> 服务日志: {os.path.abspath(LOG_PATH)}", flush=True)
    plog("service", f"===== 服务启动 pid={os.getpid()} port={args.port} {'MOCK' if args.mock else '真实模型'} =====")

    def load_and_ready():
        try:
            if args.mock:
                install_mock_models()
            else:
                asr, vad, mt_model, mt_tok, tts = load_models()
                engine.asr, engine.vad = asr, vad
                engine.mt_model, engine.mt_tok, engine.tts = mt_model, mt_tok, tts
                t0 = time.perf_counter()
                engine.get_style()
                noise = (np.random.randn(SAMPLE_RATE // 2) * 0.01).astype(np.float32)
                transcribe(engine.asr, noise)
                translate(engine.mt_model, engine.mt_tok, "你好。")
                synthesize(engine.tts, engine.get_style(), "Hello.")
                record_cost_stats.clear()
                model_info["warmup_s"] = time.perf_counter() - t0
                plog("model_jiazai", f"预热完成 {model_info['warmup_s']:.1f}s, 模型常驻")
            model_info["phase"] = "ready"
            broadcast_state()
            print(f">>> 模型就绪，浏览器打开 http://127.0.0.1:{args.port}", flush=True)
            if not args.no_auto_start:
                engine.start_capture()
        except Exception as e:
            model_info["phase"] = "error"
            model_info["error"] = repr(e)
            plog("service", f"[错误] 模型加载失败: {e!r}")
            broadcast_state()

    import uvicorn
    threading.Thread(target=load_and_ready, daemon=True).start()
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        if engine.running: engine.stop_capture()
        plog("service", "===== 服务退出 =====")

if __name__ == "__main__":
    main()
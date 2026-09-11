# -*- coding: utf-8 -*-
"""
实时语音翻译 Demo · Web 版：中文语音输入 → 英文语音同传（整句识别翻译，输出流式 · 静音断句基线）
====================================================================
在 test_demo_copy.py 基础上改为"常驻服务 + 浏览器界面 + 同传输出"：

    【对比基线版】断句只靠 停顿超时 / 15s 超长，无标点检测。
    标点断句版见 demo_web_asr_full.py (端口 8322)。

    模型常驻    启动即在后台线程并行加载三模型并预热，之后常驻显存，
                每次调用零加载等待。
    整句同传    检测到说话立即"开会话"，说话期间不做识别；停顿超过断句
                停顿（或超长）判句完后，整句一次识别、一次翻译（上下文
                完整，质量最好），译文再按标点切段流式合成、连续播放。
    噪声门限    开会话要过两道确认：VAD 报有语音段 + 段内能量(RMS)超过
                门限（绝对门限 UI 可调 + 自适应噪声底×1.5 + 最短时长 0.5s）。
                环境噪声连会话都开不起来，更不会送识别。
    全双工      始终开启：播放译文期间麦克风照常收音，可以边听边说下一句
                （按戴耳机设计，不做外放半双工防回声）。

日志（单一全局日志 logs/service_log.txt，行首 tag 便于 grep）：
    [record_log]   每次会话的完整过程（开会话/整句识别/整句翻译/合成/存档）
    [model_jiazai] 模型加载与预热
    [service]      其余服务事件（启动/开麦停麦/丢弃噪声/异常）

运行：
    conda activate demo
    python demo_web_asr_silence.py        # 端口 8324，就绪后自动开麦
    python demo_web_asr_silence.py --mic 1    # 指定麦克风
    python demo_web_asr_silence.py --mock     # 不加载模型，假数据联调界面/日志

退出：Ctrl+C；存档：demo_output/web_时间戳/（每句 识别输入.wav + 合成.wav + transcript.txt）
"""

import os
# 必须在 import transformers/huggingface_hub 之前设置：跳过联网版本校验，用本地缓存。
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
SAMPLE_RATE = 16000        # ASR / VAD 工作采样率
SR_TTS = 44100             # Supertonic 输出采样率
MAX_UTT_S = 15.0           # 单次会话最长强制收尾（一直不停说时切下一句）
MIN_UTT_S = 0.5            # 语音段短于此视为噪声（咳嗽/碰撞声），不开会话
VAD_POLL_S = 0.3           # 会话开/关检测间隔
LEVEL_POLL_S = 0.1         # 电平推送间隔
STREAM_GAP_S = 0.12        # 段间补的静音
MIN_FRAG_WORDS = 2         # 碎片过短阈值（英文词数）
MIN_FRAG_CJK = 4           # 碎片过短阈值（中文字数）
ASR_DEVICE = "cuda:0"
VAD_DEVICE = "cpu"         # fsmn-vad 很小，放 CPU 避免抢显存
MT_MODEL = "tencent/Hy-MT2-1.8B"
MT_MAX_NEW_TOKENS = 512
TTS_LANG = "en"            # 本 demo 固定 中→英
OUTPUT_ROOT = "demo_output"
LOG_PATH = os.path.join("logs", "service_log.txt")

# 运行期配置（前端可实时改 tail_silence / speech_rms，其余由命令行指定）
CFG = {
    "tail_silence": 0.6,   # 停顿多久判定说完了（收尾）
    "speech_rms": 0.003,   # 语音能量门限：低于它视为环境噪声，UI 可调
                           # （麦克风增益差异很大，默认值取低，主要靠信噪比挡噪声）
    "tts_steps": 6,        # 合成质量 5(低,最快)~12(高,最慢)
    "voice": "M1",         # Supertonic 预置音色
    "device": None,        # 麦克风编号，None=系统默认
}


# ---------------- 全局日志（行首 [tag] 便于 grep） ----------------
_log_lock = threading.Lock()


def plog(tag, msg):
    """追加一条全局日志并推给前端日志面板。tag: record_log / model_jiazai / service。"""
    line = f"[{tag}] [{datetime.now():%Y-%m-%d %H:%M:%S.%f}] {msg}"
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with _log_lock:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    broadcast({"type": "log", "line": line})


# ---------------- record_cost 耗时统计 ----------------
record_cost_stats = {}  # {阶段: [累计秒, 次数]}


def record_cost(stage, t0):
    cost = time.perf_counter() - t0
    acc = record_cost_stats.setdefault(stage, [0.0, 0])
    acc[0] += cost
    acc[1] += 1
    return cost


# ---------------- WebSocket 广播（推理线程 → 浏览器） ----------------
_clients = set()   # 已连接的浏览器
_loop = None       # uvicorn 事件循环，startup 时赋值


def broadcast(event: dict):
    """线程安全推给所有浏览器；loop 未起/无连接时丢弃。"""
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


# ---------------- 模型加载（并行，常驻显存） ----------------
def build_funasr(**kw):
    """加载 funasr 模型；新版关闭进度条/版本检查，旧版参数自动忽略。"""
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


model_info = {"phase": "loading", "mock": False}   # loading → ready / error


def _load_asr_and_vad():
    plog("model_jiazai", "开始加载 ASR SenseVoiceSmall + VAD fsmn-vad")
    t0 = time.perf_counter()
    from modelscope import snapshot_download
    asr_dir = snapshot_download("iic/SenseVoiceSmall")
    # 不挂内部 vad_model：断句/开关会话由独立 fsmn-vad 完成，喂进来的已是
    # 切好的短句，省显存省时间。（funasr 的 fp16=True 有 dtype 不一致 bug，勿开。）
    asr = build_funasr(model=asr_dir, device=ASR_DEVICE)
    vad = build_funasr(model="fsmn-vad", device=VAD_DEVICE)
    model_info["asr_dev"] = _param_device(asr)
    model_info["vad_dev"] = _param_device(vad)
    plog("model_jiazai", f"ASR({model_info['asr_dev']}) + VAD({model_info['vad_dev']}) 就绪, "
                         f"耗时 {time.perf_counter() - t0:.1f}s (与翻译/合成并行)")
    return asr, vad


def _load_mt():
    plog("model_jiazai", f"开始加载翻译模型 {MT_MODEL}")
    t0 = time.perf_counter()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mt_tok = AutoTokenizer.from_pretrained(MT_MODEL, trust_remote_code=True)
    # device_map="auto" 在 8GB 显存 + 并行加载时会把大部分层卸载到 CPU，生成暴跌。
    # 强制整个模型进 GPU，OOM 再退回 auto。
    try:
        mt_model = AutoModelForCausalLM.from_pretrained(
            MT_MODEL, dtype=torch.bfloat16,
            device_map={"": "cuda:0"},   # 字符串而非整数：新版 torch 对 int 设备解析报错
            trust_remote_code=True,
        ).eval()
    except (Exception,):
        mt_model = AutoModelForCausalLM.from_pretrained(
            MT_MODEL, dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True,
        ).eval()
    n_cpu = sum(1 for p in mt_model.parameters() if not p.is_cuda)
    model_info["mt_dev"] = (f"部分CPU! ({n_cpu} 个参数张量不在 GPU, 生成会大幅变慢)" if n_cpu
                            else "cuda:0 (bf16)")
    plog("model_jiazai", f"翻译模型就绪, 设备: {model_info['mt_dev']}, "
                         f"耗时 {time.perf_counter() - t0:.1f}s")
    return mt_model, mt_tok


def _load_tts():
    plog("model_jiazai", "开始加载语音合成 Supertonic")
    t0 = time.perf_counter()
    from supertonic import TTS
    tts = TTS(auto_download=True)
    try:
        import onnxruntime as ort
        prov = ort.get_available_providers()
    except Exception:
        prov = ["未知"]
    model_info["tts_dev"] = f"CPU (supertonic 包固定 CPU, ORT: {prov})"
    plog("model_jiazai", f"语音合成就绪, 耗时 {time.perf_counter() - t0:.1f}s")
    return tts


def load_models():
    """并行加载三模型，全部常驻。返回 (asr, vad, mt_model, mt_tok, tts)。"""
    plog("model_jiazai", f"开始并行加载三个模型 (python/{sys.version.split()[0]})")
    import torch
    model_info["gpu"] = torch.cuda.is_available()
    if model_info["gpu"]:
        model_info["gpu_name"] = torch.cuda.get_device_name(0)
        model_info["vram_total"] = torch.cuda.get_device_properties(0).total_memory / 2**30
        plog("model_jiazai", f"GPU: {model_info['gpu_name']} "
                             f"(总显存 {model_info['vram_total']:.1f} GiB)")
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
        plog("model_jiazai", f"三模型全部加载完成, 总耗时 {model_info['wall_s']:.1f}s, "
                             f"显存 已分配 {model_info['vram_alloc']:.2f} / "
                             f"共 {model_info['vram_total']:.1f} GiB")
    else:
        plog("model_jiazai", f"三模型全部加载完成(CPU 模式), 总耗时 {model_info['wall_s']:.1f}s")
    return asr, vad, mt_model, mt_tok, tts


# ---------------- 三段流水线（自带 record_cost 耗时统计） ----------------
def transcribe(asr, speech):
    """中文识别。返回 (文本, 耗时)。mock 模式返回随音频时长增长的假文本，模拟流式。"""
    if model_info["mock"]:
        time.sleep(0.1)
        chars = int(len(speech) / SAMPLE_RATE * 5)   # 5 字/秒
        text = "你好，这是模拟模式下的同声传译测试，边说边翻译，说完这句会自动收尾存档。"
        return text[:chars], 0.1
    from funasr.utils.postprocess_utils import rich_transcription_postprocess
    t0 = time.perf_counter()
    res = asr.generate(input=speech, language="zh", use_itn=True, batch_size_s=60)
    text = rich_transcription_postprocess(res[0]["text"]).strip()
    return text, record_cost("asr", t0)


def translate(mt_model, mt_tok, text):
    """中译英。返回 (文本, 耗时)。"""
    if model_info["mock"]:
        time.sleep(0.2)
        return "Hello, this is a mock streaming translation result. Have a nice day.", 0.2
    import torch
    t0 = time.perf_counter()
    prompt = f"将以下文本翻译成英语,注意只需要输出翻译后的结果,不要额外解释:\n\n{text}"
    messages = [{"role": "user", "content": prompt}]
    inputs = mt_tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(mt_model.device)
    with torch.no_grad():
        out = mt_model.generate(**inputs, max_new_tokens=MT_MAX_NEW_TOKENS, do_sample=False)
    en = mt_tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    return en, record_cost("mt", t0)


def synthesize(tts, style, text):
    """英文合成。返回 (wav(1,N), 时长, 耗时)。"""
    t0 = time.perf_counter()
    if model_info["mock"]:
        time.sleep(0.1)
        n = int(max(0.8, min(3.0, 0.15 * len(text))) * SR_TTS)
        wav = (0.05 * np.sin(2 * np.pi * 220 * np.arange(n) / SR_TTS)).astype(np.float32).reshape(1, -1)
        return wav, n / SR_TTS, record_cost("tts", t0)
    wav, duration = tts.synthesize(text=text, lang=TTS_LANG, voice_style=style,
                                   total_steps=CFG["tts_steps"], speed=1.0)
    return wav, float(duration[0]), record_cost("tts", t0)


def split_by_punct(text):
    """按标点把译文切成短段（伪流式）；过短碎片并入相邻段。"""
    parts = [p.strip() for p in re.split(
        r"(?<=[,.!?;:])\s+|(?<=[，。！？；：])", text.strip()) if p.strip()]
    if len(parts) <= 1:
        return parts

    def too_short(p):
        return len(p.split()) < MIN_FRAG_WORDS and len(re.findall(r"[一-鿿]", p)) < MIN_FRAG_CJK

    merged = [parts[0]]
    for p in parts[1:]:
        if too_short(p) or too_short(merged[-1]):
            merged[-1] = f"{merged[-1]} {p}"
        else:
            merged.append(p)
    return merged


def resample_to_16k(audio, orig_sr):
    """任意采样率 → 16kHz 单声道（线性插值，测试脚本也用）。"""
    if orig_sr == SAMPLE_RATE:
        return audio.astype(np.float32)
    n_out = int(round(len(audio) * SAMPLE_RATE / orig_sr))
    x_old = np.linspace(0.0, 1.0, len(audio), endpoint=False)
    x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


class MicRecorder:
    """后台持续采集麦克风。
    始终全双工：播放期间照常收音，可以边听边说（按戴耳机设计）。
    """

    def __init__(self, device=None):
        self.blocks = []
        self.level = 0.0                          # 最近一块的 RMS，供电平表
        self.rms_hist = deque(maxlen=300)         # 近 30s 的 RMS 历史，估噪声底
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            blocksize=SAMPLE_RATE // 10, device=device, callback=self._on_audio)

    def _on_audio(self, indata, frames, time_info, status):
        self.level = float(np.sqrt(np.mean(indata[:, 0] ** 2)))
        self.rms_hist.append(self.level)
        self.blocks.append(indata[:, 0].copy())

    def start(self):
        self._stream.start()

    def stop(self):
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass

    def audio(self):
        return np.concatenate(self.blocks) if self.blocks else np.empty(0, np.float32)

    def clear(self):
        self.blocks.clear()


def effective_thr(recorder):
    """生效门限 = max(UI 绝对门限, 自适应噪声底×1.5)。
    噪声底取近 30s RMS 的 20 分位，采满 3s 启用（冷启动只有绝对门限兜底）。
    """
    if len(recorder.rms_hist) >= 30:
        floor = sorted(recorder.rms_hist)[int(len(recorder.rms_hist) * 0.2)]
    else:
        floor = 0.0
    return max(CFG["speech_rms"], floor * 1.5)


def is_speech(seg, recorder):
    """能量门限：VAD 之后、开会话/识别之前的一道关，环境噪声不触发同传。

    用段内 100ms 块 RMS 的 P90（语音峰值）而非全段平均——语句里的停顿会把
    平均拉低造成漏拾；稳态噪声（风扇）峰值≈均值挡得住，带起伏的说话放得行。
    返回 (是否语音, 峰值, 生效门限)。
    """
    bs = SAMPLE_RATE // 10
    n = len(seg) // bs
    if n >= 1:
        peak = float(np.percentile(
            np.sqrt(np.mean(seg[:n * bs].reshape(n, bs) ** 2, axis=1)), 90))
    else:
        peak = float(np.sqrt(np.mean(np.asarray(seg) ** 2)))
    thr = effective_thr(recorder)
    return (len(seg) >= MIN_UTT_S * SAMPLE_RATE and peak >= thr), peak, thr


# ---------------- 播放线程（伪流式） ----------------
class TtsStreamer:
    """常驻播放线程：从队列取合成好的音频段连续写入扬声器。

    write() 自带阻塞背压：播放慢于合成自动限速、快于合成等待下一段，
    天然实现"边合成边输出"。
    """

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
                    if item is None:
                        break
                    if item[0] != "chunk":
                        continue
                    if not self._started:
                        self._stream.start()
                        self._started = True
                        broadcast({"type": "speaking", "on": True})
                    self._stream.write(np.asarray(item[1], dtype=np.float32).reshape(-1, 1))
                    if self.q.empty():   # 队列空 = 本轮播放结束
                        broadcast({"type": "speaking", "on": False})
                finally:
                    self.q.task_done()
        except Exception as e:
            plog("service", f"[警告] 播放线程异常退出: {e}")

    def put_chunk(self, wav):
        self.q.put(("chunk", wav))

    def close(self, drain=True):
        """drain=True 等全部播完再退出；False 立即停止。"""
        if drain and self.thread.is_alive():
            self.q.join()
        self.q.put(None)
        self.thread.join(timeout=5)
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass


# ---------------- 同传会话（一次说话：说完后整句识别翻译） ----------------
class Session:
    """同传会话：vad_loop 负责开/关会话；说完（停顿/超长）后 worker 线程
    整句识别→整句翻译→译文切段流式合成播放 + 存档。"""

    def __init__(self, engine, first_seg):
        self.engine = engine
        self.blocks = [first_seg]       # 本会话 16k 音频（开场段先放进来，不丢首音节）
        self.zh = ""                    # 整句识别文本（收尾时填）
        self.en_parts = []              # 已翻译的英文片段（按顺序）
        self.frags = 0                  # 已合成段数
        self.wavs = []                  # 已合成音频 (1,N)，收尾时拼整句存档
        self.cost_asr = self.cost_mt = self.cost_tts = 0.0
        self.first_audio_s = None       # 会话开始 → 第一段音频就绪（同传首包）
        self.closing = False
        self.t0 = time.perf_counter()

    def absorb(self, rec):
        """把麦克风新攒的块收进会话（recorder 可能已停止）。"""
        if rec is not None and rec.blocks:
            self.blocks.extend(rec.blocks)
            rec.blocks = []

    def audio(self):
        return np.concatenate(self.blocks) if self.blocks else np.empty(0, np.float32)

    def seconds(self):
        return sum(len(b) for b in self.blocks) / SAMPLE_RATE


# ---------------- 服务引擎（常驻模型 + 采集/同传运行时） ----------------
class Engine:
    def __init__(self):
        self.asr = self.vad = self.mt_model = self.mt_tok = self.tts = None
        self.recorder = None
        self.streamer = None
        self.running = False
        self.session_dir = None
        self.sess = None                # 当前同传会话
        self.n = 0                      # 已开/完成会话数（句号）
        self._drops = 0                 # 连续丢弃的噪声段数（限频日志用）
        self._utt_q = queue.Queue()     # 流水线任务: "final" / None
        self._styles = {}               # voice -> style 缓存

    def get_style(self):
        """当前音色的 style（缓存；mock 模式返回 None）。"""
        if model_info["mock"]:
            return None
        if CFG["voice"] not in self._styles:
            self._styles[CFG["voice"]] = self.tts.get_voice_style(voice_name=CFG["voice"])
        return self._styles[CFG["voice"]]

    def ensure_session(self):
        if self.session_dir:
            return
        self.session_dir = os.path.join(OUTPUT_ROOT, datetime.now().strftime("web_%Y%m%d_%H%M%S"))
        os.makedirs(self.session_dir, exist_ok=True)
        with open(os.path.join(self.session_dir, "transcript.txt"), "w", encoding="utf-8") as f:
            f.write(f"===== web session {datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
        plog("service", f"存档目录: {self.session_dir}")

    # ---- 会话开关（vad_loop 线程调用） ----
    def try_open_session(self, rec):
        """能量门限 + VAD 双确认后才开会话——噪声连识别都不会触发。"""
        audio = rec.audio()
        if len(audio) < int(0.6 * SAMPLE_RATE):
            return
        if len(audio) > MAX_UTT_S * 2 * SAMPLE_RATE and not model_info["mock"]:
            rec.clear()   # 长时间无有效语音，防止缓冲无限攒
            return
        res = self.vad.generate(input=audio)
        segs = (res or [{}])[0].get("value") or []
        if not segs:
            return
        a = max(0, int(segs[0][0] * SAMPLE_RATE / 1000))
        b = min(len(audio), int(segs[-1][1] * SAMPLE_RATE / 1000))
        seg = audio[a:b]
        if model_info["mock"]:   # mock 无真实能量概念，直接放行
            ok, rms, thr = True, 0.0, 0.0
        else:
            ok, rms, thr = is_speech(seg, rec)
        if not ok:
            self._drops += 1
            if self._drops == 1 or self._drops % 10 == 0:   # 限频，别让噪声刷屏
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
                           f"峰值 {rms:.4f} >= 门限 {thr:.4f}), 进入边说边译")
        broadcast({"type": "sess_open", "n": self.n})

    def maybe_close_session(self):
        """说完（尾部静音/超长）→ 排队收尾。"""
        sess = self.sess
        if sess.closing:
            return
        audio = sess.audio()
        buf_ms = len(audio) * 1000 / SAMPLE_RATE
        res = self.vad.generate(input=audio)
        segs = (res or [{}])[0].get("value") or []
        if not segs:
            return
        fell_silent = (buf_ms - segs[-1][1]) >= CFG["tail_silence"] * 1000
        too_long = buf_ms >= MAX_UTT_S * 1000
        if not (fell_silent or too_long):
            return
        sess.closing = True
        plog("service", f"句#{self.n} 语音结束 (会话 {buf_ms / 1000:.1f}s, "
                        f"{'静音收尾' if fell_silent else '超长收尾'}), 等待最终处理")
        self._utt_q.put("final")

    # ---- 同传流水线（worker 线程调用） ----
    def finalize(self, sess):
        """收尾：整句一次识别 → 整句一次翻译 → 译文切段流式播放 → 存档。"""
        sess.absorb(self.recorder)
        zh, cost = transcribe(self.asr, sess.audio())
        sess.cost_asr += cost
        sess.zh = zh
        plog("record_log", f'句#{self.n} | 整句识别 {cost:.2f}s ({sess.seconds():.1f}s 音频): "{zh}"')
        self.sess = None   # 立刻释放会话，下一句可随时开；译文后台继续处理

        if not any(ch.isalnum() for ch in zh):
            plog("record_log", f"句#{self.n} | 未识别到有效内容, 忽略")
            self.n -= 1
            broadcast({"type": "sess_cancel", "n": self.n + 1})
            return

        en, cost_mt = translate(self.mt_model, self.mt_tok, zh)
        sess.cost_mt += cost_mt
        plog("record_log", f'句#{self.n} | 整句翻译 {cost_mt:.2f}s: "{en}"')
        self.speak_frags(sess, en)

        # 存档：识别输入(16k) + 合成输出(44.1k) + transcript
        fname = f"utt_{self.n:04d}.wav"
        import soundfile as sf
        sf.write(os.path.join(self.session_dir, f"utt_{self.n:04d}_in.wav"),
                 sess.audio(), SAMPLE_RATE)
        combined = np.concatenate(sess.wavs, axis=1) if sess.wavs else np.zeros((1, 0), np.float32)
        self.tts.save_audio(combined, os.path.join(self.session_dir, fname))
        en_all = "".join(sess.en_parts)
        with open(os.path.join(self.session_dir, "transcript.txt"), "a", encoding="utf-8") as f:
            f.write(f"[句#{self.n}] 你说: {sess.zh}\n[句#{self.n}] 译文: {en_all}\n"
                    f"[句#{self.n}] {fname}\n\n")
        total_s = time.perf_counter() - sess.t0
        plog("record_log", f"句#{self.n} | 完成: 会话 {total_s:.1f}s, 识别 {len(sess.zh)} 字, "
                           f"播放 {combined.shape[1] / SR_TTS:.1f}s, 存档 {fname} "
                           f"(+{sess.frags} 段, in.wav)")
        broadcast({"type": "utt", "n": self.n, "zh": sess.zh, "en": en_all,
                   "asr": round(sess.cost_asr, 2), "mt": round(sess.cost_mt, 2),
                   "tts": round(sess.cost_tts, 2),
                   "first_latency": round(sess.first_audio_s or 0, 2),
                   "total": round(total_s, 2), "dur": round(combined.shape[1] / SR_TTS, 1),
                   "frags": sess.frags, "wav": f"{os.path.basename(self.session_dir)}/{fname}"})

    def speak_frags(self, sess, en):
        """翻译片段 → 切段合成 → 先入队播放再写盘（出声不等存档）。"""
        frags = split_by_punct(en)
        gap = np.zeros((int(STREAM_GAP_S * SR_TTS), 1), dtype=np.float32)
        streamer = self.streamer
        for i, frag in enumerate(frags):
            wav, dur, cost = synthesize(self.tts, self.get_style(), frag)
            sess.cost_tts += cost
            sess.frags += 1
            if sess.first_audio_s is None:   # 同传首包：会话开始 → 第一段音频就绪
                sess.first_audio_s = time.perf_counter() - sess.t0
                plog("record_log", f"句#{self.n} | 同传首包: {sess.first_audio_s:.2f}s "
                                   f"(开会话→第一段译文出声)")
            sess.wavs.append(wav)
            piece = np.concatenate([wav.squeeze(axis=0).reshape(-1, 1),
                                    gap if i < len(frags) - 1 else np.zeros((0, 1), np.float32)])
            if streamer:
                streamer.put_chunk(piece)   # 先送播放，写盘放在后面不挡出声
            fname = f"utt_{self.n:04d}_chunk{sess.frags:02d}.wav"
            self.tts.save_audio(wav, os.path.join(self.session_dir, fname))
            plog("record_log", f'句#{self.n} | 段{sess.frags}: "{frag}" '
                               f"合成 {cost:.2f}s, 音频 {dur:.1f}s -> {fname}")
        sess.en_parts.append(en + " ")   # 拼整句译文时空格分隔

    # ---- 采集开始/停止 ----
    def start_capture(self):
        if self.running:
            return
        if model_info["phase"] != "ready":
            raise RuntimeError("模型尚未就绪")
        self.ensure_session()
        self.recorder = MicRecorder(device=CFG["device"])
        self.recorder.start()
        self.streamer = TtsStreamer(self)
        self.running = True
        plog("service", f"麦克风采集开始 (设备={'默认' if CFG['device'] is None else CFG['device']}), "
                        f"断句停顿 {CFG['tail_silence']}s, 噪声门限 {CFG['speech_rms']}")
        threading.Thread(target=self._vad_loop, daemon=True).start()
        threading.Thread(target=self._worker, daemon=True).start()
        threading.Thread(target=self._level_loop, daemon=True).start()
        broadcast_state()

    def stop_capture(self):
        if not self.running:
            return
        self.running = False
        if self.recorder:
            self.recorder.stop()
        plog("service", f"麦克风采集停止, 共 {self.n} 个会话")
        threading.Thread(target=self._shutdown, daemon=True).start()
        broadcast_state()

    def _shutdown(self):
        """后台收尾：等在途会话处理完、剩余音频播完，再退出各线程（不阻塞 HTTP）。"""
        if self.sess and not self.sess.closing:
            self.sess.closing = True
            self._utt_q.put("final")   # 把说到一半的会话收尾存档
        self._utt_q.join()
        self._utt_q.put(None)          # 唤醒 worker 退出
        if self.streamer:
            self.streamer.close(drain=True)   # 等剩余音频播完
        self.recorder = self.streamer = None

    def _vad_loop(self):
        while self.running:
            time.sleep(VAD_POLL_S)
            rec = self.recorder
            if rec is None:
                break
            try:
                if self.sess is None:
                    self.try_open_session(rec)
                else:
                    self.sess.absorb(rec)
                    self.maybe_close_session()
            except Exception as e:
                plog("service", f"[警告] 会话监测出错: {e!r}")

    def _worker(self):
        while True:
            job = self._utt_q.get()
            if job is None:
                break
            sess = self.sess
            try:
                if sess is not None:
                    self.finalize(sess)
            except Exception as e:
                plog("record_log", f"句#{self.n} 处理出错: {e!r}")
                broadcast({"type": "utt_error", "msg": str(e)})
                if job == "final":
                    self.sess = None   # 兜底，别让会话卡死
            self._utt_q.task_done()

    def _level_loop(self):
        """电平表：定期把 RMS 推给前端（麦克风监测），说话标记与门限同口径。"""
        while self.running and self.recorder is not None:
            time.sleep(LEVEL_POLL_S)
            if self.recorder is not None:
                thr = effective_thr(self.recorder)
                broadcast({"type": "level", "rms": round(self.recorder.level, 4),
                           "speech": self.recorder.level > thr})


engine = Engine()


# ---------------- 状态快照 / 广播 ----------------
def status_event():
    st = {k: v for k, v in model_info.items()}
    st.update({"type": "state", "running": engine.running, "config": dict(CFG),
               "session": os.path.basename(engine.session_dir) if engine.session_dir else None,
               "n_utts": engine.n,
               "stats": {k: [round(v[0], 2), v[1]] for k, v in record_cost_stats.items()}})
    return st


def broadcast_state():
    broadcast(status_event())


# ---------------- FastAPI 应用 ----------------
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """记录事件循环（供推理线程广播用），并补发状态给加载期间就打开的页面。"""
    global _loop
    _loop = asyncio.get_running_loop()
    broadcast_state()
    yield


app = FastAPI(title="实时语音翻译 Demo", lifespan=_lifespan)
HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_web_silence.html")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    _clients.add(ws)
    try:
        await ws.send_text(json.dumps(status_event(), ensure_ascii=False))
        while True:
            await ws.receive_text()   # 仅保活
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        _clients.discard(ws)


@app.get("/")
def index():
    return FileResponse(HTML_PATH)


@app.get("/api/status")
def api_status():
    return status_event()


@app.post("/api/settings")
async def api_settings(body: dict):
    """可调：断句停顿 / 噪声门限，立即生效。"""
    if "tail_silence" in body:
        CFG["tail_silence"] = min(2.0, max(0.2, float(body["tail_silence"])))
    if "speech_rms" in body:
        CFG["speech_rms"] = min(0.08, max(0.003, float(body["speech_rms"])))
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


app.mount("/audio", StaticFiles(directory=OUTPUT_ROOT), name="audio")   # 句子 wav 回放


# ---------------- mock 模型（--mock：不加载真模型，联调界面/日志） ----------------
class MockVad:
    """mock 断句：3s 开会话，3~6s 视为一直在说，6s 后伪造静音收尾。"""

    def generate(self, input=None, **kw):
        n, ms = len(input), int(len(input) * 1000 / SAMPLE_RATE)
        if n < int(3.0 * SAMPLE_RATE):
            return [{"value": []}]
        if n < int(6.0 * SAMPLE_RATE):
            return [{"value": [[0, ms]]}]
        return [{"value": [[0, ms - 800]]}]


class MockTTS:
    def save_audio(self, wav, path):
        import soundfile as sf
        sf.write(path, np.asarray(wav, dtype=np.float32).reshape(-1, 1), SR_TTS)

    def get_voice_style(self, voice_name=None):
        return None


def install_mock_models():
    model_info["mock"] = True
    model_info.update({"gpu": False, "asr_dev": "mock", "vad_dev": "mock",
                       "mt_dev": "mock", "tts_dev": "mock", "wall_s": 0.0, "warmup_s": 0.0})
    engine.asr, engine.vad = object(), MockVad()
    engine.mt_model = engine.mt_tok = None
    engine.tts = MockTTS()
    model_info["phase"] = "ready"
    plog("model_jiazai", "MOCK 模式：跳过真实模型加载，使用假数据流水线")


# ---------------- 入口 ----------------
def parse_args():
    p = argparse.ArgumentParser(description="中文语音 → 英文语音 实时同传 Web 服务")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8324)
    p.add_argument("--mic", type=int, default=None, help="麦克风编号，默认系统默认设备")
    p.add_argument("--tts-steps", type=int, default=CFG["tts_steps"], help="合成步数 5~12")
    p.add_argument("--voice", default=CFG["voice"], help="Supertonic 音色 (M1/F1/...)")
    p.add_argument("--no-auto-start", action="store_true", help="就绪后不自动开麦")
    p.add_argument("--mock", action="store_true", help="不加载真实模型，假数据联调")
    p.add_argument("--list-devices", action="store_true", help="列出音频设备后退出")
    return p.parse_args()


def main():
    args = parse_args()
    if args.list_devices:
        print(sd.query_devices())
        return
    CFG.update(tts_steps=args.tts_steps, voice=args.voice, device=args.mic)

    print(f">>> 服务日志: {os.path.abspath(LOG_PATH)}", flush=True)
    plog("service", f"===== 服务启动 pid={os.getpid()} port={args.port} "
                    f"{'MOCK' if args.mock else '真实模型'} =====")

    def load_and_ready():
        """后台线程：加载 + 预热 → 常驻 → 自动开麦。"""
        try:
            if args.mock:
                install_mock_models()
            else:
                asr, vad, mt_model, mt_tok, tts = load_models()
                engine.asr, engine.vad = asr, vad
                engine.mt_model, engine.mt_tok, engine.tts = mt_model, mt_tok, tts
                t0 = time.perf_counter()
                engine.get_style()   # 顺带加载默认音色 style
                noise = (np.random.randn(SAMPLE_RATE // 2) * 0.01).astype(np.float32)
                transcribe(engine.asr, noise)
                translate(engine.mt_model, engine.mt_tok, "你好。")
                synthesize(engine.tts, engine.get_style(), "Hello.")
                record_cost_stats.clear()   # 预热耗时不计入统计
                model_info["warmup_s"] = time.perf_counter() - t0
                plog("model_jiazai", f"预热完成 {model_info['warmup_s']:.1f}s, 模型常驻, "
                                     f"后续调用零加载等待")
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
        if engine.running:
            engine.stop_capture()
        plog("service", "===== 服务退出 =====")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
实时语音翻译 Demo · Web 版：中文语音输入 → 英文语音同传（语义分段流式）
====================================================================
在 demo_web_asr_full.py（标点断句版）基础上的新架构：

    语义分段    会话无录制时长上限（无硬切）。说话期间每 1s 对"当前段"
                音频（自上一个语义切分点攒起来的小 buffer）做一次只识别
                不翻译的检测，文本尾部出现终结标点（。？！；…）即判定
                语义完整：整段文本立刻送翻译，剩余音频滚入下一段继续。
    整段送入    翻译和合成都以完整语义段为单位：切分时直接用增量识别的
                文本送翻译（不重识别）；只有说完后的最后一段补一次整段
                识别保证准确。译文再按标点切段流式合成、连续播放。
    双线程流水  翻译线程与合成线程分开：段2的翻译不等段1的合成完，
                多段连续说话时各段首包延迟更低。
    收尾兜底    停顿超过断句停顿判定说完（尾部无标点的半截话容忍停顿
                多等 1s，最多 2 次），冲洗最后一段。
    噪声门限    开会话要过两道确认：VAD 报有语音段 + 段内能量(RMS)超过
                门限（绝对门限 UI 可调 + 自适应噪声底×1.5 + 最短时长 0.5s）。
    全双工      始终开启：播放译文期间麦克风照常收音，可以边听边说下一句
                （按戴耳机设计，不做外放半双工防回声）。

日志（单一全局日志 logs/service_log.txt，行首 tag 便于 grep）：
    [record_log]   每次会话/语义段的完整过程（开会话/检测/切分/翻译/合成/存档）
    [model_jiazai] 模型加载与预热
    [service]      其余服务事件（启动/开麦停麦/丢弃噪声/异常）

运行：
    conda activate demo
    python demo_web_asr_semantic.py       # 端口 8325，就绪后自动开麦
    python demo_web_asr_semantic.py --mic 1   # 指定麦克风
    python demo_web_asr_semantic.py --mock    # 不加载模型，假数据联调界面/日志

    对比版本：标点断句 demo_web_asr_full.py (8322) / 静音断句 demo_web_asr_silence.py (8324)

退出：Ctrl+C；存档：demo_output/web_时间戳/（每个语义段 识别输入.wav + 合成.wav + transcript.txt）
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
MIN_UTT_S = 0.5            # 语音段短于此视为噪声（咳嗽/碰撞声），不开会话
NOISE_BUF_CLEAR_S = 30.0   # 迟迟无有效语音时，清掉攒了太久的噪声缓冲
VAD_POLL_S = 0.3           # 会话开/关检测间隔
STREAM_POLL_S = 1.0        # 语义切分检测（只识别看标点，不翻译）间隔
SEG_MIN_S = 1.0            # 语义切分所需的最短段音频（太短不切）
OVERLONG_S = 10.0          # 超长句门槛：超过它绕过"停顿才切"限制，按语义尽快出段
LEVEL_POLL_S = 0.1         # 电平推送间隔
STREAM_GAP_S = 0.12        # 段间补的静音
MIN_FRAG_WORDS = 2         # 碎片过短阈值（英文词数）
MIN_FRAG_CJK = 4           # 碎片过短阈值（中文字数）
TERMINAL_PUNCT = re.compile(r"[。？！；!?;…]\s*$")   # 文本尾部的终结标点（切分信号）
PUNCT_SILENCE_MS = 100    # 超长旁路切分的最小尾音确认（正常切分门槛是 400ms）
# SenseVoice 会把情感标记输出成 emoji（如 😔），翻译和合成都不需要
EMOJI_RE = re.compile(r"[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF️]+")
ASR_DEVICE = "cuda:0"
VAD_DEVICE = "cpu"         # fsmn-vad 很小，放 CPU 避免抢显存
MT_MODEL = "tencent/Hy-MT2-1.8B"
MT_MAX_NEW_TOKENS = 512
TTS_LANG = "en"            # 本 demo 固定 中→英
OUTPUT_ROOT = "demo_output"
LOG_PATH = os.path.join("logs", "service_log.txt")

# 运行期配置（前端可实时改 tail_silence / speech_rms，其余由命令行指定）
CFG = {
    "tail_silence": 1.5,   # 兜底停顿阈值：静音达到它就冲洗尾段结束会话
    "speech_rms": 0.001,   # 绝对门限默认调低，主要靠 VAD 和自适应底噪挡噪声
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


def _tts_gpu_bootstrap():
    """让 Supertonic 走 GPU：onnxruntime-gpu(CUDA12 构建) + torch 自带的 cu12 DLL。
    supertonic 硬编码纯 CPU provider 列表（config.py），这里扩成 CUDA 优先；
    任一步失败都静默回退 CPU（loader 自带回退）。必须在 import supertonic 前调用。"""
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
def transcribe(asr, speech, tag="asr"):
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
    return EMOJI_RE.sub("", text).strip(), record_cost(tag, t0)


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
    return EMOJI_RE.sub("", en).strip(), record_cost("mt", t0)


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

    def trim(self, keep_s):
        """只保留最近 keep_s 秒（裁前不裁后）——长时间无语音时防缓冲无限涨，
        且最近说过的话不丢（旧的 clear() 会把句首一起吃掉）。"""
        total = sum(len(b) for b in self.blocks)
        limit = int(keep_s * SAMPLE_RATE)
        while self.blocks and total > limit:
            total -= len(self.blocks[0])
            self.blocks.pop(0)

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


# ---------------- 同传会话（语义分段流式） ----------------
class Session:
    """一次连续说话（VAD 开→关，无时长上限），内部按语义切分：

    blocks 只攒"当前段"音频（自上一个语义切分点起的小 buffer）；detect
    每 1s 对它识别一次，尾部出现终结标点 → cut() 切出整段：文本送翻译、
    音频切点取 VAD 报的最后一个语音结束点（估算），剩余音频滚入下一段。
    """

    def __init__(self, engine, first_seg):
        self.engine = engine
        self.lock = threading.Lock()
        self.blocks = [first_seg]       # 当前段 16k 音频（不丢首音节）
        self.detect_zh = ""             # 当前段最近一次识别文本（切分/容忍判断用）
        self.prev_seg_zh = ""           # 上一段切出的文本（跨界去重用）
        self.pending_cut = False        # 是否处于延迟确认状态
        self.pending_cut_time = 0       # 触发候选切分的时间戳
        self.last_detect_zh = ""        # 上一次 detect_once 识别出的文本（判断 ASR 是否稳定）
        self.asr_stable_count = 0       # ASR 连续输出相同文本的次数
        self.k = 0                      # 本会话已切出的语义段数
        self.last_detect = time.perf_counter()
        self.closing = False
        self.t0 = time.perf_counter()

    def absorb(self, rec):
        """把麦克风新攒的块收进当前段（recorder 可能已停止）。"""
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

    def trailing_silence_ms(self, vad):
        """计算当前 buffer 末尾的静音时长 (ms)——物理停顿检测。"""
        audio = self.audio()
        if len(audio) == 0:
            return 0
        res = vad.generate(input=audio)
        segs = (res or [{}])[0].get("value") or []
        if not segs:
            return int(len(audio) * 1000 / SAMPLE_RATE)
        return max(0, int(len(audio) * 1000 / SAMPLE_RATE - segs[-1][1]))

    def cut(self, zh, vad):
        """语义段完成：整段切出，但保留最后 300ms 作为下一段安全头防吞字
        （音频到达快于文本刷新，全切会吞掉下一段句首）。"""
        with self.lock:
            if not self.blocks:
                return None
            audio = np.concatenate(self.blocks)

            # 保留最后 150ms 作为下一段安全头（防止下一句首字被吞；
            # 过长的安全尾会让边界字在下一段重复，配合跨界去重使用）
            safe_tail_samples = int(0.15 * SAMPLE_RATE)
            if len(audio) > safe_tail_samples:
                seg_audio = audio[:-safe_tail_samples]
                self.blocks = [audio[-safe_tail_samples:]]
            else:
                seg_audio = audio
                self.blocks = []

            self.detect_zh = ""
            self.prev_seg_zh = zh
            self.k += 1
            return seg_audio

    def take_all(self):
        """取走当前段全部音频（收尾冲洗最后一段用）。"""
        with self.lock:
            blocks, self.blocks = self.blocks, []
            return blocks


# 语义不完整词表：清洗标点后若以此结尾，即使有终结标点也不切（拦截 ASR 在
# 语流中补的伪标点，避免从"因为/然后/意思是…"处腰斩）
INCOMPLETE_ENDINGS = [
    "因为", "虽然", "但是", "不过", "而且", "并且", "或者", "如果", "假如",
    "即使", "尽管", "无论", "不管", "只要", "只有", "除了", "关于", "对于",
    "比如", "例如", "像", "以及", "还有", "另外", "此外", "同时",
    "正在", "准备", "打算", "计划", "想要", "需要", "可能", "应该",
    "第一", "第二", "第三", "首先", "其次", "最后", "然后", "接着", "于是",
    "总之", "总的来说", "换句话说", "也就是说", "就是说",
    "我觉得", "我认为", "他说", "她说", "他们说", "意思是"
]


# ---------------- 服务引擎（常驻模型 + 采集/同传运行时） ----------------
class Engine:
    def __init__(self):
        self.asr = self.vad = self.mt_model = self.mt_tok = self.tts = None
        self.recorder = None
        self.streamer = None
        self.running = False
        self.session_dir = None
        self.sess = None                # 当前同传会话
        self.n = 0                      # 已开会话数（句号）
        self.n_seg = 0                  # 已完成的语义段数
        self._mt_q = queue.Queue()      # 翻译任务: {"seg"|"flush", ...} / None
        self._tts_q = queue.Queue()     # 合成任务: {"tts", ...} / None
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
        """VAD 确认有语音即开会话。入口宽松：能量只记录不拦截，音频永不因
        门限被丢弃（噪声缓冲只裁前保后）；垃圾内容由出段口的
        VAD+实义字符检查拦截（出口严格），静音到不了翻译。"""
        audio = rec.audio()
        if len(audio) < int(0.6 * SAMPLE_RATE):
            return
        if len(audio) > NOISE_BUF_CLEAR_S * SAMPLE_RATE and not model_info["mock"]:
            rec.trim(NOISE_BUF_CLEAR_S)   # 裁前保后，句首不丢
            audio = rec.audio()
        res = self.vad.generate(input=audio)
        segs = (res or [{}])[0].get("value") or []
        if not segs:
            return
        a = max(0, int(segs[0][0] * SAMPLE_RATE / 1000) - int(0.3 * SAMPLE_RATE))  # 回退0.3s防首音节被裁
        b = min(len(audio), int(segs[-1][1] * SAMPLE_RATE / 1000))
        seg = audio[a:b]
        if model_info["mock"]:   # mock 无真实能量概念
            rms, thr = 0.0, 0.0
        else:
            _, rms, thr = is_speech(seg, rec)   # 能量只作记录，不再拦截

        self.ensure_session()
        self.n += 1
        self.sess = Session(self, seg)
        rec.clear()
        plog("record_log", f"句#{self.n} | 会话开始 ({len(seg) / SAMPLE_RATE:.1f}s 语音, "
                           f"能量峰值 {rms:.4f}, 门限参考 {thr:.4f}), 语义切分同传中")
        broadcast({"type": "sess_open", "n": self.n})

    def maybe_close_session(self):
        """说完（当前段尾部静音）→ 冲洗最后一段并结束会话。"""
        sess = self.sess
        if sess.closing or sess.seconds() < 0.1:
            return
        silence_ms = sess.trailing_silence_ms(self.vad)

        # 停顿达到设定阈值，直接触发兜底
        if silence_ms >= CFG["tail_silence"] * 1000:
            sess.closing = True
            blocks = sess.take_all()
            plog("service", f"句#{self.n} 语音结束 (当前段 {sess.seconds():.1f}s, "
                            f"尾音 {silence_ms}ms), 冲洗最后一段")
            self._mt_q.put({"type": "flush", "n": self.n, "k": sess.k + 1,
                            "prev_zh": sess.prev_seg_zh,
                            "audio": np.concatenate(blocks) if blocks else np.empty(0, np.float32)})
            self.sess = None

    # ---- 语义切分（detect 线程调用） ----
    def detect_once(self, sess):
        """语义切分检测：基于 VAD 尾音静音作为主触发时机。
        没停顿坚决不切（防伪句号腰斩）；超长句例外——按语义尽快出段。"""
        if sess.seconds() < SEG_MIN_S:
            return

        # 1. 检查物理尾音静音（没停顿坚决不切，防伪句号腰斩）
        silence_ms = sess.trailing_silence_ms(self.vad)
        overlong = sess.seconds() > OVERLONG_S
        if silence_ms < (PUNCT_SILENCE_MS if overlong else 400):
            return  # 连续说话中的伪句号不切；超长旁路也要求换气级(100ms)停顿确认

        # 2. 用户停顿（或句子超长），跑一次 ASR 看文本
        zh, cost = transcribe(self.asr, sess.audio(), tag="asr_detect")
        # 跨界去重：安全尾里的边界字可能被下一段开头重复识别（如"…识别。"+"别…"）
        prev_tail = (sess.prev_seg_zh or "").rstrip("。！？.!?；;…，、 ")
        if prev_tail and zh[:1] == prev_tail[-1]:
            plog("service", f"句#{self.n} 跨界去重: 丢弃段首重复字'{zh[0]}'")
            zh = zh[1:].lstrip("，、 ")
        sess.last_detect_zh = sess.detect_zh
        sess.detect_zh = zh
        valid_chars = len(re.findall(r'[一-鿿a-zA-Z0-9]', zh))

        # 3. 无标点：超长句按语义兜底硬切（语义不可用时的最后手段）；
        #    否则交给 maybe_close_session 物理兜底
        if not TERMINAL_PUNCT.search(zh):
            if overlong and valid_chars > 20:
                plog("service", f"句#{self.n} 超长无标点({sess.seconds():.1f}s), 强制切分")
                seg_audio = sess.cut(zh, self.vad)
                if seg_audio is not None:
                    self.n_seg += 1
                    self._mt_q.put({"type": "seg", "n": self.n, "k": sess.k, "zh": zh, "audio": seg_audio})
            return

        # 4. 短句保护
        if sess.seconds() < 2.0 and valid_chars < 6:
            return

        # 5. 语义不完整保护
        zh_clean = zh.rstrip("。！？.!?；;… ")
        if any(zh_clean.endswith(w) for w in INCOMPLETE_ENDINGS):
            return

        # 6. 停顿了，有标点，且语义完整 -> 真正切分！
        seg_audio = sess.cut(zh, self.vad)
        if seg_audio is None:
            return
        self.n_seg += 1
        plog("record_log", f'句#{self.n} | 段{sess.k} 语义切分 ({cost:.2f}s 检测, '
                           f'尾音 {silence_ms}ms): "{zh}"')
        self._mt_q.put({"type": "seg", "n": self.n, "k": sess.k, "zh": zh, "audio": seg_audio})

    # ---- 翻译线程（完整语义段 → 完整译文） ----
    def mt_handle(self, job):
        if job["type"] == "seg":
            en, cost = translate(self.mt_model, self.mt_tok, job["zh"])
            plog("record_log", f'句#{job["n"]} | 段{job["k"]} 整段翻译 {cost:.2f}s: "{en}"')
            self._tts_q.put({"type": "tts", "n": job["n"], "k": job["k"],
                             "zh": job["zh"], "en": en, "audio": job["audio"]})
        else:   # flush: 说完后的最后一段，先补一次整段识别保证准确
            audio = job["audio"]
            if len(audio) < int(0.3 * SAMPLE_RATE):
                plog("service", f"句#{job['n']} 最后一段过短({len(audio) / SAMPLE_RATE:.1f}s), 忽略")
                if job["k"] == 1:
                    self.n -= 1
                    broadcast({"type": "sess_cancel", "n": self.n + 1})
                return
            zh, cost = transcribe(self.asr, audio)
            # 跨界去重（与 detect_once 同规则）：flush 音频含上一段的安全尾
            prev_tail = (job.get("prev_zh") or "").rstrip("。！？.!?；;…，、 ")
            if prev_tail and zh[:1] == prev_tail[-1]:
                plog("service", f"句#{job['n']} 跨界去重: 丢弃段首重复字'{zh[0]}'")
                zh = zh[1:].lstrip("，、 ")
            valid_chars = len(re.findall(r'[一-鿿a-zA-Z0-9]', zh))
            # 出口拦截：不足 2 个有效字符（如"嗯。""我。"），直接丢弃防幻觉播出
            if valid_chars < 2:
                plog("record_log", f"句#{job['n']} | 最后一段有效字符不足({zh}), 忽略防幻觉")
                if job["k"] == 1:
                    self.n -= 1
                    broadcast({"type": "sess_cancel", "n": self.n + 1})
                return
            plog("record_log", f'句#{job["n"]} | 段{job["k"]} 收尾识别 {cost:.2f}s '
                               f'({len(audio) / SAMPLE_RATE:.1f}s 音频): "{zh}"')
            en, cost_mt = translate(self.mt_model, self.mt_tok, zh)
            plog("record_log", f'句#{job["n"]} | 段{job["k"]} 整段翻译 {cost_mt:.2f}s: "{en}"')
            self._tts_q.put({"type": "tts", "n": job["n"], "k": job["k"],
                             "zh": zh, "en": en, "audio": audio})

    # ---- 合成+存档线程（完整译文 → 切段流式播放 + 存档） ----
    def tts_handle(self, job):
        frags = split_by_punct(job["en"])
        gap = np.zeros((int(STREAM_GAP_S * SR_TTS), 1), dtype=np.float32)
        wavs = []
        cost_tts = 0.0
        for i, frag in enumerate(frags):
            wav, dur, cost = synthesize(self.tts, self.get_style(), frag)
            cost_tts += cost
            wavs.append(wav)
            piece = np.concatenate([wav.squeeze(axis=0).reshape(-1, 1),
                                    gap if i < len(frags) - 1 else np.zeros((0, 1), np.float32)])
            if self.streamer:
                self.streamer.put_chunk(piece)   # 先送播放，写盘放在后面不挡出声
        # 存档：该语义段的 识别输入(16k) + 合成输出(44.1k) + transcript
        fname = f"utt_{job['n']:04d}s{job['k']:02d}.wav"
        import soundfile as sf
        sf.write(os.path.join(self.session_dir, f"utt_{job['n']:04d}s{job['k']:02d}_in.wav"),
                 job["audio"], SAMPLE_RATE)
        combined = np.concatenate(wavs, axis=1) if wavs else np.zeros((1, 0), np.float32)
        self.tts.save_audio(combined, os.path.join(self.session_dir, fname))
        with open(os.path.join(self.session_dir, "transcript.txt"), "a", encoding="utf-8") as f:
            f.write(f"[句#{job['n']}-段{job['k']}] 你说: {job['zh']}\n"
                    f"[句#{job['n']}-段{job['k']}] 译文: {job['en']}\n"
                    f"[句#{job['n']}-段{job['k']}] {fname}\n\n")
        plog("record_log", f"句#{job['n']} | 段{job['k']} 完成: 合成 {cost_tts:.2f}s, "
                           f"播放 {combined.shape[1] / SR_TTS:.1f}s, 存档 {fname} (+{len(frags)} 段)")
        broadcast({"type": "seg", "n": job["n"], "k": job["k"], "zh": job["zh"], "en": job["en"],
                   "mt": 0, "tts": round(cost_tts, 2),
                   "dur": round(combined.shape[1] / SR_TTS, 1), "frags": len(frags),
                   "wav": f"{os.path.basename(self.session_dir)}/{fname}"})

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
                        f"语义切分检测 {STREAM_POLL_S}s, 断句停顿 {CFG['tail_silence']}s, "
                        f"噪声门限 {CFG['speech_rms']}")
        threading.Thread(target=self._vad_loop, daemon=True).start()
        threading.Thread(target=self._detect_loop, daemon=True).start()
        threading.Thread(target=self._mt_worker, daemon=True).start()
        threading.Thread(target=self._tts_worker, daemon=True).start()
        threading.Thread(target=self._level_loop, daemon=True).start()
        broadcast_state()

    def stop_capture(self):
        if not self.running:
            return
        self.running = False
        if self.recorder:
            self.recorder.stop()
        plog("service", f"麦克风采集停止, 共 {self.n} 个会话 / {self.n_seg} 个语义段")
        threading.Thread(target=self._shutdown, daemon=True).start()
        broadcast_state()

    def _shutdown(self):
        """后台收尾：冲洗最后一段、排空翻译/合成队列、播完剩余音频（不阻塞 HTTP）。"""
        if self.sess and not self.sess.closing:
            self.sess.closing = True
            blocks = self.sess.take_all()
            self._mt_q.put({"type": "flush", "n": self.n, "k": self.sess.k + 1,
                            "prev_zh": self.sess.prev_seg_zh,
                            "audio": np.concatenate(blocks) if blocks else np.empty(0, np.float32)})
            self.sess = None
        self._mt_q.put(None)    # 停翻译线程（其在退出前排空并投递合成任务）
        self._mt_q.join()
        self._tts_q.put(None)   # 停合成线程
        self._tts_q.join()
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

    def _detect_loop(self):
        """语义切分检测：每 1s 对当前段识别一次（只看标点，不翻译）。"""
        while self.running:
            time.sleep(STREAM_POLL_S)
            sess = self.sess
            if sess is None or sess.closing:
                continue
            try:
                self.detect_once(sess)
            except Exception as e:
                plog("service", f"[警告] 语义切分检测出错: {e!r}")

    def _mt_worker(self):
        while True:
            job = self._mt_q.get()
            try:
                if job is None:
                    break
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
                if job is None:
                    break
                self.tts_handle(job)
            except Exception as e:
                plog("record_log", f"合成任务出错: {e!r}")
                broadcast({"type": "utt_error", "msg": str(e)})
            finally:
                self._tts_q.task_done()

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
               "n_utts": engine.n_seg,
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


app = FastAPI(title="实时语音翻译 Demo · 语义分段", lifespan=_lifespan)
HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_web_semantic.html")


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


app.mount("/audio", StaticFiles(directory=OUTPUT_ROOT), name="audio")   # 语义段 wav 回放


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
    p = argparse.ArgumentParser(description="中文语音 → 英文语音 语义分段同传 Web 服务")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8325)
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

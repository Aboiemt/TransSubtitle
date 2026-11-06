"""基于能量门限的语音分段 + faster-whisper 识别（支持边说边出临时结果）。"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from faster_whisper import WhisperModel

from config import AppConfig

# 有显卡时的默认模型：实测歌曲识别 召回 57% / 精确 85%（base 为 45% / 49%）
_GPU_DEFAULT_MODEL = "large-v3-turbo"
_CPU_DEFAULT_MODEL = "base"


def _setup_cuda() -> bool:
    """ctranslate2 按文件名加载 cuBLAS（不认 add_dll_directory），
    需把 pip 版 nvidia-cublas-cu12 的 DLL 预加载；cuDNN 已随 ctranslate2 附带。"""
    if sys.platform != "win32":
        return True
    from settings_store import _app_dir

    # 优先程序目录 tools\cuda（随程序拷走、不占 C 盘），其次 pip 包，最后打包内置
    dirs: list[str] = [str(_app_dir() / "tools" / "cuda")]
    try:
        import nvidia.cublas as nc  # type: ignore[import-not-found]

        dirs += [os.path.join(p, "bin") for p in nc.__path__]
    except ImportError:
        pass
    meipass = getattr(sys, "_MEIPASS", None)  # 打包后
    if meipass:
        dirs.append(os.path.join(meipass, "nvidia", "cublas", "bin"))
    for d in dirs:
        if not os.path.isfile(os.path.join(d, "cublas64_12.dll")):
            continue
        os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
        try:
            for name in ("cublasLt64_12.dll", "cublas64_12.dll"):
                ctypes.CDLL(os.path.join(d, name))
            return True
        except OSError:
            return False
    return False


def _model_source(name: str) -> str:
    """项目 models\\whisper-<name>\\ 里有模型就直接用（随程序拷走、免下载），
    否则返回模型名，由 faster-whisper 自动下载到 HuggingFace 缓存。"""
    from settings_store import _app_dir

    local = _app_dir() / "models" / f"whisper-{name}"
    return str(local) if (local / "model.bin").is_file() else name


def _cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0 and _setup_cuda()
    except Exception:  # noqa: BLE001
        return False

# Whisper 在静音 / 音乐上常见的幻听（整段丢弃）
_HALLUCINATION_MARKERS = (
    "ご視聴ありがとう",
    "チャンネル登録",
    "字幕作成",
    "明镜与点点",
    "请不吝点赞",
    "字幕由",
    "Amara.org",
)


@dataclass
class TranscriptResult:
    text: str
    language: str | None
    audio_seconds: float


class SpeechSegmenter:
    """
    把连续音频切成语音片段。
    - on_partial: 说话过程中周期性回调（临时字幕）
    - on_segment: 静音或达到最长时长时回调（定稿）
    """

    def __init__(
        self,
        config: AppConfig,
        on_segment: Callable[[np.ndarray], None],
        on_partial: Callable[[np.ndarray], None] | None = None,
    ) -> None:
        self.config = config
        self.on_segment = on_segment
        self.on_partial = on_partial
        self._buffer: list[np.ndarray] = []
        self._speeching = False
        self._silence_blocks = 0
        self._speech_blocks = 0
        self._last_partial_blocks = 0
        self._lock = threading.Lock()

        self._silence_need = max(1, config.silence_ms // config.block_ms)
        self._min_speech = max(1, config.min_speech_ms // config.block_ms)
        self._max_speech = max(1, config.max_speech_ms // config.block_ms)
        self._partial_need = max(1, config.partial_ms // config.block_ms)

    def feed(self, chunk: np.ndarray) -> None:
        rms = float(np.sqrt(np.mean(np.square(chunk))) + 1e-12)
        is_speech = rms >= self.config.rms_threshold

        partial_audio: np.ndarray | None = None
        final_audio: np.ndarray | None = None

        with self._lock:
            if is_speech:
                self._silence_blocks = 0
                if not self._speeching:
                    self._speeching = True
                    self._speech_blocks = 0
                    self._last_partial_blocks = 0
                    self._buffer = []
                self._buffer.append(chunk)
                self._speech_blocks += 1

                # 边说边出：每隔 partial_ms 发一次临时片段
                if (
                    self.on_partial
                    and self._speech_blocks >= self._min_speech
                    and (self._speech_blocks - self._last_partial_blocks)
                    >= self._partial_need
                ):
                    partial_audio = np.concatenate(self._buffer)
                    self._last_partial_blocks = self._speech_blocks

                if self._speech_blocks >= self._max_speech:
                    final_audio = self._take_buffer_locked()
            else:
                if self._speeching:
                    self._buffer.append(chunk)
                    self._silence_blocks += 1
                    if self._silence_blocks >= self._silence_need:
                        if self._speech_blocks >= self._min_speech:
                            final_audio = self._take_buffer_locked()
                        else:
                            self._reset_locked()

        if partial_audio is not None and self.on_partial:
            self.on_partial(partial_audio)
        if final_audio is not None:
            self.on_segment(final_audio)

    def _take_buffer_locked(self) -> np.ndarray | None:
        if not self._buffer:
            self._reset_locked()
            return None
        audio = np.concatenate(self._buffer)
        self._reset_locked()
        return audio

    def _reset_locked(self) -> None:
        self._buffer = []
        self._speeching = False
        self._silence_blocks = 0
        self._speech_blocks = 0
        self._last_partial_blocks = 0

    def flush(self) -> None:
        final_audio: np.ndarray | None = None
        with self._lock:
            if self._speeching and self._speech_blocks >= self._min_speech:
                final_audio = self._take_buffer_locked()
            else:
                self._reset_locked()
        if final_audio is not None:
            self.on_segment(final_audio)


class TimedChunkCollector:
    """固定窗口采集：听满 N 秒后整段回调一次（不边说边出）。"""

    def __init__(
        self,
        config: AppConfig,
        window_sec: float,
        on_chunk: Callable[[np.ndarray], None],
        on_progress: Callable[[float, float], None] | None = None,
    ) -> None:
        self.config = config
        self.window_sec = max(1.0, float(window_sec))
        self.on_chunk = on_chunk
        self.on_progress = on_progress
        self._buffer: list[np.ndarray] = []
        self._samples = 0
        self._lock = threading.Lock()
        self._need = int(config.sample_rate * self.window_sec)
        self._last_progress_tick = -1

    def set_window(self, window_sec: float) -> None:
        with self._lock:
            self.window_sec = max(1.0, float(window_sec))
            self._need = int(self.config.sample_rate * self.window_sec)
            self._buffer = []
            self._samples = 0
            self._last_progress_tick = -1

    def feed(self, chunk: np.ndarray) -> None:
        ready: np.ndarray | None = None
        elapsed = 0.0
        total = self.window_sec
        report = False

        with self._lock:
            self._buffer.append(chunk)
            self._samples += int(chunk.shape[0])
            elapsed = self._samples / float(self.config.sample_rate)
            # 每 0.25 秒汇报一次，保证每轮都能看到倒计时
            tick = int(elapsed * 4)
            if tick != self._last_progress_tick:
                self._last_progress_tick = tick
                report = True
            if self._samples >= self._need:
                ready = np.concatenate(self._buffer)
                self._buffer = []
                self._samples = 0
                self._last_progress_tick = -1

        if report and self.on_progress and ready is None:
            self.on_progress(min(elapsed, total), total)
        if ready is not None:
            if self.on_progress:
                self.on_progress(total, total)
            self.on_chunk(ready)

    def flush(self) -> None:
        ready: np.ndarray | None = None
        with self._lock:
            if self._samples > int(self.config.sample_rate * 0.4):
                ready = np.concatenate(self._buffer) if self._buffer else None
            self._buffer = []
            self._samples = 0
            self._last_progress_tick = -1
        if ready is not None:
            self.on_chunk(ready)


class WhisperASR:
    """懒加载 Whisper：先出界面，后台加载模型。"""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._model: WhisperModel | None = None
        self._lock = threading.Lock()
        self._load_error: str | None = None
        self.description = ""  # 例如 "large-v3-turbo · GPU"
        self.on_gpu = False

    @property
    def ready(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def _plan(self) -> list[tuple[str, str, str]]:
        """按优先级列出 (模型, 设备, 精度)；whisper_device / whisper_model 可为 auto。"""
        cfg = self.config
        plans: list[tuple[str, str, str]] = []
        if cfg.whisper_device in ("auto", "cuda") and _cuda_available():
            model = _GPU_DEFAULT_MODEL if cfg.whisper_model == "auto" else cfg.whisper_model
            plans.append((model, "cuda", "float16"))
        if cfg.whisper_device != "cuda" or not plans:
            model = _CPU_DEFAULT_MODEL if cfg.whisper_model == "auto" else cfg.whisper_model
            plans.append((model, "cpu", cfg.whisper_compute_type))
        return plans

    def load(self) -> None:
        """同步加载（应在后台线程调用）。显卡不可用时自动退回 CPU。"""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            errors: list[str] = []
            for model_name, device, compute in self._plan():
                try:
                    model = WhisperModel(
                        _model_source(model_name),
                        device=device,
                        compute_type=compute,
                        cpu_threads=min(8, os.cpu_count() or 4),
                    )
                    # cuBLAS 等库要到第一次推理才加载：先跑一次确认可用，顺便预热
                    segs, _ = model.transcribe(
                        np.zeros(16000, dtype=np.float32), language="ja", beam_size=1
                    )
                    list(segs)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{model_name}/{device}: {exc}")
                    continue
                self._model = model
                self.on_gpu = device == "cuda"
                self.description = f"{model_name} · {'GPU' if self.on_gpu else 'CPU'}"
                self._load_error = None
                return
            self._load_error = "；".join(errors)
            raise RuntimeError(self._load_error)

    def transcribe(self, audio: np.ndarray) -> TranscriptResult:
        if self._model is None:
            raise RuntimeError("识别模型尚未加载完成")

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        peak = float(np.max(np.abs(audio)) + 1e-9)
        if peak > 0:
            audio = audio / peak * 0.95

        seconds = len(audio) / float(self.config.sample_rate)
        with self._lock:
            segments, info = self._model.transcribe(
                audio,
                language=self.config.whisper_language,
                vad_filter=False,
                beam_size=1,
                best_of=1,
                condition_on_previous_text=False,
                without_timestamps=True,
            )
            texts = [s.text.strip() for s in segments if s.text and s.text.strip()]

        text = " ".join(texts).strip()
        lang = getattr(info, "language", None)
        return TranscriptResult(text=text, language=lang, audio_seconds=seconds)

    def transcribe_words(
        self,
        audio: np.ndarray,
        language: str | None = None,
        initial_prompt: str | None = None,
    ) -> tuple[list[tuple[float, float, str]], str | None]:
        """同传用：返回 [(start, end, word)]（相对 audio 起点的秒数）与语言。"""
        if self._model is None:
            raise RuntimeError("识别模型尚未加载完成")

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        peak = float(np.max(np.abs(audio)) + 1e-9)
        if peak > 0:
            audio = audio / peak * 0.95

        words: list[tuple[float, float, str]] = []
        with self._lock:
            segments, info = self._model.transcribe(
                audio,
                language=language,
                initial_prompt=initial_prompt or None,
                vad_filter=self.config.simul_vad_filter,
                vad_parameters={"min_silence_duration_ms": 300},
                beam_size=1,
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=False,
                word_timestamps=True,
            )
            for seg in segments:
                if seg.no_speech_prob > 0.6 and seg.avg_logprob < -1.0:
                    continue
                if seg.compression_ratio > 2.4:  # 复读式幻听
                    continue
                if any(m in seg.text for m in _HALLUCINATION_MARKERS):
                    continue
                for w in seg.words or []:
                    if w.word:
                        words.append((float(w.start), float(w.end), w.word))
        return words, getattr(info, "language", None)

"""应用配置：settings.json（优先）> .env > 默认值。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from settings_store import _app_dir, load_settings

_APP_DIR = _app_dir()
load_dotenv(_APP_DIR / ".env")


def _s(key: str, default: str = "") -> str:
    saved = load_settings()
    if key in saved and saved[key] not in (None, ""):
        return str(saved[key])
    return os.getenv(key.upper() if key.startswith("deepseek") else key, default)


@dataclass
class AppConfig:
    # 音频
    sample_rate: int = 16000
    channels: int = 1
    block_ms: int = 30
    silence_ms: int = 380
    min_speech_ms: int = 280
    max_speech_ms: int = 2500
    partial_ms: int = 1000
    rms_threshold: float = 0.012

    # ASR
    whisper_model: str = "auto"  # auto = 有显卡用 large-v3-turbo，否则 base
    whisper_device: str = "auto"  # auto / cuda / cpu
    whisper_compute_type: str = "int8"
    whisper_language: str | None = None

    # 翻译 API（可在软件设置里改，写入 settings.json）
    deepseek_api_key: str = field(default="")
    deepseek_base_url: str = field(default="")
    deepseek_model: str = field(default="")
    translate_min_interval: float = 2.5
    translate_partial: bool = False
    reasoning_effort: str = "off"
    capture_mode: str = "simul"
    cached_models: list[str] = field(default_factory=list)

    # 同声传译（capture_mode = simul）
    simul_min_chunk_ms: int = 500  # 每攒多少新音频跑一轮识别
    simul_endpoint_ms: int = 600  # 静音多久算一句话结束
    simul_buffer_trim_sec: float = 6.0  # 缓冲超过此长度就在句末截断
    simul_buffer_max_sec: float = 12.0  # 缓冲硬上限
    # 显卡识别很快，可用更长缓冲换准确率（实测歌曲同传精确率 62% → 87%）
    simul_buffer_trim_sec_gpu: float = 12.0
    simul_buffer_max_sec_gpu: float = 20.0
    simul_idle_flush_ms: int = 900  # 已确认文本停顿多久就送翻译
    simul_min_unit_chars: int = 8  # 遇逗号切意群的最少字数
    simul_max_unit_chars: int = 24  # 无标点时的意群最大字数
    simul_min_interval: float = 0.0  # 同传翻译请求最小间隔（秒）
    # Silero VAD 只认说话声，会把伴奏里的歌声整段丢掉（实测整首歌 289s 只保留 0.8s）；
    # 默认关闭，静音段由能量门限跳过，幻听由 no_speech / 复读 / 关键词过滤兜底
    simul_vad_filter: bool = False
    simul_retranslate: bool = True  # 本地模型时：每次识别更新都重译整句（更快出字）

    # 本地翻译模型（llama.cpp + GGUF，如 Sakura；只负责日→中）
    translator_backend: str = "api"  # api / local
    local_model_path: str = ""  # 留空 = 自动用 models\*.gguf
    local_server_exe: str = ""  # 留空 = tools\llama\llama-server.exe
    local_port: int = 8089
    local_gpu_layers: int = 99  # 放到显卡的层数；0 = 纯 CPU

    # UI
    overlay_width: int = 1080
    overlay_height: int = 168
    bottom_margin: int = 72
    font_source: str = "Microsoft YaHei UI"
    font_target: str = "Yu Gothic UI"
    history_limit: int = 50
    translate_mode: str = "ja2zh"


def get_config() -> AppConfig:
    saved = load_settings()
    cfg = AppConfig()

    cfg.deepseek_api_key = str(
        saved.get("deepseek_api_key")
        or os.getenv("DEEPSEEK_API_KEY", "")
    )
    cfg.deepseek_base_url = str(
        saved.get("deepseek_base_url")
        or os.getenv("DEEPSEEK_BASE_URL", "http://127.0.0.1:7863/v1")
    )
    cfg.deepseek_model = str(
        saved.get("deepseek_model")
        or os.getenv("DEEPSEEK_MODEL", "deepseek-v4.1-flash")
    )
    cfg.reasoning_effort = str(saved.get("reasoning_effort") or "off")
    cfg.capture_mode = str(saved.get("capture_mode") or "simul")
    cfg.translate_mode = str(saved.get("translate_mode") or "ja2zh")
    if saved.get("translator_backend") in ("api", "local"):
        cfg.translator_backend = str(saved["translator_backend"])
    if saved.get("local_model_path"):
        cfg.local_model_path = str(saved["local_model_path"])
    if "translate_min_interval" in saved:
        try:
            cfg.translate_min_interval = float(saved["translate_min_interval"])
        except (TypeError, ValueError):
            pass
    if saved.get("whisper_model"):
        cfg.whisper_model = str(saved["whisper_model"])
    lang = saved.get("whisper_language")
    if lang in ("zh", "ja"):
        cfg.whisper_language = lang
    models = saved.get("cached_models")
    if isinstance(models, list):
        cfg.cached_models = [str(x) for x in models if x]

    return cfg

"""应用设置持久化（可随项目跨电脑拷贝 settings.json）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _app_dir() -> Path:
    """开发环境=源码目录；打包后=exe 同目录。"""
    import sys

    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


SETTINGS_PATH = _app_dir() / "settings.json"

# 会写入 settings.json 的字段
PERSIST_KEYS = (
    "deepseek_api_key",
    "deepseek_base_url",
    "deepseek_model",
    "reasoning_effort",
    "capture_mode",
    "translate_mode",
    "translate_min_interval",
    "whisper_model",
    "whisper_language",
    "cached_models",
    "translator_backend",
    "local_model_path",
)


def load_settings() -> dict[str, Any]:
    if not SETTINGS_PATH.exists():
        return {}
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_settings(data: dict[str, Any]) -> None:
    existing = load_settings()
    existing.update({k: v for k, v in data.items() if k in PERSIST_KEYS or k == "cached_models"})
    SETTINGS_PATH.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_from_config(config: object, extra: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {}
    for key in PERSIST_KEYS:
        if key == "cached_models":
            continue
        if hasattr(config, key):
            payload[key] = getattr(config, key)
    if extra:
        payload.update(extra)
    save_settings(payload)

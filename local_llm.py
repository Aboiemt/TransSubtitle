"""本地 llama.cpp 翻译服务：自动启动 llama-server 加载 Sakura 等 GGUF 模型。"""

from __future__ import annotations

import json
import subprocess
import threading
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

from config import AppConfig
from settings_store import _app_dir

# 模型下拉框里代表本地模型的条目
LOCAL_MODEL_LABEL = "本地 Sakura"

# Sakura v1.0 官方提示词
SAKURA_SYSTEM = (
    "你是一个轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文，"
    "并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。"
)
SAKURA_USER = "将下面的日文文本翻译成中文："


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _app_dir() / p


def find_server_exe(config: AppConfig) -> Path | None:
    candidates = []
    if config.local_server_exe:
        candidates.append(_resolve(config.local_server_exe))
    candidates.append(_app_dir() / "tools" / "llama" / "llama-server.exe")
    return next((p for p in candidates if p.is_file()), None)


def find_model(config: AppConfig) -> Path | None:
    if config.local_model_path:
        p = _resolve(config.local_model_path)
        return p if p.is_file() else None
    models = list((_app_dir() / "models").glob("*.gguf"))
    # 优先量化版（更快），fp16 放最后
    models.sort(key=lambda p: ("fp16" in p.name.lower() or "f16" in p.name.lower(), p.name))
    return models[0] if models else None


class LocalLLMServer:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self.ready = False
        self.error: str | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.config.local_port}/v1"

    def _healthy(self) -> bool:
        url = f"http://127.0.0.1:{self.config.local_port}/health"
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                return json.loads(resp.read().decode("utf-8")).get("status") == "ok"
        except Exception:  # noqa: BLE001
            return False

    def _warmup(self) -> None:
        """首个请求要编译 GPU 着色器（数秒），启动时先跑一次。"""
        body = json.dumps(
            {
                "messages": [
                    {"role": "system", "content": SAKURA_SYSTEM},
                    {"role": "user", "content": SAKURA_USER + "こんにちは。"},
                ],
                "max_tokens": 8,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(req, timeout=60).read()
        except Exception:  # noqa: BLE001
            pass

    def start(self, on_status: Callable[[str], None] | None = None) -> bool:
        """阻塞启动（请在后台线程调用）。成功返回 True。"""
        say = on_status or (lambda _s: None)
        with self._lock:
            if self.ready and (self._proc is None or self._proc.poll() is None):
                return True
            self.ready = False
            if self._healthy():  # 端口上已有服务（例如上次未退出的进程），直接复用
                say("本地模型预热中…")
                self._warmup()
                self.ready = True
                return True

            exe = find_server_exe(self.config)
            model = find_model(self.config)
            if exe is None:
                self.error = "未找到 tools\\llama\\llama-server.exe"
                return False
            if model is None:
                self.error = "未找到 models\\*.gguf 模型文件"
                return False

            say(f"正在加载本地模型 {model.name}…")
            log = open(_app_dir() / "llama-server.log", "w", encoding="utf-8")  # noqa: SIM115
            args = [
                str(exe),
                "-m", str(model),
                "-ngl", str(self.config.local_gpu_layers),
                "-c", "4096",
                "-np", "1",
                "--host", "127.0.0.1",
                "--port", str(self.config.local_port),
            ]
            try:
                self._proc = subprocess.Popen(
                    args,
                    cwd=str(exe.parent),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except OSError as exc:
                self.error = f"启动 llama-server 失败: {exc}"
                return False

            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if self._proc.poll() is not None:
                    self.error = "llama-server 已退出，详见 llama-server.log"
                    self._proc = None
                    return False
                if self._healthy():
                    break
                time.sleep(0.5)
            else:
                self.error = "本地模型加载超时"
                return False

            say("本地模型预热中…")
            self._warmup()
            self.error = None
            self.ready = True
            return True

    def stop(self) -> None:
        with self._lock:
            self.ready = False
            if self._proc is not None and self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
            self._proc = None

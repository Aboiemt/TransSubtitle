"""Windows 系统声卡环回采集（soundcard / WASAPI loopback）。"""

from __future__ import annotations

import queue
import threading
from typing import Callable

import numpy as np
import soundcard as sc

from config import AppConfig


def list_loopback_devices() -> list[dict]:
    result: list[dict] = []
    for i, mic in enumerate(sc.all_microphones(include_loopback=True)):
        if not getattr(mic, "isloopback", False):
            continue
        result.append(
            {
                "index": i,
                "name": mic.name,
                "hostapi": "WASAPI Loopback",
                "channels": int(getattr(mic, "channels", 2) or 2),
                "default_samplerate": 48000.0,
                "is_wasapi": True,
                "id": str(mic.id),
            }
        )
    return result


def list_input_devices() -> list[dict]:
    result: list[dict] = []
    for i, mic in enumerate(sc.all_microphones(include_loopback=True)):
        if getattr(mic, "isloopback", False):
            continue
        result.append(
            {
                "index": i,
                "name": mic.name,
                "hostapi": "Input",
                "channels": int(getattr(mic, "channels", 1) or 1),
                "default_samplerate": 48000.0,
                "is_wasapi": False,
                "id": str(mic.id),
            }
        )
    return result


def _all_mics_with_loopback() -> list:
    return list(sc.all_microphones(include_loopback=True))


def pick_default_loopback_mic():
    """
    跟随系统默认扬声器环回。
    若默认是 ToDesk 等远程虚拟声卡，同时提供 Realtek 本机环回更稳，
    仍优先默认扬声器（保证「听到什么录什么」）。
    """
    mics = _all_mics_with_loopback()
    loopbacks = [m for m in mics if getattr(m, "isloopback", False)]
    if not loopbacks:
        return None

    try:
        spk = sc.default_speaker()
        for m in loopbacks:
            if spk.name == m.name or spk.name in m.name or m.name in spk.name:
                return m
    except Exception:
        pass

    for m in loopbacks:
        if "realtek" in m.name.lower():
            return m
    return loopbacks[0]


def resolve_mic(device: int | None):
    """device 为 all_microphones(include_loopback=True) 的枚举下标。"""
    mics = _all_mics_with_loopback()
    if device is None:
        return pick_default_loopback_mic()
    if 0 <= device < len(mics):
        return mics[device]
    return None


def _resample_linear(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    duration = audio.shape[0] / float(src_rate)
    dst_len = max(1, int(round(duration * dst_rate)))
    x_old = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=dst_len, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


class SystemAudioCapture:
    """持续采集系统播放声，回调 float32 mono @ config.sample_rate。"""

    def __init__(
        self,
        config: AppConfig,
        on_audio: Callable[[np.ndarray], None] | None = None,
        device: int | None = None,
        use_loopback: bool | None = None,
    ) -> None:
        del use_loopback
        self.config = config
        self.on_audio = on_audio
        self.device = device
        self.mic = resolve_mic(device)
        self.use_loopback = bool(getattr(self.mic, "isloopback", False)) if self.mic else True
        self._capture_rate = 48000
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)
        self._running = False
        self._thread: threading.Thread | None = None
        self._worker: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._running

    @property
    def device_name(self) -> str:
        if self.mic is None:
            return "(未找到环回设备)"
        tag = " [loopback]" if self.use_loopback else ""
        return f"{self.mic.name}{tag}"

    def _push(self, mono16k: np.ndarray) -> None:
        block = max(1, int(self.config.sample_rate * self.config.block_ms / 1000))
        for i in range(0, len(mono16k), block):
            chunk = mono16k[i : i + block]
            if chunk.size == 0:
                continue
            try:
                self._q.put_nowait(chunk)
            except queue.Full:
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._q.put_nowait(chunk)
                except queue.Full:
                    pass

    def _drain(self) -> None:
        while self._running:
            try:
                chunk = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            if self.on_audio:
                self.on_audio(chunk)

    def _capture_loop(self) -> None:
        assert self.mic is not None
        # 每次读约 100ms，延迟低且稳
        frames = max(256, int(self._capture_rate * 0.1))
        try:
            with self.mic.recorder(
                samplerate=self._capture_rate,
                channels=2,
                blocksize=frames,
            ) as rec:
                while self._running:
                    data = rec.record(numframes=frames)
                    if data is None or len(data) == 0:
                        continue
                    mono = np.mean(np.asarray(data, dtype=np.float32), axis=1)
                    mono16k = _resample_linear(
                        mono, self._capture_rate, self.config.sample_rate
                    )
                    self._push(mono16k)
        except Exception as exc:
            # 让 UI 侧能感知：塞一个空块并打印
            print(f"[audio] 采集线程异常: {exc}")

    def start(self) -> None:
        if self._running:
            return
        self.mic = resolve_mic(self.device)
        if self.mic is None:
            raise RuntimeError(
                "未找到可用环回设备。请运行: python main.py --list-devices"
            )
        self.use_loopback = bool(getattr(self.mic, "isloopback", False))

        self._running = True
        self._worker = threading.Thread(target=self._drain, daemon=True)
        self._worker.start()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=1.0)
        self._worker = None

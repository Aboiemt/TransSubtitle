"""中日双语字幕：系统声音 → Whisper → LLM（同传 / 实时 / 定时整段）。"""

from __future__ import annotations

import argparse
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from asr import SpeechSegmenter, TimedChunkCollector, WhisperASR
from audio_capture import SystemAudioCapture, list_input_devices, list_loopback_devices
from config import AppConfig, get_config
from local_llm import LOCAL_MODEL_LABEL, LocalLLMServer
from settings_store import save_from_config
from settings_ui import SettingsWindow
from simul import SimulInterpreter
from translator import DeepSeekTranslator
from ui import SubtitleOverlay


class RealtimeSubtitleApp:
    def __init__(
        self,
        config: AppConfig,
        device: int | None = None,
        use_loopback: bool | None = None,
    ) -> None:
        self.config = config
        self.asr = WhisperASR(config)  # 不立刻加载权重
        self.translator = DeepSeekTranslator(config)
        self.local_llm = LocalLLMServer(config)
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._partial_lock = threading.Lock()
        self._final_lock = threading.Lock()
        self._gen = 0
        self._gen_lock = threading.Lock()
        self._device = device
        self._use_loopback = use_loopback
        self._settings_win: SettingsWindow | None = None

        self.segmenter = SpeechSegmenter(
            config,
            on_segment=self._on_final,
            on_partial=self._on_partial,
        )
        self.timed = TimedChunkCollector(
            config,
            window_sec=self._mode_seconds(config.capture_mode) or 5.0,
            on_chunk=self._on_timed_chunk,
            on_progress=self._on_timed_progress,
        )

        self.capture = SystemAudioCapture(
            config,
            on_audio=self._on_audio,
            device=device,
            use_loopback=use_loopback,
        )

        self.ui = SubtitleOverlay(
            config,
            on_start=self.start_capture,
            on_stop=self.stop_capture,
            on_mode_change=self._on_mode_change,
            on_model_change=self._on_model_change,
            on_capture_mode_change=self._on_capture_mode_change,
            on_open_settings=self.open_settings,
            device_label=self.capture.device_name,
        )

        self.simul = SimulInterpreter(
            config,
            self.asr,
            get_translator=lambda: self.translator,
            on_view=self.ui.push_simul,
            on_status=self.ui.post_status,
        )

        # UI 先出来，Whisper 后台加载
        self.ui.after(50, self._start_asr_loading)
        if config.translator_backend == "local":
            self.ui.after(100, self._start_local_llm)

        if not self.translator.ready:
            self.ui.after(400, self.open_settings)
        elif not config.cached_models:
            self.ui.after(600, self._silent_fetch_models)

    def _start_asr_loading(self) -> None:
        self.ui.set_asr_ready(
            False,
            "正在加载语音识别模型（首次使用会下载），界面可先设置…",
        )

        def work() -> None:
            try:
                self.asr.load()
                self.ui.after(
                    0,
                    lambda: self.ui.set_asr_ready(
                        True, f"识别模型已就绪（{self.asr.description}），点击开始"
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                msg = f"识别模型加载失败: {exc}"
                self.ui.after(0, lambda: self.ui.set_asr_ready(False, msg))

        threading.Thread(target=work, daemon=True).start()

    def _start_local_llm(self) -> None:
        """后台启动 llama-server（加载 + 预热约数秒）。"""

        def work() -> None:
            ok = self.local_llm.start(on_status=self.ui.post_status)
            if ok:
                self.ui.post_status("本地 Sakura 已就绪（日→中走本地，其它方向走 API）")
            else:
                self.ui.post_status(f"本地模型启动失败: {self.local_llm.error}")

        threading.Thread(target=work, daemon=True).start()

    def open_settings(self) -> None:
        if self._settings_win is not None and self._settings_win.winfo_exists():
            self._settings_win.focus()
            return
        self._settings_win = SettingsWindow(
            self.ui,
            self.config,
            on_saved=self._on_settings_saved,
        )

    def _on_settings_saved(self, config: AppConfig, models: list[str]) -> None:
        self.config = config
        self.translator = DeepSeekTranslator(config)
        self.ui.set_model_list(models, config.deepseek_model)
        self.ui.set_status(f"已应用设置 · 模型 {config.deepseek_model}")

    def _silent_fetch_models(self) -> None:
        def work() -> None:
            try:
                from api_models import fetch_remote_models

                models = fetch_remote_models(
                    self.config.deepseek_api_key,
                    self.config.deepseek_base_url,
                )
                if not models:
                    return

                def apply() -> None:
                    self.config.cached_models = models
                    if self.config.deepseek_model not in models:
                        self.config.deepseek_model = models[0]
                    save_from_config(
                        self.config,
                        extra={"cached_models": models},
                    )
                    self.ui.set_model_list(models, self.config.deepseek_model)
                    self.ui.set_status(f"已同步上游模型 {len(models)} 个")

                self.ui.after(0, apply)
            except Exception:
                pass

        threading.Thread(target=work, daemon=True).start()

    @staticmethod
    def _mode_seconds(mode: str) -> float | None:
        if mode == "chunk5":
            return 5.0
        if mode == "chunk10":
            return 10.0
        return None

    def _is_timed(self) -> bool:
        return self._mode_seconds(self.config.capture_mode) is not None

    def _is_simul(self) -> bool:
        return self.config.capture_mode == "simul"

    def _on_audio(self, chunk: np.ndarray) -> None:
        if self._is_simul():
            self.simul.feed(chunk)
        elif self._is_timed():
            self.timed.feed(chunk)
        else:
            self.segmenter.feed(chunk)

    def _bump_gen(self) -> int:
        with self._gen_lock:
            self._gen += 1
            return self._gen

    def _current_gen(self) -> int:
        with self._gen_lock:
            return self._gen

    def _on_mode_change(self, mode: str) -> None:
        self.config.translate_mode = mode
        save_from_config(self.config)
        self.ui.set_status(f"翻译方向: {mode}")

    def _on_model_change(self, model: str) -> None:
        if model == LOCAL_MODEL_LABEL:
            self.config.translator_backend = "local"
            self.translator = DeepSeekTranslator(self.config)
            save_from_config(self.config)
            self._start_local_llm()
            return
        self.config.translator_backend = "api"
        self.config.deepseek_model = model
        self.translator = DeepSeekTranslator(self.config)
        save_from_config(self.config)
        # 释放显存
        threading.Thread(target=self.local_llm.stop, daemon=True).start()
        self.ui.set_status(f"已切换模型: {model}")

    def _on_capture_mode_change(self, mode: str) -> None:
        self.config.capture_mode = mode
        save_from_config(self.config)
        capturing = self.capture.running
        if mode == "simul":
            if capturing:
                self.simul.start()
            self.ui.set_status("同传模式：边听边译，滚动字幕")
            return
        if self.simul.running:
            self.simul.stop()
        secs = self._mode_seconds(mode)
        if secs is not None:
            self.timed.set_window(secs)
            self.ui.set_status(f"定时模式：听满 {int(secs)} 秒后识别翻译")
        else:
            self.ui.set_status("实时模式：边说边出字幕")

    def start_capture(self) -> None:
        if not self.asr.ready:
            self.ui.set_status("识别模型还在加载，请稍候…")
            return
        try:
            if self._is_simul():
                self.simul.start()
            self.capture.start()
            self.ui.set_device(self.capture.device_name)
            secs = self._mode_seconds(self.config.capture_mode)
            if self._is_simul():
                self.ui.set_status("同传中 · 正在听…")
            elif secs:
                self.timed.set_window(secs)
                self.ui.set_status(f"录音中 0/{int(secs)}s · 还剩 {int(secs)}s")
            else:
                self.ui.set_status("正在监听系统声音…")
        except Exception as exc:
            self.ui.set_status(f"启动失败: {exc}")
            traceback.print_exc()

    def stop_capture(self) -> None:
        self.capture.stop()
        if self.simul.running:
            self.simul.stop()
        elif self._is_timed():
            self.timed.flush()
        else:
            self.segmenter.flush()
        self.ui.set_status("已停止")

    def _on_timed_progress(self, elapsed: float, total: float) -> None:
        remain = max(0, int(round(total - elapsed)))
        # 显示「已录/总时长」和「还剩几秒」，每轮都会刷新
        self.ui.set_status(
            f"录音中 {min(int(elapsed), int(total))}/{int(total)}s · 还剩 {remain}s"
        )

    def _on_timed_chunk(self, audio: np.ndarray) -> None:
        gen = self._bump_gen()
        self.executor.submit(self._process_final, audio, gen, timed=True)

    def _on_partial(self, audio: np.ndarray) -> None:
        if self._is_timed():
            return
        if not self._partial_lock.acquire(blocking=False):
            return
        self.executor.submit(self._process_partial, audio)

    def _on_final(self, audio: np.ndarray) -> None:
        if self._is_timed():
            return
        gen = self._bump_gen()
        self.executor.submit(self._process_final, audio, gen, timed=False)

    def _process_partial(self, audio: np.ndarray) -> None:
        try:
            result = self.asr.transcribe(audio)
            if not result.text:
                return
            placeholder = (
                self.ui._target_text if getattr(self.ui, "_target_text", "") else "…"
            )
            self.ui.push_subtitle(
                result.text,
                placeholder if placeholder != "…" else "（识别中，定稿后翻译）",
                "实时识别中…",
                interim=True,
            )
            if not getattr(self.config, "translate_partial", False):
                return

            gen = self._current_gen()

            def on_delta(partial_tr: str) -> None:
                if self._current_gen() != gen:
                    return
                self.ui.push_subtitle(
                    result.text, partial_tr, "实时翻译中…", interim=True
                )

            self.translator.translate(
                result.text, result.language, on_delta=on_delta
            )
        except Exception as exc:
            self.ui.set_status(f"临时识别错误: {exc}")
            traceback.print_exc()
        finally:
            self._partial_lock.release()

    def _process_final(
        self, audio: np.ndarray, gen: int, timed: bool = False
    ) -> None:
        with self._final_lock:
            try:
                self.ui.set_status("识别中…" if timed else "定稿识别…")
                result = self.asr.transcribe(audio)
                if not result.text:
                    secs = self._mode_seconds(self.config.capture_mode)
                    if secs:
                        self.ui.set_status(f"本段无语音，继续录音 0/{int(secs)}s…")
                    else:
                        self.ui.set_status("监听中…")
                    return

                self.ui.push_subtitle(result.text, "…", "翻译中…", interim=True)

                def on_delta(partial_tr: str) -> None:
                    if self._current_gen() != gen:
                        return
                    self.ui.push_subtitle(
                        result.text, partial_tr, "翻译中…", interim=True
                    )

                tr = self.translator.translate(
                    result.text, result.language, on_delta=on_delta
                )
                if self._current_gen() != gen:
                    return
                meta = f"{tr.source_lang}→{tr.target_lang} | {result.audio_seconds:.1f}s"
                self.ui.push_subtitle(
                    tr.source_text, tr.target_text, meta, interim=False
                )
                secs = self._mode_seconds(self.config.capture_mode)
                if secs:
                    self.ui.set_status(f"下一段录音中 0/{int(secs)}s…")
            except Exception as exc:
                self.ui.set_status(f"处理错误: {exc}")
                traceback.print_exc()

    def run(self) -> None:
        self.ui.mainloop()
        # 窗口已销毁，不能再走 stop_capture（会更新已不存在的控件）
        self.capture.stop()
        self.simul.stop()
        self.local_llm.stop()
        self.executor.shutdown(wait=False, cancel_futures=True)


def cmd_list_devices() -> None:
    print("WASAPI 环回设备（捕获系统播放声，推荐）：")
    loops = list_loopback_devices()
    if not loops:
        print("  (无)")
    for d in loops:
        print(
            f"  [{d['index']}] {d['name']} | ch={d['channels']} | "
            f"sr={d['default_samplerate']}"
        )
    print("\n普通输入设备（麦克风等）：")
    for d in list_input_devices():
        print(
            f"  [{d['index']}] {d['name']} | ch={d['channels']} | "
            f"sr={d['default_samplerate']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="中日双语实时字幕翻译")
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="列出音频输入设备后退出",
    )
    parser.add_argument("--device", type=int, default=None, help="音频设备索引")
    parser.add_argument(
        "--loopback",
        action="store_true",
        help="对指定/默认输出设备启用 WASAPI 环回",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Whisper 模型: auto/base/small/large-v3-turbo（auto = 有显卡用 large-v3-turbo）",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "ja2zh", "zh2ja"],
        default=None,
        help="翻译方向",
    )
    parser.add_argument(
        "--capture",
        choices=["simul", "realtime", "chunk5", "chunk10"],
        default=None,
        help="采集模式：同传 / 实时 / 听5秒 / 听10秒",
    )
    parser.add_argument(
        "--lang",
        choices=["zh", "ja"],
        default=None,
        help="强制 ASR 语言（默认自动）",
    )
    args = parser.parse_args()

    if args.list_devices:
        cmd_list_devices()
        return 0

    config = get_config()
    if args.model:
        config.whisper_model = args.model
    if args.mode:
        config.translate_mode = args.mode
    if args.lang:
        config.whisper_language = args.lang
    if args.capture:
        config.capture_mode = args.capture

    print(f"Whisper: {config.whisper_model} ({config.whisper_device}) · 后台加载，auto = 有显卡用 large-v3-turbo")
    if config.translator_backend == "local":
        print(f"LLM 模型: {LOCAL_MODEL_LABEL}（日→中本地，其它方向 {config.deepseek_model}）")
    else:
        print(f"LLM 模型: {config.deepseek_model}")
    print(f"翻译方向: {config.translate_mode}")
    print(f"采集模式: {config.capture_mode} | 思考: {config.reasoning_effort}")

    use_loopback = True if args.loopback else None
    app = RealtimeSubtitleApp(config, device=args.device, use_loopback=use_loopback)
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

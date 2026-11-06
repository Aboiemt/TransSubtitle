"""参考图风格的无边框深色胶囊字幕浮层。"""

from __future__ import annotations

import queue
import tkinter as tk
from collections.abc import Callable

import customtkinter as ctk

from config import AppConfig
from local_llm import LOCAL_MODEL_LABEL

# 与系统标题栏抠透明，做出圆角胶囊外形
_CHROMA = "#010203"

_MODE_LABELS = {
    "ja2zh": "日 → 中",
    "zh2ja": "中 → 日",
    "auto": "自动",
}
_MODE_VALUES = {v: k for k, v in _MODE_LABELS.items()}

_DISPLAY_MODES = ("双语", "原文", "译文")

_CAPTURE_LABELS = {
    "simul": "同传",
    "realtime": "实时",
    "chunk5": "听5秒",
    "chunk10": "听10秒",
}
_CAPTURE_VALUES = {v: k for k, v in _CAPTURE_LABELS.items()}

# 同传滚动字幕按语言选字体（Yu Gothic 缺简体字形，YaHei 含假名）
_LANG_FONT = {"ja": "Yu Gothic UI", "zh": "Microsoft YaHei UI"}
_MODE_LANGS = {"ja2zh": ("ja", "zh"), "zh2ja": ("zh", "ja")}

_FALLBACK_MODELS = (
    "deepseek-v4.1-flash",
    "hy3",
    "glm-5.3-flash",
    "glm-5.3",
    "minimax-m3",
    "deepseek-v4-pro",
    "auto",
)


class SubtitleOverlay(ctk.CTk):
    def __init__(
        self,
        config: AppConfig,
        on_start: Callable[[], None] | None = None,
        on_stop: Callable[[], None] | None = None,
        on_mode_change: Callable[[str], None] | None = None,
        on_model_change: Callable[[str], None] | None = None,
        on_capture_mode_change: Callable[[str], None] | None = None,
        on_open_settings: Callable[[], None] | None = None,
        device_label: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.on_start = on_start
        self.on_stop = on_stop
        self.on_mode_change = on_mode_change
        self.on_model_change = on_model_change
        self.on_capture_mode_change = on_capture_mode_change
        self.on_open_settings = on_open_settings
        self._device_label = device_label
        self._model_ids: list[str] = list(
            config.cached_models or _FALLBACK_MODELS
        )
        if config.deepseek_model and config.deepseek_model not in self._model_ids:
            self._model_ids = [config.deepseek_model, *self._model_ids]
        self._model_ids = [LOCAL_MODEL_LABEL, *self._model_ids]
        self._ui_q: queue.Queue[tuple[str, str, str, bool]] = queue.Queue()
        self._running = False
        self._locked = False
        self._pinned = True
        self._asr_ready = False
        self._display_mode = "双语"
        self._source_text = ""
        self._target_text = ""
        self._interim = False
        self._status = "正在后台加载语音识别模型…"
        # 同传视图：(原文已确认, 原文未确认, 译文已完成, 译文生成中)
        self._simul_view: tuple[str, str, str, str] = ("", "", "", "")
        self._simul_dirty = False
        self._status_q: queue.Queue[str] = queue.Queue()

        ctk.set_appearance_mode("dark")
        self.title("中日双语实时字幕")
        w = max(config.overlay_width, 980)
        h = max(config.overlay_height, 168)
        self.minsize(720, 140)
        self._win_w = w
        self._base_height = h
        self.overrideredirect(True)
        self.configure(fg_color=_CHROMA)
        self.wm_attributes("-transparentcolor", _CHROMA)
        self.attributes("-topmost", True)
        self.attributes("-alpha", 0.96)
        # 先设尺寸，稍后贴底（避开任务栏）
        self.geometry(f"{w}x{h}+0+0")
        self.after(50, self._place_bottom)

        self._drag_x = 0
        self._drag_y = 0

        # 圆角胶囊主体
        self.shell = ctk.CTkFrame(
            self,
            fg_color="#0B0B0D",
            corner_radius=28,
            border_width=1,
            border_color="#2A2A30",
        )
        self.shell.pack(fill="both", expand=True, padx=2, pady=2)
        self.shell.bind("<ButtonPress-1>", self._start_drag)
        self.shell.bind("<B1-Motion>", self._on_drag)

        # —— 左上角常驻状态（录音倒计时等，有字幕时也不隐藏）——
        self.status_badge = ctk.CTkLabel(
            self.shell,
            text=self._status,
            font=(config.font_source, 12),
            text_color="#7A8A9A",
            anchor="w",
        )
        self.status_badge.place(relx=0.0, rely=0.0, anchor="nw", x=18, y=12)

        # —— 右上角工具 ——
        tools = ctk.CTkFrame(self.shell, fg_color="transparent")
        tools.place(relx=1.0, rely=0.0, anchor="ne", x=-14, y=10)

        self.btn_region = self._icon_btn(tools, "⬚", self._noop, tip="区域（预留）")
        self.btn_lock = self._icon_btn(tools, "🔒", self._toggle_lock, tip="锁定位置")
        self.btn_pin = self._icon_btn(
            tools, "📌", self._toggle_pin, tip="置顶", active=True
        )

        sep = ctk.CTkFrame(tools, width=1, height=18, fg_color="#3A3A42")
        sep.pack(side="left", padx=8)

        self.btn_more = self._icon_btn(tools, "⋯", self._show_more, tip="更多")
        self.btn_settings = self._icon_btn(tools, "⚙", self._open_settings, tip="API设置")
        self.btn_min = self._icon_btn(tools, "─", self._minimize, tip="最小化")
        self.btn_close = self._icon_btn(tools, "✕", self._on_close, tip="关闭")

        for wdg in (
            self.btn_region,
            self.btn_lock,
            self.btn_pin,
            self.btn_more,
            self.btn_settings,
            self.btn_min,
            self.btn_close,
            sep,
        ):
            wdg.bind("<ButtonPress-1>", lambda e: "break")

        # —— 字幕区 ——
        text_box = ctk.CTkFrame(self.shell, fg_color="transparent")
        text_box.pack(fill="both", expand=True, padx=28, pady=(36, 52))
        text_box.bind("<ButtonPress-1>", self._start_drag)
        text_box.bind("<B1-Motion>", self._on_drag)

        self.source_label = ctk.CTkLabel(
            text_box,
            text="",
            font=(config.font_source, 17),
            text_color="#D0D0D6",
            wraplength=w - 80,
            justify="left",
            anchor="w",
        )
        self.source_label.pack(fill="x", anchor="w")

        self.target_label = ctk.CTkLabel(
            text_box,
            text="",
            font=(config.font_target, 20),
            text_color="#FFFFFF",
            wraplength=w - 80,
            justify="left",
            anchor="w",
        )
        self.target_label.pack(fill="x", anchor="w", pady=(6, 0))

        self.hint_label = ctk.CTkLabel(
            text_box,
            text=self._status,
            font=(config.font_source, 14),
            text_color="#6A6A74",
            anchor="w",
        )
        self.hint_label.pack(fill="x", anchor="w", pady=(8, 0))
        self.text_box = text_box

        # —— 同传滚动字幕：多行滚动，未确认部分颜色更暗 ——
        self.simul_frame = ctk.CTkFrame(text_box, fg_color="transparent")
        self.simul_src = self._make_ticker(self.simul_frame, 17, 1, "#D0D0D6", "#6E6E78")
        self.simul_tgt = self._make_ticker(self.simul_frame, 20, 2, "#FFFFFF", "#8FA3B8")
        self.simul_src.pack(fill="x", anchor="w")
        self.simul_tgt.pack(fill="x", anchor="w", pady=(4, 0))
        self._apply_simul_fonts()

        # —— 底部控制条 ——
        bar = ctk.CTkFrame(self.shell, fg_color="transparent")
        bar.place(relx=0.5, rely=1.0, anchor="s", y=-14)

        self.btn_mic = ctk.CTkButton(
            bar,
            text="●",
            width=36,
            height=36,
            corner_radius=18,
            fg_color="#1C1C22",
            hover_color="#2C2C34",
            text_color="#E8E8EE",
            font=("", 14),
            command=self._toggle,
            state="disabled",
        )
        self.btn_mic.pack(side="left", padx=(0, 8))

        if config.translator_backend == "local":
            model_key = LOCAL_MODEL_LABEL
        else:
            model_key = config.deepseek_model or (
                self._model_ids[1] if len(self._model_ids) > 1 else "auto"
            )
        self.model_var = tk.StringVar(value=model_key)
        self.model_menu = ctk.CTkOptionMenu(
            bar,
            values=self._model_ids or [model_key],
            variable=self.model_var,
            width=150,
            height=34,
            corner_radius=17,
            fg_color="#1C1C22",
            button_color="#1C1C22",
            button_hover_color="#2C2C34",
            dropdown_fg_color="#1A1A20",
            dropdown_hover_color="#2C2C34",
            text_color="#E8E8EE",
            font=(config.font_source, 12),
            dropdown_font=(config.font_source, 12),
            command=self._change_model,
        )
        self.model_menu.pack(side="left", padx=(0, 8))

        mode_label = _MODE_LABELS.get(config.translate_mode, "日 → 中")
        self.dir_var = tk.StringVar(value=mode_label)
        self.dir_menu = ctk.CTkOptionMenu(
            bar,
            values=list(_MODE_LABELS.values()),
            variable=self.dir_var,
            width=96,
            height=34,
            corner_radius=17,
            fg_color="#1C1C22",
            button_color="#1C1C22",
            button_hover_color="#2C2C34",
            dropdown_fg_color="#1A1A20",
            dropdown_hover_color="#2C2C34",
            text_color="#E8E8EE",
            font=(config.font_source, 13),
            dropdown_font=(config.font_source, 13),
            command=self._change_dir,
        )
        self.dir_menu.pack(side="left", padx=(0, 8))

        capture_label = _CAPTURE_LABELS.get(config.capture_mode, "同传")
        self.capture_var = tk.StringVar(value=capture_label)
        self.capture_menu = ctk.CTkOptionMenu(
            bar,
            values=list(_CAPTURE_LABELS.values()),
            variable=self.capture_var,
            width=88,
            height=34,
            corner_radius=17,
            fg_color="#1C1C22",
            button_color="#1C1C22",
            button_hover_color="#2C2C34",
            dropdown_fg_color="#1A1A20",
            dropdown_hover_color="#2C2C34",
            text_color="#E8E8EE",
            font=(config.font_source, 13),
            dropdown_font=(config.font_source, 13),
            command=self._change_capture,
        )
        self.capture_menu.pack(side="left", padx=(0, 8))

        self.display_var = tk.StringVar(value="双语")
        self.display_menu = ctk.CTkOptionMenu(
            bar,
            values=list(_DISPLAY_MODES),
            variable=self.display_var,
            width=78,
            height=34,
            corner_radius=17,
            fg_color="#1C1C22",
            button_color="#1C1C22",
            button_hover_color="#2C2C34",
            dropdown_fg_color="#1A1A20",
            dropdown_hover_color="#2C2C34",
            text_color="#E8E8EE",
            font=(config.font_source, 13),
            dropdown_font=(config.font_source, 13),
            command=self._change_display,
        )
        self.display_menu.pack(side="left")

        self.bind("<Configure>", self._on_resize)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(80, self._poll_ui)
        self._refresh_subtitle_view()
        self.after(120, self._apply_layout)

    def _place_bottom(self) -> None:
        """默认放在屏幕底部居中，距任务栏留出间距。"""
        self.update_idletasks()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        # 窗口可能尚未映射（winfo_width()==1），用配置尺寸；geometry 的宽高会被 CTk 缩放，坐标不会
        ws = ctk.ScalingTracker.get_window_scaling(self)
        w, h = self._win_w, self._base_height
        margin = int(getattr(self.config, "bottom_margin", 72))
        x = max(0, (sw - round(w * ws)) // 2)
        y = max(0, sh - round(h * ws) - margin)
        self.geometry(f"{w}x{h}+{x}+{y}")

    # —— helpers ——
    def _icon_btn(
        self,
        parent: ctk.CTkFrame,
        text: str,
        command: Callable[[], None],
        tip: str = "",
        active: bool = False,
    ) -> ctk.CTkButton:
        btn = ctk.CTkButton(
            parent,
            text=text,
            width=28,
            height=28,
            corner_radius=14,
            fg_color="#1A1A20" if active else "transparent",
            hover_color="#2A2A32",
            text_color="#5B8CFF" if active else "#B8B8C0",
            font=(self.config.font_source, 11),
            command=command,
        )
        btn.pack(side="left", padx=2)
        if tip:
            self._bind_tooltip(btn, tip)
        return btn

    def _bind_tooltip(self, widget: ctk.CTkBaseClass, text: str) -> None:
        tip: dict[str, tk.Toplevel | None] = {"win": None}

        def show(_event: object = None) -> None:
            if tip["win"] is not None:
                return
            win = tk.Toplevel(self)
            win.wm_overrideredirect(True)
            win.attributes("-topmost", True)
            x = widget.winfo_rootx()
            y = widget.winfo_rooty() + widget.winfo_height() + 6
            win.geometry(f"+{x}+{y}")
            tk.Label(
                win,
                text=text,
                bg="#222228",
                fg="#DDDDE4",
                font=(self.config.font_source, 10),
                padx=8,
                pady=3,
            ).pack()
            tip["win"] = win

        def hide(_event: object = None) -> None:
            if tip["win"] is not None:
                tip["win"].destroy()
                tip["win"] = None

        widget.bind("<Enter>", show)
        widget.bind("<Leave>", hide)

    def _noop(self) -> None:
        self.set_status("区域捕获功能预留")

    def _on_resize(self, _event: tk.Event | None = None) -> None:  # type: ignore[type-arg]
        wrap = max(400, self.winfo_width() - 80)
        self.source_label.configure(wraplength=wrap)
        self.target_label.configure(wraplength=wrap)

    def _start_drag(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        if self._locked:
            return
        self._drag_x = event.x_root
        self._drag_y = event.y_root

    def _on_drag(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        if self._locked:
            return
        dx = event.x_root - self._drag_x
        dy = event.y_root - self._drag_y
        self._drag_x = event.x_root
        self._drag_y = event.y_root
        self.geometry(f"+{self.winfo_x() + dx}+{self.winfo_y() + dy}")

    def _toggle_lock(self) -> None:
        self._locked = not self._locked
        self.btn_lock.configure(
            fg_color="#1A1A20" if self._locked else "transparent",
            text_color="#5B8CFF" if self._locked else "#B8B8C0",
        )
        self.set_status("已锁定位置" if self._locked else "已解锁，可拖动")

    def _toggle_pin(self) -> None:
        self._pinned = not self._pinned
        self.attributes("-topmost", self._pinned)
        self.btn_pin.configure(
            fg_color="#1A1A20" if self._pinned else "transparent",
            text_color="#5B8CFF" if self._pinned else "#B8B8C0",
        )
        self.set_status("已置顶" if self._pinned else "取消置顶")

    def _minimize(self) -> None:
        self.overrideredirect(False)
        self.update_idletasks()
        self.iconify()
        self.bind("<Map>", self._on_restore)

    def _on_restore(self, _event: object = None) -> None:
        if self.state() == "normal":
            self.overrideredirect(True)
            self.attributes("-topmost", self._pinned)
            self.unbind("<Map>")

    def _show_more(self) -> None:
        menu = tk.Menu(self, tearoff=0, bg="#1A1A20", fg="#E8E8EE", bd=0)
        menu.add_command(
            label=f"设备: {self._device_label or '未知'}",
            state="disabled",
        )
        menu.add_separator()
        menu.add_command(label="API / 模型设置", command=self._open_settings)
        menu.add_separator()
        menu.add_command(label="透明度 100%", command=lambda: self.attributes("-alpha", 1.0))
        menu.add_command(label="透明度 90%", command=lambda: self.attributes("-alpha", 0.9))
        menu.add_command(label="透明度 75%", command=lambda: self.attributes("-alpha", 0.75))
        try:
            menu.tk_popup(self.winfo_pointerx(), self.winfo_pointery())
        finally:
            menu.grab_release()

    def _open_settings(self) -> None:
        if self.on_open_settings:
            self.on_open_settings()

    def set_model_list(self, models: list[str], current: str | None = None) -> None:
        ids = [m for m in models if m]
        if not ids:
            return
        cur = current or self.config.deepseek_model or ids[0]
        if cur not in ids:
            ids = [cur, *ids]
        self._model_ids = [LOCAL_MODEL_LABEL, *ids]
        self.model_menu.configure(values=self._model_ids)
        self.config.deepseek_model = cur
        # 本地模型模式下下拉框继续显示本地条目
        local = self.config.translator_backend == "local"
        self.model_var.set(LOCAL_MODEL_LABEL if local else cur)

    def _change_model(self, model_id: str) -> None:
        if model_id != LOCAL_MODEL_LABEL:
            self.config.deepseek_model = model_id
        if self.on_model_change:
            self.on_model_change(model_id)
        self.set_status(f"模型: {model_id}")

    def _change_dir(self, label: str) -> None:
        mode = _MODE_VALUES.get(label, "ja2zh")
        self.config.translate_mode = mode
        if self.on_mode_change:
            self.on_mode_change(mode)
        self.set_status(f"方向: {label}")
        self._apply_simul_fonts()
        self._apply_layout()

    def _change_capture(self, label: str) -> None:
        mode = _CAPTURE_VALUES.get(label, "simul")
        self.config.capture_mode = mode
        if self.on_capture_mode_change:
            self.on_capture_mode_change(mode)
        self.set_status(f"采集: {label}")
        self._apply_layout()

    def _change_display(self, label: str) -> None:
        self._display_mode = label
        self._refresh_subtitle_view()
        self._apply_layout()
        self.set_status(f"显示: {label}")

    # —— 同传滚动字幕 ——
    def _is_simul(self) -> bool:
        return self.config.capture_mode == "simul"

    def _make_ticker(
        self,
        parent: ctk.CTkFrame,
        size: int,
        lines: int,
        color: str,
        tail_color: str,
    ) -> tk.Text:
        # 原生 tk.Text：高度按「行」计，滚到底时整行对齐，不会露出半行
        box = tk.Text(
            parent,
            height=lines,
            wrap="char",
            bg="#0B0B0D",
            fg=color,
            bd=0,
            highlightthickness=0,
            padx=0,
            pady=0,
            cursor="arrow",
            insertwidth=0,
            takefocus=0,
        )
        box._ticker_size = size  # type: ignore[attr-defined]
        box.tag_configure("tail", foreground=tail_color)
        box.configure(state="disabled")

        def press(event: tk.Event) -> str:  # type: ignore[type-arg]
            self._start_drag(event)
            return "break"

        def motion(event: tk.Event) -> str:  # type: ignore[type-arg]
            self._on_drag(event)
            return "break"

        box.bind("<ButtonPress-1>", press)
        box.bind("<B1-Motion>", motion)
        box.bind("<Double-Button-1>", lambda _e: "break")
        return box

    def _apply_simul_fonts(self) -> None:
        src_lang, tgt_lang = _MODE_LANGS.get(self.config.translate_mode, ("zh", "zh"))
        scale = ctk.ScalingTracker.get_widget_scaling(self)
        for box, lang in ((self.simul_src, src_lang), (self.simul_tgt, tgt_lang)):
            size = box._ticker_size  # type: ignore[attr-defined]
            box.configure(font=(_LANG_FONT[lang], -round(size * scale)))

    def _apply_layout(self) -> None:
        """同传模式改用滚动文本框，并按需调整窗口高度（底边不动）。"""
        simul = self._is_simul()
        for wdg in (self.source_label, self.target_label, self.simul_frame, self.hint_label):
            wdg.pack_forget()
        has_text = self._has_text()
        if simul:
            for box in (self.simul_src, self.simul_tgt):
                box.pack_forget()
            if self._display_mode != "译文":
                self.simul_src.pack(fill="x", anchor="w")
            if self._display_mode != "原文":
                self.simul_tgt.pack(fill="x", anchor="w", pady=(4, 0))
            if has_text:
                self.simul_frame.pack(fill="x", anchor="w")
        else:
            self.source_label.pack(fill="x", anchor="w")
            self.target_label.pack(fill="x", anchor="w", pady=(6, 0))
        if not has_text:
            self.hint_label.configure(text=self._status)
            self.hint_label.pack(fill="x", anchor="w", pady=(8, 0))
        self._fit_height()

    def _fit_height(self) -> None:
        self.update_idletasks()
        ws = ctk.ScalingTracker.get_window_scaling(self)
        cur_h_px = self.winfo_height()
        if self._is_simul():
            # 文本区高度 = 同传框所需高度；上下留白和控制条不变
            need = self.simul_frame.winfo_reqheight()
            new_h_px = cur_h_px - self.text_box.winfo_height() + need
            new_h = max(round(new_h_px / ws), 140)
        else:
            new_h = self._base_height
        new_h_px = round(new_h * ws)
        if abs(new_h_px - cur_h_px) <= 1:
            return
        bottom = self.winfo_y() + cur_h_px
        w = max(round(self.winfo_width() / ws), self._win_w)
        self.geometry(f"{w}x{new_h}+{self.winfo_x()}+{max(0, bottom - new_h_px)}")

    def _has_text(self) -> bool:
        if self._is_simul():
            return any(self._simul_view)
        return bool(self._source_text or self._target_text)

    @staticmethod
    def _render_ticker(box: tk.Text, stable: str, tail: str) -> None:
        box.configure(state="normal")
        box.delete("1.0", "end")
        if stable:
            box.insert("end", stable)
        if tail:
            box.insert("end", tail, "tail")
        box.yview_moveto(1.0)  # 始终显示最后几行
        box.configure(state="disabled")

    def push_simul(self, view: tuple[str, str, str, str]) -> None:
        """线程安全：同传引擎推送最新视图，UI 轮询时渲染。"""
        self._simul_view = view
        self._simul_dirty = True

    def post_status(self, text: str) -> None:
        """线程安全版 set_status。"""
        self._status_q.put(text)

    def _refresh_simul_view(self) -> None:
        had_frame = bool(self.simul_frame.winfo_manager())
        src, tail, tgt, live = self._simul_view
        self._render_ticker(self.simul_src, src, tail)
        self._render_ticker(self.simul_tgt, tgt, live)
        if any(self._simul_view) != had_frame:
            self._apply_layout()

    def set_asr_ready(self, ready: bool, message: str | None = None) -> None:
        self._asr_ready = ready
        if ready:
            self.btn_mic.configure(state="normal")
            self.set_status(message or "识别模型已就绪，点击麦克风开始")
        else:
            if self._running:
                self._running = False
                self.btn_mic.configure(fg_color="#1C1C22", text_color="#E8E8EE", text="●")
                if self.on_stop:
                    self.on_stop()
            self.btn_mic.configure(state="disabled")
            self.set_status(message or "识别模型加载中…")

    def _toggle(self) -> None:
        if not self._asr_ready and not self._running:
            self.set_status("识别模型还在加载，请稍候…")
            return
        if not self._running:
            self._running = True
            self.btn_mic.configure(fg_color="#2A4A7A", text_color="#7EB6FF", text="■")
            self.set_status("正在监听系统声音…")
            if self.on_start:
                self.on_start()
        else:
            self._running = False
            self.btn_mic.configure(fg_color="#1C1C22", text_color="#E8E8EE", text="●")
            self.set_status("已停止")
            if self.on_stop:
                self.on_stop()

    def _on_close(self) -> None:
        if self._running and self.on_stop:
            self.on_stop()
        self.destroy()

    def set_status(self, text: str) -> None:
        self._status = text
        # 左上角常驻，避免出字幕后倒计时消失
        self.status_badge.configure(text=text)
        if not self._has_text():
            self.hint_label.configure(text=text)
            self.hint_label.pack(fill="x", anchor="w", pady=(8, 0))

    def set_device(self, text: str) -> None:
        self._device_label = text

    def push_subtitle(
        self,
        source: str,
        target: str,
        meta: str = "",
        interim: bool = False,
    ) -> None:
        # meta 不覆盖录音倒计时等运行状态（由 set_status 单独维护）
        self._ui_q.put((source, target, meta, interim))

    def _refresh_subtitle_view(self) -> None:
        if self._is_simul():
            self.status_badge.configure(text=self._status)
            return
        mode = self._display_mode
        src = self._source_text
        tgt = self._target_text

        src_color = "#9A9AA4" if self._interim else "#D0D0D6"
        tgt_color = "#A8B8C8" if self._interim else "#FFFFFF"
        self.source_label.configure(text_color=src_color)
        self.target_label.configure(text_color=tgt_color)
        # 状态徽章始终同步
        self.status_badge.configure(text=self._status)

        if not src and not tgt:
            self.source_label.configure(text="")
            self.target_label.configure(text="")
            self.hint_label.configure(text=self._status)
            self.hint_label.pack(fill="x", anchor="w", pady=(8, 0))
            return

        self.hint_label.pack_forget()
        if mode == "原文":
            self.source_label.configure(text=src)
            self.target_label.configure(text="")
        elif mode == "译文":
            self.source_label.configure(text="")
            self.target_label.configure(text=tgt or src)
        else:
            self.source_label.configure(text=src)
            self.target_label.configure(text=tgt)

    def _poll_ui(self) -> None:
        try:
            while True:
                source, target, meta, interim = self._ui_q.get_nowait()
                self._source_text = source or ""
                self._target_text = target or ""
                self._interim = interim
                # 翻译 meta 不再覆盖「录音中 x/5s」状态
                del meta
                self._refresh_subtitle_view()
        except queue.Empty:
            pass
        try:
            while True:
                self.set_status(self._status_q.get_nowait())
        except queue.Empty:
            pass
        if self._simul_dirty and self._is_simul():
            self._simul_dirty = False
            self._refresh_simul_view()
        self.after(50, self._poll_ui)

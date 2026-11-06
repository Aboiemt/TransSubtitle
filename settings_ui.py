"""API / 模型设置窗口。"""

from __future__ import annotations

import threading
from collections.abc import Callable

import customtkinter as ctk

from api_models import fetch_remote_models, normalize_base_url
from config import AppConfig
from settings_store import save_from_config


class SettingsWindow(ctk.CTkToplevel):
    def __init__(
        self,
        master: ctk.CTk,
        config: AppConfig,
        on_saved: Callable[[AppConfig, list[str]], None] | None = None,
    ) -> None:
        super().__init__(master)
        self.config = config
        self.on_saved = on_saved
        self._models: list[str] = list(config.cached_models or [])

        self.title("API / 模型设置")
        self.geometry("520x460")
        self.resizable(False, False)
        self.configure(fg_color="#121218")
        self.attributes("-topmost", True)
        self.grab_set()

        pad = {"padx": 18, "pady": (8, 0)}

        ctk.CTkLabel(
            self,
            text="跨电脑使用：设置会保存到程序目录 settings.json",
            font=(config.font_source, 12),
            text_color="#8A8A96",
        ).pack(anchor="w", **pad)

        ctk.CTkLabel(self, text="API 地址", anchor="w").pack(anchor="w", **pad)
        self.entry_url = ctk.CTkEntry(
            self,
            height=34,
            placeholder_text="http://127.0.0.1:7863/v1",
        )
        self.entry_url.pack(fill="x", padx=18, pady=(4, 0))
        self.entry_url.insert(0, config.deepseek_base_url or "")

        ctk.CTkLabel(self, text="API Key", anchor="w").pack(anchor="w", **pad)
        self.entry_key = ctk.CTkEntry(self, height=34, show="*")
        self.entry_key.pack(fill="x", padx=18, pady=(4, 0))
        self.entry_key.insert(0, config.deepseek_api_key or "")

        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=18, pady=(12, 0))

        ctk.CTkLabel(row, text="当前模型").pack(side="left")
        self.btn_fetch = ctk.CTkButton(
            row,
            text="从上游拉取模型",
            width=120,
            height=30,
            fg_color="#2A4A7A",
            hover_color="#1B3558",
            command=self._fetch_models,
        )
        self.btn_fetch.pack(side="right")

        values = self._models or [config.deepseek_model or "（请先拉取）"]
        self.model_var = ctk.StringVar(value=config.deepseek_model or values[0])
        self.model_menu = ctk.CTkOptionMenu(
            self,
            values=values,
            variable=self.model_var,
            height=34,
            fg_color="#1C1C22",
            button_color="#1C1C22",
            dropdown_fg_color="#1A1A20",
        )
        self.model_menu.pack(fill="x", padx=18, pady=(8, 0))

        ctk.CTkLabel(self, text="思考强度", anchor="w").pack(anchor="w", **pad)
        self.effort_var = ctk.StringVar(value=config.reasoning_effort or "off")
        ctk.CTkOptionMenu(
            self,
            values=["off", "low", "medium", "high", "max"],
            variable=self.effort_var,
            height=34,
            fg_color="#1C1C22",
            button_color="#1C1C22",
            dropdown_fg_color="#1A1A20",
        ).pack(fill="x", padx=18, pady=(4, 0))

        self.msg = ctk.CTkLabel(
            self,
            text="",
            text_color="#7DD3C7",
            font=(config.font_source, 12),
            wraplength=470,
            justify="left",
        )
        self.msg.pack(anchor="w", padx=18, pady=(14, 0))

        btns = ctk.CTkFrame(self, fg_color="transparent")
        btns.pack(fill="x", padx=18, pady=18)
        ctk.CTkButton(
            btns,
            text="取消",
            width=90,
            fg_color="#2A2A32",
            hover_color="#3A3A44",
            command=self.destroy,
        ).pack(side="right", padx=(8, 0))
        ctk.CTkButton(
            btns,
            text="保存",
            width=90,
            fg_color="#2A6F97",
            hover_color="#1B4F6E",
            command=self._save,
        ).pack(side="right")

    def _set_msg(self, text: str, ok: bool = True) -> None:
        self.msg.configure(text=text, text_color="#7DD3C7" if ok else "#E08080")

    def _fetch_models(self) -> None:
        self.btn_fetch.configure(state="disabled", text="拉取中…")
        self._set_msg("正在请求上游 /v1/models …")

        url = self.entry_url.get().strip()
        key = self.entry_key.get().strip()

        def work() -> None:
            try:
                models = fetch_remote_models(key, url)
                self.after(0, lambda: self._on_fetched(models, None))
            except Exception as exc:  # noqa: BLE001
                self.after(0, lambda: self._on_fetched([], str(exc)))

        threading.Thread(target=work, daemon=True).start()

    def _on_fetched(self, models: list[str], err: str | None) -> None:
        self.btn_fetch.configure(state="normal", text="从上游拉取模型")
        if err:
            self._set_msg(f"拉取失败: {err}", ok=False)
            return
        if not models:
            self._set_msg("上游未返回任何模型", ok=False)
            return
        self._models = models
        current = self.model_var.get()
        self.model_menu.configure(values=models)
        if current in models:
            self.model_var.set(current)
        else:
            self.model_var.set(models[0])
        self._set_msg(f"已获取 {len(models)} 个模型，可选择后点保存")

    def _save(self) -> None:
        url = normalize_base_url(self.entry_url.get().strip())
        key = self.entry_key.get().strip()
        model = self.model_var.get().strip()
        if not url or not key:
            self._set_msg("API 地址和 Key 不能为空", ok=False)
            return
        if not model or model.startswith("（"):
            self._set_msg("请先拉取并选择模型", ok=False)
            return

        self.config.deepseek_base_url = url
        self.config.deepseek_api_key = key
        self.config.deepseek_model = model
        self.config.reasoning_effort = self.effort_var.get()
        self.config.cached_models = list(self._models or [model])

        save_from_config(
            self.config,
            extra={"cached_models": self.config.cached_models},
        )
        self._set_msg("已保存到 settings.json")
        if self.on_saved:
            self.on_saved(self.config, self.config.cached_models)
        self.after(350, self.destroy)

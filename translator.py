"""OpenAI 兼容中日文流式翻译（WorkBuddy / Buzz 等）。"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from openai import OpenAI, RateLimitError

from api_models import normalize_base_url
from config import AppConfig
from local_llm import SAKURA_SYSTEM, SAKURA_USER


@dataclass
class TranslationResult:
    source_text: str
    source_lang: str  # zh / ja / unknown
    target_text: str
    target_lang: str


_ZH_RE = re.compile(r"[\u4e00-\u9fff]")
_JA_RE = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff]")


def detect_cjk_lang(text: str, whisper_lang: str | None = None) -> str:
    if whisper_lang in ("zh", "ja"):
        return whisper_lang
    zh = len(_ZH_RE.findall(text))
    ja = len(_JA_RE.findall(text))
    if ja > zh and ja > 0:
        return "ja"
    if zh > 0:
        return "zh"
    return "unknown"


class DeepSeekTranslator:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        if not config.deepseek_api_key or config.deepseek_api_key.startswith("sk-your"):
            self._client = None
        else:
            self._client = OpenAI(
                api_key=config.deepseek_api_key,
                base_url=normalize_base_url(config.deepseek_base_url),
            )
        # 本地 llama-server（Sakura，仅日→中）；不重试，失败立即回报
        self._local = None
        if config.translator_backend == "local":
            self._local = OpenAI(
                api_key="local",
                base_url=f"http://127.0.0.1:{config.local_port}/v1",
                timeout=30.0,
                max_retries=0,
            )
        self._rate_lock = threading.Lock()
        self._last_call_at = 0.0
        self._cache: dict[str, str] = {}
        self._cache_lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self._client is not None or self._local is not None

    @property
    def fast(self) -> bool:
        """本地模型延迟约 0.1~0.2s，同传可以每次识别更新都重译整句。"""
        return self._local is not None

    def _use_local(self, src: str, tgt: str) -> bool:
        return self._local is not None and tgt == "zh" and src in ("ja", "unknown")

    @staticmethod
    def _local_kwargs() -> dict:
        # Sakura 官方推荐参数
        return {
            "model": "sakura",
            "temperature": 0.1,
            "top_p": 0.3,
            "frequency_penalty": 0.1,
            "max_tokens": 256,
        }

    @staticmethod
    def _sakura_messages(text: str, context: list[tuple[str, str]]) -> list[dict]:
        msgs = [{"role": "system", "content": SAKURA_SYSTEM}]
        for s, t in context:
            msgs.append({"role": "user", "content": SAKURA_USER + s})
            msgs.append({"role": "assistant", "content": t})
        msgs.append({"role": "user", "content": SAKURA_USER + text})
        return msgs

    def _resolve_langs(
        self, text: str, whisper_lang: str | None
    ) -> tuple[str, str, str, str]:
        mode = self.config.translate_mode
        src = detect_cjk_lang(text, whisper_lang)

        if mode == "zh2ja":
            src, tgt = "zh", "ja"
        elif mode == "ja2zh":
            src, tgt = "ja", "zh"
        else:
            if src == "ja":
                tgt = "zh"
            elif src == "zh":
                tgt = "ja"
            else:
                src, tgt = "unknown", "zh"

        src_name = {"zh": "中文", "ja": "日文", "unknown": "原文"}.get(src, "原文")
        tgt_name = {"zh": "中文", "ja": "日文"}.get(tgt, "目标语言")
        return src, tgt, src_name, tgt_name

    def _user_prompt(self, text: str, src_name: str, tgt_name: str) -> str:
        # WorkBuddy 等网关常忽略 system，指令放进 user 更稳
        return (
            f"任务：{src_name}译{tgt_name}字幕。\n"
            "只输出译文；禁止解释、禁止引号、禁止前后缀、禁止复述任务。\n"
            f"原文：\n{text}"
        )

    def _wait_rate_limit(self, interval: float | None = None) -> None:
        if interval is None:
            interval = float(getattr(self.config, "translate_min_interval", 2.5) or 0)
        if interval <= 0:
            return
        with self._rate_lock:
            now = time.monotonic()
            wait = self._last_call_at + interval - now
            if wait > 0:
                time.sleep(wait)
            self._last_call_at = time.monotonic()

    def _extract_text(self, message: object) -> str:
        content = (getattr(message, "content", None) or "").strip()
        if content:
            return content
        # 部分思考模型只把内容放在 reasoning_content
        reasoning = (getattr(message, "reasoning_content", None) or "").strip()
        if not reasoning:
            return ""
        for key in ("翻译：", "译文：", "Translation:", "译："):
            if key in reasoning:
                tail = reasoning.split(key)[-1].strip()
                line = tail.splitlines()[0].strip(" 「」\"'")
                if line:
                    return line
        # 取最后一行像句子的中文/日文
        for line in reversed(reasoning.splitlines()):
            s = line.strip(" `-•\t")
            if len(s) >= 2 and (_ZH_RE.search(s) or _JA_RE.search(s)):
                if "任务" in s or "原文" in s or "禁止" in s:
                    continue
                return s
        return ""

    def _create_kwargs(self) -> dict:
        effort = (getattr(self.config, "reasoning_effort", "off") or "off").lower()
        # WorkBuddy: reasoning_effort=off 关闭思考；再带 thinking.disabled 兜底
        extra: dict = {
            "reasoning_effort": effort,
            "thinking": {"type": "disabled"} if effort in ("off", "none", "disabled") else {"type": "enabled"},
        }
        if effort in ("off", "none", "disabled"):
            extra["enable_thinking"] = False
            extra["chat_template_kwargs"] = {"enable_thinking": False}
        return {
            "model": self.config.deepseek_model,
            "temperature": 0.1,
            "max_tokens": 256,
            "extra_body": extra,
        }

    def _request(
        self,
        messages: list[dict],
        on_delta: Callable[[str], None] | None = None,
        interval: float | None = None,
        local: bool = False,
    ) -> tuple[str, str | None]:
        """发请求（有 on_delta 时流式）。返回 (译文, 错误提示)。"""
        client = self._local if local else self._client
        base_kwargs = self._local_kwargs() if local else self._create_kwargs()
        try:
            if not local:
                self._wait_rate_limit(interval)
            if on_delta is not None:
                stream = client.chat.completions.create(
                    messages=messages,
                    stream=True,
                    **base_kwargs,
                )
                acc = ""
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    piece = delta.content or ""
                    if not piece:
                        continue
                    acc += piece
                    on_delta(acc.strip())
                translated = acc.strip()
                if not translated and not local:
                    self._wait_rate_limit(interval)
                    resp = client.chat.completions.create(
                        messages=messages,
                        **base_kwargs,
                    )
                    translated = self._extract_text(resp.choices[0].message)
                    if translated:
                        on_delta(translated)
            else:
                resp = client.chat.completions.create(
                    messages=messages,
                    **base_kwargs,
                )
                translated = self._extract_text(resp.choices[0].message)
        except RateLimitError:
            with self._rate_lock:
                self._last_call_at = time.monotonic() + 15
            return "", "[请求过于频繁，请稍候]"
        except Exception as exc:  # noqa: BLE001
            if local:
                return "", f"[本地模型失败: {exc}]"
            return "", f"[翻译失败: {exc}]"
        return translated, None

    def _simul_prompt(
        self,
        text: str,
        context: list[tuple[str, str]],
        src_name: str,
        tgt_name: str,
    ) -> str:
        lines = [
            f"你是专业的{src_name}→{tgt_name}同声传译员，正在为直播/视频实时出字幕。",
            "原文是语音识别的流式片段：可能只是半句话，也可能有同音错字。",
        ]
        if context:
            lines.append("【前文】（已译，仅供理解上下文，不要重复输出）")
            for s, t in context:
                lines.append(f"原文：{s}")
                lines.append(f"译文：{t}")
        lines += [
            "【当前片段】",
            text,
            "要求：只输出当前片段的译文；与前文译文自然衔接；"
            "片段没说完就只译已说出的部分，不要补全或猜测后文；"
            "口语化、简洁；不要解释、不要引号、不要任何前缀。",
        ]
        return "\n".join(lines)

    @staticmethod
    def _clean(text: str) -> str:
        t = text.strip()
        for prefix in ("译文：", "译文:", "翻译：", "翻译:", "訳文：", "訳："):
            if t.startswith(prefix):
                t = t[len(prefix) :].strip()
        if len(t) >= 2 and t[0] in "「『\"“" and t[-1] in "」』\"”":
            t = t[1:-1].strip()
        return t

    def translate_simul(
        self,
        text: str,
        context: list[tuple[str, str]],
        on_delta: Callable[[str], None] | None = None,
    ) -> TranslationResult:
        """同传：带前文上下文翻译一个意群（不走缓存，不受 translate_min_interval 限制）。"""
        text = text.strip()
        src, tgt, src_name, tgt_name = self._resolve_langs(text, None)
        if not text:
            return TranslationResult("", src, "", tgt)
        local = self._use_local(src, tgt)
        if not local and not self._client:
            return TranslationResult(text, src, "[未配置 API Key，跳过翻译]", tgt)

        if local:
            messages = self._sakura_messages(text, context[-2:])
        else:
            messages = [
                {
                    "role": "user",
                    "content": self._simul_prompt(text, context, src_name, tgt_name),
                }
            ]
        delta = (lambda t: on_delta(self._clean(t))) if on_delta else None
        translated, err = self._request(
            messages,
            delta,
            interval=float(getattr(self.config, "simul_min_interval", 0.0) or 0),
            local=local,
        )
        return TranslationResult(text, src, err or self._clean(translated), tgt)

    def translate(
        self,
        text: str,
        whisper_lang: str | None = None,
        on_delta: Callable[[str], None] | None = None,
    ) -> TranslationResult:
        text = text.strip()
        if not text:
            return TranslationResult("", "unknown", "", "unknown")

        src, tgt, src_name, tgt_name = self._resolve_langs(text, whisper_lang)
        local = self._use_local(src, tgt)
        model_key = "local" if local else self.config.deepseek_model
        cache_key = f"{model_key}|{src}|{tgt}|{text}"

        with self._cache_lock:
            cached = self._cache.get(cache_key)
        if cached is not None:
            if on_delta:
                on_delta(cached)
            return TranslationResult(text, src, cached, tgt)

        if not local and not self._client:
            msg = "[未配置 API Key，跳过翻译]"
            if on_delta:
                on_delta(msg)
            return TranslationResult(text, src, msg, tgt)

        if local:
            messages = self._sakura_messages(text, [])
        else:
            messages = [
                {"role": "user", "content": self._user_prompt(text, src_name, tgt_name)}
            ]
        translated, err = self._request(messages, on_delta, local=local)
        if err:
            if on_delta:
                on_delta(err)
            return TranslationResult(text, src, err, tgt)

        if translated:
            with self._cache_lock:
                self._cache[cache_key] = translated
                if len(self._cache) > 200:
                    for k in list(self._cache.keys())[:50]:
                        self._cache.pop(k, None)

        return TranslationResult(text, src, translated, tgt)

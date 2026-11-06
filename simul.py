"""同声传译：流式识别（LocalAgreement）+ 按意群增量翻译。

流程：
  系统声音 ──► OnlineASR（滚动缓冲，每 ~0.5s 识别一次）
               连续两轮识别结果的公共前缀 = 「已确认」，其余为「未确认尾巴」
           ──► 切块器（遇句末 / 逗号 / 字数上限 / 停顿 即切出一个意群）
           ──► 翻译线程（串行、带前文上下文、流式输出）
           ──► 滚动字幕
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from asr import WhisperASR
from config import AppConfig
from translator import DeepSeekTranslator

# 比较时忽略的字符（标点 / 空白）；注意不要包含长音符「ー」
_IGNORE_RE = re.compile(r"[\s、。，,．.！!？?…・「」『』（）()\"'“”‘’：:；;]")
_SENT_END = "。！？!?…"
_CLAUSE_END = "、，,；;：:"

# 没有上文时的默认 prompt：引导 Whisper 输出标点，便于按句/意群切分
_DEFAULT_PROMPT = {
    "ja": "はい、そうですね。",
    "zh": "以下是简体中文普通话的句子。",
}

# (源原文已确认, 源原文未确认, 译文已完成, 译文生成中)
SimulView = tuple[str, str, str, str]


def _norm(text: str) -> str:
    return _IGNORE_RE.sub("", text).lower()


def _join(words: list["Word"]) -> str:
    return "".join(w.text for w in words)


@dataclass
class Word:
    start: float  # 绝对时间（秒）
    end: float
    text: str


class OnlineASR:
    """whisper_streaming 风格的在线识别（LocalAgreement-2，按字符比较，适配中日文）。"""

    def __init__(self, asr: WhisperASR, config: AppConfig) -> None:
        self.asr = asr
        self.config = config
        self.sr = config.sample_rate
        self.language: str | None = None
        self.context = ""  # 已移出音频缓冲的已确认文本（作 prompt）
        self.reset(0.0)

    # —— 缓冲管理 ——
    def reset(self, offset: float) -> None:
        self.audio = np.zeros(0, dtype=np.float32)
        self.offset = offset  # 缓冲区起点的绝对时间
        self.prev: list[Word] = []  # 上一轮未确认假设
        self.committed_in_buf: list[Word] = []
        self.last_committed_time = offset

    @property
    def buffer_seconds(self) -> float:
        return len(self.audio) / float(self.sr)

    @property
    def end_time(self) -> float:
        return self.offset + self.buffer_seconds

    def insert_audio(self, chunk: np.ndarray) -> None:
        self.audio = np.concatenate([self.audio, chunk.astype(np.float32, copy=False)])

    def drop_silence(self, keep_sec: float = 0.5) -> None:
        """缓冲里没有语音时丢掉旧音频，只保留一小段预卷。"""
        keep = int(keep_sec * self.sr)
        if len(self.audio) > keep:
            cut = len(self.audio) - keep
            self.audio = self.audio[cut:]
            self.offset += cut / float(self.sr)
            self.last_committed_time = max(self.last_committed_time, self.offset)

    def _chop(self, t: float) -> None:
        n = int(round((t - self.offset) * self.sr))
        if n <= 0:
            return
        gone = [w for w in self.committed_in_buf if w.end <= t]
        self.committed_in_buf = [w for w in self.committed_in_buf if w.end > t]
        self._remember(_join(gone))
        self.audio = self.audio[n:]
        self.offset = t

    def _remember(self, text: str) -> None:
        self.context = (self.context + text)[-200:]

    def _prompt(self) -> str | None:
        ctx = self.context.strip()[-100:]
        default = _DEFAULT_PROMPT.get(self.language or "")
        if self.language == "zh":
            return f"{default}{ctx}"  # 始终带上，避免输出繁体
        return ctx or default

    # —— 识别 ——
    def process(self) -> tuple[str, str]:
        """跑一轮识别，返回 (新确认文本, 未确认尾巴)。"""
        if len(self.audio) < int(0.3 * self.sr):
            return "", _join(self.prev)

        raw, _lang = self.asr.transcribe_words(
            self.audio, language=self.language, initial_prompt=self._prompt()
        )
        new = [Word(s + self.offset, e + self.offset, t) for s, e, t in raw]
        # 已确认的词（中点落在已确认时间之前）不再参与
        new = [w for w in new if (w.start + w.end) / 2 > self.last_committed_time - 0.05]
        new = self._drop_overlap(new)

        commit = self._agree(new)
        text = _join(commit)
        text += self._maybe_trim()
        return text, _join(self.prev)

    def _drop_overlap(self, new: list[Word]) -> list[Word]:
        """去掉与已确认尾部重复的开头几个词（时间戳不准时会出现）。"""
        if not new or not self.committed_in_buf:
            return new
        tail = _norm(_join(self.committed_in_buf[-6:]))
        drop = 0
        for i in range(1, min(len(new), 4) + 1):
            if new[i - 1].start >= self.last_committed_time:
                break
            head = _norm(_join(new[:i]))
            if head and tail.endswith(head):
                drop = i
        return new[drop:]

    def _agree(self, new: list[Word]) -> list[Word]:
        """按字符取本轮与上一轮假设的最长公共前缀，整词确认。"""
        parts = [_norm(w.text) for w in new]
        a = _norm(_join(self.prev))
        b = "".join(parts)
        lcp = 0
        for x, y in zip(a, b):
            if x != y:
                break
            lcp += 1

        k = 0
        pos = 0
        last_real = 0
        for i, p in enumerate(parts):
            if pos + len(p) > lcp:
                break
            pos += len(p)
            k = i + 1
            if p:
                last_real = k
        # 至少要确认一个实词；纯标点只跟随已确认实词
        if last_real == 0:
            k = 0

        commit = new[:k]
        self.prev = new[k:]
        if commit:
            self.committed_in_buf.extend(commit)
            self.last_committed_time = max(self.last_committed_time, commit[-1].end)
        return commit

    def _buffer_limits(self) -> tuple[float, float]:
        cfg = self.config
        if getattr(self.asr, "on_gpu", False):
            return cfg.simul_buffer_trim_sec_gpu, cfg.simul_buffer_max_sec_gpu
        return cfg.simul_buffer_trim_sec, cfg.simul_buffer_max_sec

    def _maybe_trim(self) -> str:
        """缓冲过长时从已确认的句末处截掉前面的音频。返回被强制确认的文本。"""
        trim_sec, max_sec = self._buffer_limits()
        dur = self.buffer_seconds
        if dur < trim_sec:
            return ""
        cut: float | None = None
        for w in reversed(self.committed_in_buf):
            s = w.text.strip()
            if s and s[-1] in _SENT_END:
                cut = w.end
                break
        if cut is None and dur > max_sec and self.committed_in_buf:
            cut = self.committed_in_buf[-1].end
        if cut is not None:
            self._chop(cut)
            return ""
        if dur > max_sec * 1.4:
            # 一直无法达成一致（音乐 / 噪声）：强制确认后清空
            return self.finish()
        return ""

    def finish(self, speech_end: float | None = None) -> str:
        """一句话结束：把未确认尾巴确认（丢掉落在静音里的幻听词），清空缓冲。"""
        words = self.prev
        if speech_end is not None:
            words = [w for w in words if w.start < speech_end + 0.2]
        text = _join(words)
        self._remember(_join(self.committed_in_buf) + text)
        self.reset(self.end_time)
        return text


@dataclass
class _Piece:
    src: str
    tgt: str = ""
    done: bool = False


class SimulInterpreter:
    """同传引擎：一个识别线程 + 一个翻译线程。"""

    def __init__(
        self,
        config: AppConfig,
        asr: WhisperASR,
        get_translator: Callable[[], DeepSeekTranslator],
        on_view: Callable[[SimulView], None],
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.asr = asr
        self.get_translator = get_translator
        self.on_view = on_view
        self.on_status = on_status
        self.sr = config.sample_rate

        self._cond = threading.Condition()
        self._running = False
        self._session = 0
        self._asr_thread: threading.Thread | None = None
        self._tr_thread: threading.Thread | None = None
        self._reset_state()

    @property
    def running(self) -> bool:
        return self._running

    def _reset_state(self) -> None:
        # 音频输入
        self._incoming: list[np.ndarray] = []
        self._incoming_samples = 0
        self._silence_samples = 0
        self._heard = False
        # 文本
        self._tail = ""
        self._pending = ""  # 已确认、未送翻译
        self._last_commit_at = 0.0
        self._units: list[str] = []  # 待翻译的意群
        self._pieces: list[_Piece] = []
        self._view_start = 0
        self._last_status_at = 0.0
        # 重译模式（本地快模型）：当前未定稿整句的草稿译文
        self._fast = False
        self._draft_src = ""
        self._draft_tgt = ""

    # —— 生命周期 ——
    def start(self) -> None:
        with self._cond:
            if self._running:
                return
            self._session += 1
            self._reset_state()
            self._running = True
            session = self._session
        self.on_view(("", "", "", ""))
        self._asr_thread = threading.Thread(
            target=self._asr_loop, args=(session,), daemon=True
        )
        self._tr_thread = threading.Thread(
            target=self._translate_loop, args=(session,), daemon=True
        )
        self._asr_thread.start()
        self._tr_thread.start()

    def stop(self) -> None:
        """异步停止：识别线程收尾最后一句，翻译线程译完剩余内容后退出。"""
        with self._cond:
            self._running = False
            self._cond.notify_all()

    def _alive(self, session: int) -> bool:
        return session == self._session

    def feed(self, chunk: np.ndarray) -> None:
        rms = float(np.sqrt(np.mean(np.square(chunk))) + 1e-12)
        with self._cond:
            if not self._running:
                return
            self._incoming.append(chunk)
            self._incoming_samples += int(chunk.shape[0])
            if rms >= self.config.rms_threshold:
                self._silence_samples = 0
                self._heard = True
            else:
                self._silence_samples += int(chunk.shape[0])
            self._cond.notify_all()

    def _language(self) -> str | None:
        if self.config.whisper_language in ("zh", "ja"):
            return self.config.whisper_language
        return {"ja2zh": "ja", "zh2ja": "zh"}.get(self.config.translate_mode)

    def _status(self, text: str, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_status_at < 0.5:
            return
        self._last_status_at = now
        if self.on_status:
            self.on_status(text)

    # —— 识别线程 ——
    def _asr_loop(self, session: int) -> None:
        online = OnlineASR(self.asr, self.config)
        min_samples = int(self.sr * self.config.simul_min_chunk_ms / 1000)
        endpoint_samples = int(self.sr * self.config.simul_endpoint_ms / 1000)
        heard_since_reset = False

        while self._alive(session):
            with self._cond:
                self._cond.wait_for(
                    lambda: not self._running or self._incoming_samples >= min_samples,
                    timeout=0.5,
                )
                stopping = not self._running
                if not stopping and self._incoming_samples < min_samples:
                    continue
                chunk = (
                    np.concatenate(self._incoming)
                    if self._incoming
                    else np.zeros(0, dtype=np.float32)
                )
                self._incoming = []
                self._incoming_samples = 0
                silence = self._silence_samples
                heard_since_reset = heard_since_reset or self._heard
                self._heard = False

            online.language = self._language()
            online.insert_audio(chunk)

            try:
                if stopping:
                    if heard_since_reset or online.prev:
                        committed, _ = online.process()
                        self._commit(session, committed + online.finish(), end=True)
                    else:
                        self._commit(session, "", end=True)
                    self._set_tail(session, "")
                    break

                if not heard_since_reset and not online.prev:
                    online.drop_silence()
                    continue

                t0 = time.monotonic()
                committed, tail = online.process()
                cost = time.monotonic() - t0
                end = silence >= endpoint_samples
                if end:
                    committed += online.finish(online.end_time - silence / self.sr)
                    tail = ""
                    heard_since_reset = False
                self._commit(session, committed, end=end)
                self._set_tail(session, tail)
                if self._alive(session):
                    self._status(
                        f"同传中 · 识别 {cost:.1f}s · 缓冲 {online.buffer_seconds:.1f}s"
                    )
            except Exception as exc:  # noqa: BLE001
                self._status(f"同传识别错误: {exc}", force=True)
                online.reset(online.end_time)
                heard_since_reset = False
                time.sleep(0.3)

    def _set_tail(self, session: int, tail: str) -> None:
        with self._cond:
            if not self._alive(session):
                return
            self._tail = tail
        self._emit()

    def _commit(self, session: int, text: str, end: bool) -> None:
        with self._cond:
            if not self._alive(session):
                return
            if text:
                self._pending += text
                self._last_commit_at = time.monotonic()
            self._cut(force=end)
            self._cond.notify_all()

    # —— 切意群 ——
    def _cut(self, force: bool) -> None:
        """在 self._cond 锁内调用：把 pending 中可以翻译的部分切成意群。

        重译模式（本地快模型）下草稿会实时显示，所以尽量按整句切，译文更连贯。
        """
        p = self._pending
        if not _norm(p):
            return  # 只剩标点：留给下一个意群

        max_chars = self.config.simul_max_unit_chars
        min_clause = self.config.simul_min_unit_chars
        if self._fast:
            min_clause, max_chars = max_chars, max_chars * 2

        cut = -1
        sent = max(p.rfind(c) for c in _SENT_END)
        if sent >= 0 and len(_norm(p[: sent + 1])) >= 2:
            cut = sent + 1
        else:
            clause = max(p.rfind(c) for c in _CLAUSE_END)
            if clause >= 0 and len(_norm(p[: clause + 1])) >= min_clause:
                cut = clause + 1
            elif len(_norm(p)) >= max_chars or force:
                cut = len(p)

        if cut > 0:
            self._units.append(p[:cut])
            self._pending = p[cut:]

    # —— 翻译线程 ——
    def _translate_loop(self, session: int) -> None:
        idle = self.config.simul_idle_flush_ms / 1000.0
        while self._alive(session):
            translator = self.get_translator()
            fast = bool(getattr(translator, "fast", False)) and self.config.simul_retranslate
            draft = ""
            with self._cond:
                self._fast = fast
                # 还在说（有未确认尾巴）时多等一会，避免切出半个词
                wait = idle if not self._tail else idle * 2.5
                if (
                    not self._units
                    and self._pending
                    and time.monotonic() - self._last_commit_at >= wait
                    and (not self._tail or len(_norm(self._pending)) >= 4)
                ):
                    self._cut(force=True)

                if self._units:
                    unit = "".join(self._units)
                    self._units = []
                    piece = _Piece(src=unit)
                    if fast and self._draft_tgt and _norm(self._draft_src) == _norm(unit):
                        # 草稿正好就是这一整句：直接定稿，省一次请求
                        piece.tgt = self._draft_tgt
                        piece.done = True
                    elif fast:
                        piece.tgt = self._draft_tgt  # 定稿译文回来前先显示草稿
                    self._draft_src = self._draft_tgt = ""
                    self._pieces.append(piece)
                    context = [
                        (x.src, x.tgt) for x in self._pieces[-4:-1] if x.done and x.tgt
                    ]
                elif fast and _norm(self._pending + self._tail) and (
                    self._pending + self._tail != self._draft_src
                ):
                    # 重译：当前整句（已确认 + 未确认尾巴）变了就重新译一遍草稿
                    draft = self._pending + self._tail
                    self._draft_src = draft
                    context = [(x.src, x.tgt) for x in self._pieces[-3:] if x.done and x.tgt]
                else:
                    asr_alive = self._asr_thread is not None and self._asr_thread.is_alive()
                    if not self._running and not asr_alive and not _norm(self._pending):
                        break
                    self._cond.wait(timeout=0.1 if fast else 0.15)
                    continue
            self._emit()

            if draft:
                try:
                    res = translator.translate_simul(draft, context)
                except Exception:  # noqa: BLE001
                    continue
                with self._cond:
                    if self._draft_src == draft:  # 期间没被定稿 / 清掉
                        self._draft_tgt = res.target_text
                if self._alive(session):
                    self._emit()
                continue

            if piece.done:
                continue

            def on_delta(text: str, piece: _Piece = piece) -> None:
                piece.tgt = text
                if self._alive(session):
                    self._emit()

            try:
                # 快模型不流式：0.1~0.2s 就返回，避免草稿被半截译文替换造成闪烁
                res = translator.translate_simul(
                    unit, context, on_delta=None if fast else on_delta
                )
                piece.tgt = res.target_text
            except Exception as exc:  # noqa: BLE001
                piece.tgt = f"[翻译失败: {exc}]"
            piece.done = True
            if self._alive(session):
                self._emit()

        if self._alive(session) and not self._running:
            self._status("已停止", force=True)

    # —— 视图 ——
    def _emit(self) -> None:
        with self._cond:
            pieces = self._pieces[self._view_start :]
            src = "".join(p.src for p in pieces) + "".join(self._units) + self._pending
            tgt_parts: list[str] = []
            live = ""
            for p in pieces:
                if p.done:
                    tgt_parts.append(p.tgt)
                else:
                    live = p.tgt or "…"
                    break
            else:
                if self._pending or self._tail:
                    live = self._draft_tgt
            tgt = "".join(tgt_parts)
            # 文本过长时整体前移（跳跃式，避免逐字重排）
            if len(src) > 600 or len(tgt) > 600:
                self._view_start = max(self._view_start, len(self._pieces) - 2)
            view = (src, self._tail, tgt, live)
        self.on_view(view)

"""Dual-channel stream sanitation — FSM over token deltas (no regex-on-full-text).

Problem this solves (see architecture notes): open-weight reasoning models
(Qwen3, DeepSeek-R1, GLM, Gemma-4-thinking…) emit chain-of-thought wrapped
in tags like <think>/</think>. During SSE streaming those tags routinely
split across chunks, so string-level filters miss them and raw reasoning
leaks into the UI and — worse in a voice agent — into the TTS channel.

Design: a small Finite State Machine evaluated per delta.

  TEXT  — normal pass-through. On seeing '<', hold the buffer and try to
          match it against the known reasoning-tag table:
            • full tag match     → switch to THINK, swallow the tag
            • partial tag prefix → HOLD (need more deltas; ≤17 bytes)
            • no match           → emit '<' literally and continue
  THINK — route everything into the thinking channel until the matching
          close tag arrives (tail-held to avoid splitting the closer).
          Reasoning NEVER reaches the clean channel, so downstream JSON
          parsers and the TTS queue stay untouched.

At end-of-stream, flush() resolves whatever remains:
  • TEXT residue       → emitted verbatim (it was never a tag)
  • unclosed THINK     → discarded (synthetic closure; reasoning is not
                         answer content)

Zero-dependency, O(n) per byte, holdback bounded by the longest tag.
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional

DEFAULT_OPEN_TAGS = {
    "<think>": "</think>",
    "<thinking>": "</thinking>",
    "<thought>": "</thought>",
    "<channel_thought>": "</channel_thought>",
    "<reasoning>": "</reasoning>",
}


class StreamSanitizer:
    TEXT, THINK = "text", "think"

    def __init__(self,
                 tags: Optional[dict[str, str]] = None,
                 on_thinking: Optional[Callable[[str], None]] = None):
        self.open_tags = dict(tags or DEFAULT_OPEN_TAGS)
        self.longest = max(len(t) for t in self.open_tags)
        self.on_thinking = on_thinking
        self.state = self.TEXT
        self.pending = ""                 # holdback buffer
        self.close_tag = ""
        self.thinking_chars = 0           # stats

    # -- helpers --------------------------------------------------------------

    def _full_match(self, buf: str) -> Optional[str]:
        low = buf.lower()
        for tag in self.open_tags:
            if low.startswith(tag):
                return tag
        return None

    def _partial_match(self, buf: str) -> bool:
        """True if buf could still become one of the open tags."""
        if len(buf) >= self.longest:
            return False
        low = buf.lower()
        return any(tag.startswith(low) for tag in self.open_tags)

    # -- streaming API ----------------------------------------------------------

    def feed(self, piece: str) -> tuple[str, str]:
        """Consume one delta.

        Returns (clean_text, thinking_text) — emit clean to the user/TTS,
        thinking may be shown in a collapsed UI channel or discarded.
        """
        self.pending += piece
        out: list[str] = []
        thk: list[str] = []

        while self.pending:
            if self.state == self.THINK:
                close = self.close_tag
                idx = self.pending.lower().find(close)
                if idx != -1:
                    thk.append(self.pending[:idx])
                    self.pending = self.pending[idx + len(close):]
                    self.state = self.TEXT
                    continue
                hold = len(close) - 1
                if len(self.pending) > hold:
                    cut = len(self.pending) - hold
                    thk.append(self.pending[:cut])
                    self.pending = self.pending[cut:]
                    continue
                break                                    # need more data

            # TEXT state
            lt = self.pending.find("<")
            if lt == -1:
                out.append(self.pending)
                self.pending = ""
                break
            if lt > 0:
                out.append(self.pending[:lt])
                self.pending = self.pending[lt:]
                continue

            # pending starts with '<'
            full = self._full_match(self.pending)
            if full:
                self.state = self.THINK
                self.close_tag = self.open_tags[full]
                self.pending = self.pending[len(full):]
                continue
            if self._partial_match(self.pending):
                break                                    # hold for more
            # literal '<' (e.g. "x < y" in code) — pass through
            out.append(self.pending[0])
            self.pending = self.pending[1:]

        clean, think = "".join(out), "".join(thk)
        if think:
            self.thinking_chars += len(think)
            if self.on_thinking:
                try:
                    self.on_thinking(think)
                except Exception:                     # noqa: BLE001
                    pass
        return clean, think

    def flush(self) -> tuple[str, str]:
        """End-of-stream resolution.

        Returns any residual clean text. Residual THINK content is
        intentionally discarded (unclosed reasoning ≠ answer).
        """
        rem_out, rem_think = "", ""
        if self.state == self.TEXT:
            rem_out = self.pending
        else:
            rem_think = self.pending                  # synthetic closure
            if rem_think and self.on_thinking:
                try:
                    self.on_thinking(rem_think)
                except Exception:                     # noqa: BLE001
                    pass
        self.pending = ""
        self.state = self.TEXT
        return rem_out, rem_think

    @property
    def thinking_active(self) -> bool:
        return self.state == self.THINK


def sanitize_stream(deltas: Iterable[str],
                    on_thinking: Optional[Callable[[str], None]] = None,
                    tags: Optional[dict[str, str]] = None):
    """Generator wrapper: raw delta iterable → clean-text iterable.

    Thinking deltas are routed to on_thinking (if given) and never yielded.
    """
    san = StreamSanitizer(tags=tags, on_thinking=on_thinking)
    for piece in deltas:
        clean, _ = san.feed(piece)
        if clean:
            yield clean
    tail, _ = san.flush()
    if tail:
        yield tail


def sanitize_full(text: str, tags: Optional[dict[str, str]] = None) -> str:
    """One-shot convenience for complete strings (non-streaming paths)."""
    san = StreamSanitizer(tags=tags)
    clean, _ = san.feed(text)
    tail, _ = san.flush()
    return clean + tail

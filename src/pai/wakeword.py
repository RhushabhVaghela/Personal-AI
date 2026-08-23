"""Wake-word gating — Gemini Live's "proactive audio" equivalent.

Two-layer hybrid design:
  1. openWakeWord (offline, CPU) detects "hey assistant"-style wake phrase
     when the model is available.
   embedding fallback: a short transcript is checked against configurable
     phrases (works with ANY provider, even online ones) before a turn is
     accepted. This mirrors ChatGPT's "wait until I ask you to respond"
     behaviour.

States: ARMED (waiting for wake word) → ACTIVE (normal VAD turns) →
returns to ARMED after `active_timeout` seconds of silence.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Optional

log = logging.getLogger("pai.wakeword")

try:
    from openwakeword.model import Model as OwwModel
    HAS_OWW = True
except Exception:  # ImportError or model-download issues
    HAS_OWW = False


class WakeWordGate:
    """Gates hands-free turns behind a wake phrase."""

    def __init__(self,
                 phrases: list[str] | None = None,
                 active_timeout: float = 90.0,
                 use_openwakeword: bool = True,
                 oww_models: list[str] | None = None):
        self.phrases = [p.lower() for p in (phrases or [
            "hey assistant", "hey assistant.", "assistant",
            "computer"])]
        self.active_timeout = active_timeout
        self.active_until = 0.0
        self._oww = None
        self._oww_frame_size = 1280   # 80ms @16k
        if use_openwakeword and HAS_OWW:
            try:
                if oww_models:
                    self._oww = OwwModel(wakeword_models=list(oww_models))
                else:
                    self._oww = OwwModel()
                log.info("wake word: openWakeWord loaded (%d models)",
                         len(self._oww.models))
            except Exception as exc:  # noqa: BLE001
                log.warning("openWakeWord init failed: %s — transcript gate only", exc)

    # -- state ---------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        return time.time() < self.active_until

    def activate(self, seconds: float | None = None) -> None:
        """Manually activate (dashboard toggle / after a reply)."""
        self.active_until = time.time() + (seconds or self.active_timeout)

    def deactivate(self) -> None:
        self.active_until = 0.0

    # -- audio path (openWakeWord) ----------------------------------------------

    def feed_audio(self, chunk: bytes) -> bool:
        """Feed PCM16 chunk; returns True on wake detection (OWW only)."""
        if not self._oww:
            return False
        for i in range(0, len(chunk) - self._oww_frame_size + 1,
                       self._oww_frame_size):
            frame = list(chunk[i:i + self._oww_frame_size])
            try:
                scores = self._oww.predict(frame)
            except Exception:
                return False
            if any(s > 0.5 for s in scores.values()):
                log.info("wake word detected (openWakeWord)")
                self.activate()
                return True
        return False

    # -- transcript path (universal fallback) -------------------------------------

    def check_transcript(self, text: str) -> tuple[bool, str]:
        """Returns (accepted, cleaned_text).

        If gate is already active → accept everything.
        If not active, the transcript must contain a wake phrase; the
        remainder is treated as the actual command ("hey assistant, what's
        the weather" → accepted with "what's the weather").
        """
        t = (text or "").strip().lower()
        if self.is_active:
            self.active_until = time.time() + self.active_timeout  # extend
            return True, text
        for ph in self.phrases:
            if ph in t:
                cleaned = t.split(ph, 1)[1].strip(" ,.!?") if ph in t else ""
                self.activate()
                log.info("wake phrase %r accepted (remainder=%r)", ph, cleaned)
                return True, cleaned or text
        return False, ""

    # -- helpers -------------------------------------------------------------------

    @staticmethod
    def available() -> bool:
        return HAS_OWW

"""Hybrid provider: VoiceChat for voice I/O + separate VLM for screen.

Voice loop is delegated to the voicechat provider; when the assistant needs
to 'see' the screen, a screenshot is sent to an OpenAI-compatible VLM
(e.g. Qwen2-VL via llama.cpp server) and its description is injected into
the voice conversation as context.
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path

import requests

from . import config, tools as pai_tools
from .provider_voicechat import VoiceChatProvider

log = logging.getLogger("pai.hybrid")


class HybridProvider:
    name = "hybrid"

    def __init__(self):
        cfg = config.get_config()
        self.cfg = cfg
        self.vc = VoiceChatProvider()

    def start(self):
        self.vc.start()

    def stop(self):
        """Stop BOTH the voice model and release resources (no leaks)."""
        if hasattr(self.vc, "stop"):
            self.vc.stop()

    def is_running(self) -> bool:
        return self.vc.is_running()

    # -- vision ------------------------------------------------------------------

    def describe_screen(self, png_bytes: bytes, question: str = "") -> str:
        b64 = base64.b64encode(png_bytes).decode()
        payload = {
            "model": self.cfg.vlm_model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": question or
                     "Describe what is on the screen concisely: windows, "
                     "titles, buttons, any focus state. Be brief."},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }],
            "max_tokens": 300,
        }
        r = requests.post(f"{self.cfg.vlm_base_url}/chat/completions",
                          json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    # -- combined turn -------------------------------------------------------------

    def turn(self, wav_path: Path, executor: pai_tools.ToolExecutor,
             max_tool_rounds: int = 5) -> dict:
        # capture a fresh frame via the public API, then try to describe it
        executor.execute("screenshot", {})
        png = executor.cap.last_frame()
        screen_desc = ""
        if png:
            try:
                screen_desc = self.describe_screen(png)
                log.info("screen: %s", screen_desc[:200])
            except Exception as exc:  # noqa: BLE001
                log.warning("VLM describe failed (%s) — continuing voice-only",
                            exc)

        result = self.vc.turn(wav_path, executor, image_path=None,
                              max_tool_rounds=max_tool_rounds)
        if screen_desc and not result.get("audio"):
            result["text"] = (result.get("text", "") +
                              "\n[screen] " + screen_desc)
        return result

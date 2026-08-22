"""Provider: modular pipeline — Whisper ASR → OpenAI-compat LLM → TTS.

Reuses your Realtime Dubbing servers:
  ASR: http://127.0.0.1:9000/v1/audio/transcriptions  (whisper_server.py)
  TTS: http://127.0.0.1:8889/v1/audio/speech          (OmniVoice / dubbing_server.py)
LLM: any OpenAI-compatible endpoint (llama.cpp server, vLLM, LM Studio...).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import requests

from . import config, tools as pai_tools

log = logging.getLogger("pai.modular")


class ModularProvider:
    name = "modular"

    def __init__(self):
        cfg = config.get_config()
        self.cfg = cfg

    # -- ASR -------------------------------------------------------------------

    def transcribe(self, wav_path: Path) -> str:
        log.info("asr: posting %s to whisper server", wav_path.name)
        with open(wav_path, "rb") as f:
            r = requests.post(
                config.WHISPER_SERVER_URL,
                files={"file": (wav_path.name, f, "audio/wav")},
                data={"model": "whisper-1"},
                timeout=120,
            )
        r.raise_for_status()
        return (r.json().get("text") or "").strip()

    # -- LLM (chat with tool-calling via prompt convention) ---------------------

    SYSTEM_PROMPT = (
        "You are a personal desktop assistant running on the user's Windows PC. "
        "You can control the computer via tools. To use a tool, reply with ONLY a "
        "JSON object: {\"tool\": \"<name>\", \"params\": {...}} inside a ```json "
        "fenced block. Available tools:\n{tools}\n"
        "After a tool result you will be asked to continue. When you have the "
        "final spoken answer, reply with plain text (no JSON) — it will be spoken "
        "aloud. Keep spoken answers short and conversational."
    )

    def think(self, transcript: str, executor: pai_tools.ToolExecutor,
              screen_context: str = "", history: list[dict] | None = None,
              max_rounds: int = 6) -> dict:
        messages = [{"role": "system",
                     "content": self.SYSTEM_PROMPT.replace(
                         "{tools}", json.dumps(pai_tools.tool_schema(), indent=1))}]
        for m in (history or [])[-8:]:
            messages.append(m)
        if screen_context:
            messages.append({"role": "user",
                             "content": f"[Screen context] {screen_context}"})
        messages.append({"role": "user", "content": transcript})

        for _ in range(max_rounds):
            reply = self._chat(messages)
            calls = pai_tools.parse_tool_calls(reply)
            if not calls:
                return {"text": reply, "tool_calls": []}
            messages.append({"role": "assistant", "content": reply})
            results = []
            for name, params in calls:
                entry = executor.execute(name, params)
                results.append(entry)
            messages.append({"role": "user",
                             "content": "Tool results: "
                             + json.dumps(results, default=str)[:4000]})
        return {"text": "I hit the tool loop limit.", "tool_calls": []}

    def _chat(self, messages: list[dict]) -> str:
        r = requests.post(
            f"{self.cfg.llm_base_url}/chat/completions",
            json={"model": self.cfg.llm_model, "messages": messages,
                  "temperature": 0.4, "max_tokens": 512},
            timeout=180,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    # -- TTS -------------------------------------------------------------------

    def speak(self, text: str, out_path: Path) -> Path:
        if self.cfg.tts_backend == "edge":
            return self._edge_tts(text, out_path)
        r = requests.post(
            config.OMNIVOICE_BASE_URL + config.OMNIVOICE_TTS_ENDPOINT,
            json={"model": "omnivoice", "input": text,
                  "response_format": "wav"},
            timeout=120,
        )
        r.raise_for_status()
        out_path.write_bytes(r.content)
        return out_path

    def _edge_tts(self, text: str, out_path: Path) -> Path:
        import asyncio

        async def _run():
            import edge_tts
            await edge_tts.Communicate(text, "en-US-AriaNeural").save(
                str(out_path.with_suffix(".mp3")))
        asyncio.run(_run())
        return out_path.with_suffix(".mp3")

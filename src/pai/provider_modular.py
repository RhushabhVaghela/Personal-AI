"""Provider: modular pipeline — ASR → LLM (w/ memory) → TTS.

Works fully local (whisper_server + any OpenAI-compat LLM + OmniVoice)
or with online profiles (OpenAI / Groq) via config.PROFILES — same code
path, different endpoints + API keys from env vars.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import requests

from . import config, tools as pai_tools
from .memory import ConversationStore

log = logging.getLogger("pai.modular")


class ModularProvider:
    name = "modular"

    def __init__(self):
        cfg = config.get_config()
        self.cfg = cfg
        self.profile = cfg.profile
        self.memory = ConversationStore(
            session=cfg.session_name,
            max_turns=cfg.memory_turns,
            summarize_after=cfg.summarize_after if hasattr(cfg, "summarize_after")
            else cfg.memory_summarize_after)

    # -- helpers -----------------------------------------------------------------

    def _key_for(self, env_name: str | None) -> str | None:
        return self.cfg.api_key(env_name)

    def is_offline(self) -> bool:
        prof = config.PROFILES.get(self.profile, {})
        return prof.get("offline", True)

    # -- ASR -------------------------------------------------------------------

    def transcribe(self, wav_path: Path) -> str:
        log.info("asr[%s]: posting %s (%d bytes)",
                 self.profile, wav_path.name, wav_path.stat().st_size)
        t0 = time.time()
        key = self._key_for(config.PROFILES[self.profile]["asr_key_env"])
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        with open(wav_path, "rb") as f:
            r = requests.post(
                config.PROFILES[self.profile]["asr_url"],
                headers=headers,
                files={"file": (wav_path.name, f, "audio/wav")},
                data={"model": "whisper-1"},
                timeout=120,
            )
        r.raise_for_status()
        text = (r.json().get("text") or "").strip()
        log.info("asr[%s]: %.1fs -> %r", self.profile,
                 time.time() - t0, text[:80])
        return text

    # -- LLM (chat w/ tool-calling via prompt convention + memory context) ------

    SYSTEM_PROMPT = (
        "You are a personal desktop assistant running on the user's Windows PC. "
        "You can control the computer via tools. To use a tool, reply with ONLY a "
        "JSON object: {\"tool\": \"<name>\", \"params\": {...}} inside a ```json "
        "fenced block. Available tools:\n{tools}\n"
        "After a tool result you will be asked to continue. When you have the "
        "final spoken answer, reply with plain text (no JSON) — it will be spoken "
        "aloud. Keep spoken answers short and conversational. You remember "
        "earlier conversation from the provided context."
    )

    def think(self, transcript: str, executor: pai_tools.ToolExecutor,
              screen_context: str = "",
              history: list[dict] | None = None,
              max_rounds: int = 6) -> dict:
        messages = [{"role": "system",
                     "content": self.SYSTEM_PROMPT.replace(
                         "{tools}", json.dumps(pai_tools.tool_schema(), indent=1))}]
        # hours-long rolling memory
        for m in self.memory.context():
            messages.append(m)
        for m in (history or [])[-8:]:
            messages.append(m)
        if screen_context:
            messages.append({"role": "user",
                             "content": f"[Screen context] {screen_context}"})
        messages.append({"role": "user", "content": transcript})
        self.memory.add("user", transcript)

        for _ in range(max_rounds):
            reply = self._chat(messages)
            calls = pai_tools.parse_tool_calls(reply)
            if not calls:
                self.memory.add("assistant", reply)
                return {"text": reply, "tool_calls": []}
            messages.append({"role": "assistant", "content": reply})
            results = []
            for name, params in calls:
                entry = executor.execute(name, params)
                results.append(entry)
                self._broadcast_tool(entry, executor)
            messages.append({"role": "user",
                             "content": "Tool results: "
                             + json.dumps(results, default=str)[:4000]})
        return {"text": "I hit the tool loop limit.", "tool_calls": []}

    def _broadcast_tool(self, entry: dict, executor: pai_tools.ToolExecutor) -> None:
        cb = getattr(executor, "on_event", None)
        if cb:
            try:
                cb(entry)
            except Exception:  # noqa: BLE001
                pass

    def _chat(self, messages: list[dict]) -> str:
        key = self._key_for(config.PROFILES[self.profile]["llm_key_env"])
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        r = requests.post(
            f"{self.cfg.llm_base_url}/chat/completions",
            headers=headers,
            json={"model": self.cfg.llm_model, "messages": messages,
                  "temperature": 0.4, "max_tokens": 512},
            timeout=180,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def switch_profile(self, profile: str) -> dict:
        info = self.cfg.apply_profile(profile)
        self.profile = profile
        log.info("modular switched to profile %s (%s)", profile,
                 config.PROFILES[profile]["label"])
        return info

    # -- TTS ---------------------------------------------------------------------

    def speak(self, text: str, out_path: Path) -> Path:
        backend = self.cfg.tts_backend
        if backend == "edge":
            return self._edge_tts(text, out_path)
        if backend == "openai":
            return self._openai_tts(text, out_path)
        if backend == "browser":
            raise BrowserTTSSignal(
                "browser TTS selected — client renders speech itself")
        # default: local OmniVoice server
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

    def _openai_tts(self, text: str, out_path: Path) -> Path:
        key = self._key_for(config.PROFILES[self.profile].get("tts_key_env")
                            or "OPENAI_API_KEY")
        r = requests.post(
            "https://api.openai.com/v1/audio/speech",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": "tts-1", "input": text[:4096],
                  "voice": "alloy", "response_format": "mp3"},
            timeout=120,
        )
        r.raise_for_status()
        out = out_path.with_suffix(".mp3")
        out.write_bytes(r.content)
        return out


class BrowserTTSSignal(RuntimeError):
    """Raised when tts_backend='browser' — dashboard speaks via Web Speech."""

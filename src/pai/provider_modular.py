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
        self.session = requests.Session()
        self.memory = ConversationStore(
            session=cfg.session_name,
            max_turns=cfg.memory_turns,
            summarize_after=cfg.memory_summarize_after)
        # LLM-backed summaries (graceful fallback to extractive)
        self.memory.attach_llm_summarizer(
            lambda msgs: self._chat(msgs))

    # -- helpers -----------------------------------------------------------------

    def _key_for(self, env_name: str | None) -> str | None:
        return self.cfg.api_key(env_name)

    def is_offline(self) -> bool:
        prof = config.PROFILES.get(self.profile, {})
        return prof.get("offline", True)

    # -- ASR -------------------------------------------------------------------

    def transcribe(self, wav_path: Path) -> str:
        # optional VibeVoice-ASR-BitNet (realtime ASR on CPU via llama.cpp)
        vv_url = getattr(self.cfg, "vibevoice_asr_url", "")
        if vv_url:
            try:
                with open(wav_path, "rb") as f:
                    r = self.session.post(
                        f"{vv_url}/v1/audio/transcriptions",
                        files={"file": (wav_path.name, f, "audio/wav")},
                        timeout=60)
                if r.ok:
                    text = (r.json().get("text") or "").strip()
                    log.info("asr[vibevoice-bitnet]: %r", text[:80])
                    return text
            except Exception as exc:  # noqa: BLE001
                log.warning("vibevoice ASR failed (%s) — falling back", exc)
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
        messages = self._build_messages(transcript, executor, screen_context)
        for m in (history or [])[-8:]:
            messages.insert(-1, m)   # before the current user turn

        for _ in range(max_rounds):
            reply = self._chat(messages)
            # strip reasoning tags that quantized models leak into the
            # text channel (would otherwise break JSON tool parsing)
            from .stream_sanitizer import sanitize_full
            clean_reply = sanitize_full(reply)
            calls = pai_tools.parse_tool_calls(clean_reply)
            if not calls:
                self.memory.add("assistant", clean_reply)
                return {"text": clean_reply, "tool_calls": []}
            messages.append({"role": "assistant", "content": clean_reply})
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

    def _build_messages(self, transcript: str,
                        executor: pai_tools.ToolExecutor,
                        screen_context: str = "") -> list[dict]:
        """Shared prompt assembly (used by think() and the streaming path)."""
        messages = [{"role": "system",
                     "content": self.SYSTEM_PROMPT.replace(
                         "{tools}", json.dumps(pai_tools.tool_schema(), indent=1))}]
        for m in self.memory.context():
            messages.append(m)
        if screen_context:
            messages.append({"role": "user",
                             "content": f"[Screen context] {screen_context}"})
        messages.append({"role": "user", "content": transcript})
        self.memory.add("user", transcript)
        return messages

    def _chat(self, messages: list[dict], deep: bool = False) -> str:
        """Non-streaming chat (used for tool-loop rounds)."""
        text = ""
        for chunk in self._chat_stream(messages, deep):
            text += chunk
        return text

    def _chat_stream(self, messages: list[dict], deep: bool = False):
        """Stream LLM output token-by-token via SSE (yields text deltas)."""
        base_url = self.cfg.llm_base_url
        model = self.cfg.llm_model
        key_env = config.PROFILES[self.profile]["llm_key_env"]
        if deep:
            if self.cfg.deep_llm_base_url:
                base_url = self.cfg.deep_llm_base_url
            if self.cfg.deep_llm_model:
                model = self.cfg.deep_llm_model
            log.info("llm[%s]: DEEP reasoning via %s", self.profile, model)
        else:
            log.info("llm[%s]: instant via %s", self.profile, model)
        key = self._key_for(key_env)
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        r = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json={"model": model, "messages": messages,
                  "temperature": 0.4, "max_tokens": 512, "stream": True},
            timeout=300 if deep else 180,
            stream=True,
        )
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload.strip() == "[DONE]":
                break
            try:
                delta = json.loads(payload)["choices"][0].get("delta", {})
                piece = delta.get("content") or ""
                if piece:
                    yield piece
            except Exception:  # noqa: BLE001
                continue

    def speak_streaming(self, text_stream, on_sentence=None) -> str:
        """Sentence-chunked streaming TTS: synthesize+return each sentence as
        soon as it completes. Returns the full text.

        Yields (sentence, audio_path_or_None) via on_sentence callback so the
        caller can start playback while generation continues.
        """
        import re as _re
        buf = ""
        full = ""
        sentences = []
        for piece in text_stream:
            buf += piece
            full += piece
            # sentence boundary heuristic
            m = None
            for mm in _re.finditer(r"[.!?](\s|$)", buf):
                m = mm
            if m:
                sentence = buf[:m.end()].strip()
                buf = buf[m.end():]
                if sentence:
                    sentences.append(sentence)
                    if on_sentence:
                        try:
                            audio_path = self._tts_one(sentence)
                            on_sentence(sentence, audio_path)
                        except Exception as exc:  # noqa: BLE001
                            log.warning("streaming TTS chunk failed: %s", exc)
                            on_sentence(sentence, None)
        tail = buf.strip()
        if tail:
            sentences.append(tail)
            if on_sentence:
                try:
                    audio_path = self._tts_one(tail)
                    on_sentence(tail, audio_path)
                except Exception:  # noqa: BLE001
                    on_sentence(tail, None)
        return full

    def _tts_one(self, sentence: str, idx: int = 0):
        """Fast single-sentence TTS → temp file (honors cfg.tts_engine)."""
        import tempfile
        out = Path(tempfile.gettempdir()) / f"pai_stream_{int(time.time()*1000)}_{idx}"
        engine = getattr(self.cfg, "tts_engine", "edge")
        if engine in ("omnivoice", "zipvoice", "vibevoice"):
            from . import tts_engines
            return tts_engines.synthesize(
                sentence, out.with_suffix(".wav"), engine=engine,
                ref_audio=(getattr(self.cfg, "tts_clone_ref", "") or None))
        if engine == "openai":
            return self._openai_tts(sentence, out.with_suffix(".mp3"))
        # edge (default; online but fast)
        try:
            return self._edge_tts(sentence, out.with_suffix(".mp3"))
        except Exception as exc:  # noqa: BLE001
            log.warning("edge TTS failed (%s); trying omnivoice server", exc)
            r = requests.post(
                config.OMNIVOICE_BASE_URL + config.OMNIVOICE_TTS_ENDPOINT,
                json={"model": "omnivoice", "input": sentence,
                      "response_format": "wav"}, timeout=60)
            p = out.with_suffix(".wav")
            p.write_bytes(r.content)
            return p

    def set_effort(self, effort: str) -> None:
        self.cfg.reasoning_effort = effort if effort in ("instant", "deep") \
            else "instant"

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
        if backend in ("omnivoice", "zipvoice", "vibevoice"):
            # local neural engines (tts_engines.py) — offline, tiny VRAM
            from . import tts_engines
            return tts_engines.synthesize(
                text, out_path.with_suffix(".wav"), engine=backend,
                ref_audio=(self.cfg.tts_clone_ref or None))
        # default: local OmniVoice HTTP server (your dubbing stack)
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
            rate_pct = int(round((self.cfg.tts_speed - 1.0) * 100))
            kwargs = {"voice": self.cfg.tts_voice}
            if rate_pct:
                kwargs["rate"] = f"{rate_pct:+d}%"
            await edge_tts.Communicate(text, **kwargs).save(
                str(out_path.with_suffix(".mp3")))
        asyncio.run(_run())
        return out_path.with_suffix(".mp3")

    def _openai_tts(self, text: str, out_path: Path) -> Path:
        key = self._key_for(config.PROFILES[self.profile].get("tts_key_env")
                            or "OPENAI_API_KEY")
        speed = max(0.25, min(4.0, float(self.cfg.tts_speed)))
        r = requests.post(
            "https://api.openai.com/v1/audio/speech",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": "tts-1", "input": text[:4096],
                  "voice": getattr(self.cfg, "tts_openai_voice", "alloy"),
                  "speed": speed, "response_format": "mp3"},
            timeout=120,
        )
        r.raise_for_status()
        out = out_path.with_suffix(".mp3")
        out.write_bytes(r.content)
        return out


class BrowserTTSSignal(RuntimeError):
    """Raised when tts_backend='browser' — dashboard speaks via Web Speech."""

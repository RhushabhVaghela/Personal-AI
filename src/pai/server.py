"""Web dashboard: WebSocket UI for the assistant.

Serves static/ (dashboard) and a WebSocket endpoint where the browser
captures mic audio (MediaRecorder → wav/webm chunks) and receives
transcripts, assistant audio, screenshots and tool events live.

Run: python -m pai.server  →  http://localhost:8765
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import socket
import subprocess
import threading
import time
from pathlib import Path

from . import config, input_control, providers, tools
from .vad import get_engine, listen_continuous
from .wakeword import WakeWordGate
from .screenshare import ScreenShareStreamer

log = logging.getLogger("pai.server")

STATIC = Path(__file__).resolve().parents[2] / "static"
ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "logs"

# Single shared provider across dashboard connections — prevents model
# processes (and VRAM) piling up on browser refreshes/reconnects.
_active_provider = None
_active_name: str | None = None
_provider_lock = asyncio.Lock()
_MAIN_LOOP: asyncio.AbstractEventLoop | None = None
_HANDSFREE = {"thread": None, "stop": None, "engine": None,
              "session": None}
_WAKE_GATE = {"gate": None}
_SHARE = {"streamer": None, "cam": None}
_SCHEDULER = {"stop": None}
_CLIENTS: set = set()   # live dashboard websockets (for proactive pushes)


async def _get_shared_provider(name: str):
    """Return the shared provider instance, switching cleanly if needed.

    Guarantees the OLD model process is fully stopped (RAM/VRAM freed)
    before the new one loads.
    """
    global _active_provider, _active_name
    async with _provider_lock:
        if _active_provider is not None and _active_name == name \
                and getattr(_active_provider, "is_running", lambda: True)():
            return _active_provider
        if _active_provider is not None:
            log.info("switching provider %s -> %s; stopping old model",
                     _active_name, name)
            if hasattr(_active_provider, "stop"):
                await asyncio.get_running_loop().run_in_executor(
                    None, _active_provider.stop)
            _active_provider = None
            _active_name = None
        prov = providers.get_provider(name)
        if hasattr(prov, "start"):
            await asyncio.get_running_loop().run_in_executor(None, prov.start)
        _active_provider = prov
        _active_name = name
        return prov


class AssistantSession:
    """One connected dashboard = one session."""

    def __init__(self, ws):
        self.ws = ws
        self.cfg = config.get_config()
        # NOTE: never mutate the global config singleton per-session —
        # each turn resolves its provider explicitly.
        self.executor = tools.ToolExecutor(
            autonomy=self.cfg.autonomy,
            on_screenshot=self._push_screenshot,
            on_event=self._push_tool_event,
            on_card=self._push_card)
        self.latest_png: bytes | None = None

    @property
    def provider(self):
        return _active_provider

    def _push_screenshot(self, png: bytes) -> None:
        self.latest_png = png

    def _push_tool_event(self, entry: dict) -> None:
        """Broadcast tool activity to the dashboard (#tools panel).

        Called from worker threads — must hop to the main event loop.
        """
        line = f"{entry['tool']}({json.dumps(entry['params'])[:60]}) -> " \
               f"{'OK' if entry.get('ok') else entry.get('result')}"
        if _MAIN_LOOP is not None and _MAIN_LOOP.is_running():
            asyncio.run_coroutine_threadsafe(
                self.send("tool", text=line, ok=bool(entry.get("ok"))),
                _MAIN_LOOP)

    def _push_card(self, kind: str, data: dict) -> None:
        """Answer cards (GPT-Live widgets): clock, weather, reminder…"""
        if _MAIN_LOOP is not None and _MAIN_LOOP.is_running():
            asyncio.run_coroutine_threadsafe(
                self.send("card", card=kind, **data), _MAIN_LOOP)

    async def deliver_reminder(self, item: dict) -> None:
        """Proactive speech: the assistant speaks up on its own."""
        text = f"Reminder: {item['text']}"
        log.info("delivering reminder: %s", item["text"][:60])
        await self.send("reply", text=f"⏰ {text}")
        provider = _active_provider
        if provider is not None and getattr(provider, "name", "") == "modular":
            try:
                loop = asyncio.get_running_loop()
                spoken = await loop.run_in_executor(
                    None, provider.speak, text,
                    LOG_DIR / "reminder.wav")
                await self._send_audio(spoken)
                return
            except Exception as exc:  # noqa: BLE001
                log.warning("reminder TTS failed (%s) — text only", exc)
        # non-modular: send a tiny edge-tts clip best-effort
        try:
            import asyncio as _a

            async def _synth():
                import edge_tts
                mp3 = Path(__import__("tempfile").gettempdir()) / "pai_rem.mp3"
                await edge_tts.Communicate(text).save(str(mp3))
                return mp3
            mp3 = await _synth()
            await self._send_audio(mp3)
        except Exception as exc:  # noqa: BLE001
            log.warning("reminder fallback TTS failed: %s", exc)

    async def send(self, kind: str, **data) -> None:
        try:
            await self.ws.send(json.dumps({"kind": kind, **data}))
        except Exception:  # noqa: BLE001
            pass

    async def handle(self, msg: dict) -> None:
        kind = msg.get("kind")
        try:
            await self._handle(msg, kind)
        except Exception as exc:  # noqa: BLE001
            # a failed backend call must NEVER drop the websocket
            log.error("handle(%s) failed: %s", kind, exc, exc_info=True)
            await self.send("error", text=f"{kind} failed: {exc}")

    async def _handle(self, msg: dict, kind: str) -> None:
        global _active_provider, _active_name
        if kind == "config":
            await self.send("config_ok",
                            provider=_active_name or "none",
                            autonomy=self.cfg.autonomy)
        elif kind == "switch":
            # switch the shared provider without dropping the connection
            name = msg.get("provider", "voicechat")
            await self.send("status", text=f"switching to {name}...")
            await _get_shared_provider(name)
            await self.send("ready", provider=name)
        elif kind == "audio_chunk":
            data = base64.b64decode(msg.get("data", ""))
            wav = self._to_wav(data)
            await self._turn(wav)
        elif kind == "text":
            await self._turn_text(msg.get("text", ""))
        elif kind == "screenshot":
            self.executor.execute("screenshot", {})
            if self.latest_png:
                await self.send("screenshot",
                                image=base64.b64encode(
                                    self.latest_png).decode())
        elif kind == "kill":
            ks = input_control.get_kill_switch()
            ks.toggle()
            await self.send("kill_state", engaged=ks.engaged)
        elif kind == "hands_free":
            await self._toggle_hands_free(bool(msg.get("on")))
        elif kind == "wake_mode":
            # Gemini 'proactive audio' parity: hands-free stays armed until
            # the wake phrase is heard (or OWW audio detection fires)
            on = bool(msg.get("on"))
            gate_on = _WAKE_GATE["gate"] is not None and on
            if on and not gate_on:
                _WAKE_GATE["gate"] = WakeWordGate(
                    phrases=msg.get("phrases") or self.cfg.wake_phrases,
                    oww_models=getattr(self.cfg, "wake_oww_models", None),
                    custom_model=getattr(self.cfg, "wake_custom_model", ""))
            elif not on:
                if _WAKE_GATE["gate"]:
                    _WAKE_GATE["gate"].deactivate()
                _WAKE_GATE["gate"] = None
                await self.send("wake_mode", on=False)
                await self._toggle_hands_free(False)
                return
            _WAKE_GATE["gate"].activate()   # first turn free, then re-arms
            await self._toggle_hands_free(True)
            await self.send("wake_mode", on=True,
                            oww=WakeWordGate.available())
        elif kind == "set_voice":
            # ChatGPT-style voice + speed selection
            voice = msg.get("voice")
            speed = msg.get("speed")
            if voice:
                self.cfg.tts_voice = str(voice)
            if speed:
                self.cfg.tts_speed = max(0.5, min(2.0, float(speed)))
            await self.send("voice_ok", voice=self.cfg.tts_voice,
                            speed=self.cfg.tts_speed)
        elif kind == "set_voicechat_tts":
            mode = msg.get("mode", "native")
            self.cfg.voicechat_tts_mode = mode if mode in ("native", "edge") \
                else "native"
            await self.send("status",
                text="🎙 Nemotron re-voicing ON — replies use your selected "
                     "voice" if self.cfg.voicechat_tts_mode == "edge"
                else "🎙 Nemotron native voice (fastest)")
        elif kind == "set_tts_engine":
            # modular TTS engine: edge | omnivoice | zipvoice | vibevoice | openai
            engine = msg.get("engine", "")
            valid = {"edge", "omnivoice", "zipvoice", "vibevoice",
                     "openai", "browser"}
            if engine in valid:
                self.cfg.tts_backend = engine
                self.cfg.tts_engine = engine
                await self.send(
                    "status",
                    text=f"🔊 TTS engine: {engine}" +
                         (" (offline)" if engine in
                          ("omnivoice", "zipvoice", "vibevoice") else ""))
            else:
                await self.send("error", text=f"unknown TTS engine {engine!r}")
        elif kind == "set_effort":
            # GPT-Live reasoning effort: instant | deep (escalates model)
            effort = msg.get("effort", "instant")
            self.cfg.reasoning_effort = effort
            deep_info = ""
            if effort == "deep":
                # spin up the on-demand 30B-class brain (lazy, auto-unloads)
                try:
                    from .deep_brain import get_deep_brain
                    db = get_deep_brain()
                    if getattr(self.cfg, "deep_pause_s2s", True) and \
                            _active_provider is not None and \
                            hasattr(_active_provider, "stop"):
                        await asyncio.get_event_loop().run_in_executor(
                            None, _active_provider.stop)
                        await self.send(
                            "status",
                            text="🧠 pausing voice model to make room…")
                    await asyncio.get_event_loop().run_in_executor(
                        None, db.ensure_up)
                    deep_info = f" — {db.hf_repo.split(':')[0].split('/')[-1]} loaded"
                    self.cfg.deep_llm_base_url = \
                        f"http://127.0.0.1:{db.port}/v1"
                    self.cfg.deep_llm_model = "deep-brain"
                    # the paused S2S model will restart on next use
                    _active_provider = None
                    _active_name = None
                except Exception as exc:  # noqa: BLE001
                    log.error("deep brain failed: %s", exc)
                    deep_info = f" (brain load failed: {exc})"
            elif effort == "instant":
                from .deep_brain import get_deep_brain
                get_deep_brain().unload()   # free VRAM when back to instant
            await self.send("status",
                            text=f"🧠 {effort} mode{deep_info}")
            if hasattr(_active_provider, "set_effort"):
                _active_provider.set_effort(effort)
        elif kind == "session":
            await self._handle_session(msg)
        elif kind == "share_screen":
            if msg.get("camera"):
                await self._toggle_webcam(bool(msg.get("on")))
            else:
                await self._toggle_share_screen(bool(msg.get("on")))
        elif kind == "stop_speak":
            # user interrupted the assistant: stop audio + reopen the mic
            _HANDSFREE["engine"].set_speaking(False) \
                if _HANDSFREE.get("engine") else None
            await self.send("status", text="stopped — mic open")
        elif kind == "switch_profile":
            profile = msg.get("profile", "local")
            try:
                self.cfg.apply_profile(profile)
                if hasattr(_active_provider, "switch_profile"):
                    _active_provider.switch_profile(profile)
                prof = config.PROFILES[profile]
                await self.send("status",
                                text=f"profile: {prof['label']}")
                await self.send("ready",
                                provider=_active_name or "modular")
            except Exception as exc:  # noqa: BLE001
                await self.send("error", text=f"profile switch failed: {exc}")

    # -- session / transcript management (ChatGPT-style history) -----------------

    def _memory(self):
        prov = _active_provider
        mem = getattr(prov, "memory", None)
        if mem is None:
            from .memory import ConversationStore
            cfg = self.cfg
            mem = ConversationStore(session=cfg.session_name,
                                    max_turns=cfg.memory_turns,
                                    summarize_after=cfg.memory_summarize_after)
        return mem

    async def _handle_session(self, msg: dict) -> None:
        action = msg.get("action", "stats")
        mem = self._memory()
        if action == "new":
            mem.reset()
            await self.send("status", text="🆕 new conversation started")
            await self.send("session", action="new", stats=mem.stats())
        elif action == "export":
            path = mem.path
            if path.exists():
                import shutil
                dest = LOG_DIR / f"session_export_{int(time.time())}.jsonl"
                shutil.copy2(path, dest)
                await self.send("status", text=f"exported → {dest.name}")
                await self.send("session", action="exported", path=str(dest))
            else:
                await self.send("error", text="no session yet")
        else:  # stats / transcript
            turns = [{"role": t["role"], "text": t["text"], "ts": t["ts"]}
                     for t in list(mem.turns)[-200:]]
            await self.send("session", action="transcript", turns=turns,
                            summary=mem.summary, stats=mem.stats())

    # -- hands-free (VAD) ---------------------------------------------------------

    async def _toggle_webcam(self, on: bool) -> None:
        from .webcam import WebcamCapture
        if on and not _SHARE["cam"]:
            cam = WebcamCapture()
            if not cam.start():
                await self.send("error",
                                text="webcam unavailable (install opencv-python?)")
                return
            _SHARE["cam"] = cam
            await self.send("share_screen", on=True, camera=True)
            await self.send("status", text="📷 camera sharing")
        elif not on and _SHARE["cam"]:
            _SHARE["cam"].stop()
            _SHARE["cam"] = None
            await self.send("share_screen", on=False, camera=True)

    async def _toggle_share_screen(self, on: bool) -> None:
        if on and not _SHARE["streamer"]:
            def push(png: bytes):
                if _MAIN_LOOP and _MAIN_LOOP.is_running():
                    asyncio.run_coroutine_threadsafe(
                        self.send("screen_stream",
                                  image=base64.b64encode(png).decode()),
                        _MAIN_LOOP)
            s = ScreenShareStreamer(on_frame=push)
            if s.start():
                _SHARE["streamer"] = s
                await self.send("share_screen", on=True)
                await self.send("status", text="🖥 sharing your screen")
        elif not on and _SHARE["streamer"]:
            _SHARE["streamer"].stop()
            _SHARE["streamer"] = None
            await self.send("share_screen", on=False)

    async def _toggle_hands_free(self, on: bool) -> None:
        hf = _HANDSFREE
        if on and hf["thread"] is not None:
            # already on — re-ack so the button doesn't feel dead
            await self.send("hands_free", on=True,
                            engine=type(hf["engine"]).__name__)
            return
        if on:
            cfg = self.cfg
            engine = get_engine(
                prefer_webrtc=cfg.vad_engine in ("auto", "webrtc"),
                sample_rate=cfg.sample_rate,
                silence_hangover_ms=cfg.vad_silence_ms)
            stop = threading.Event()
            hf.update(engine=engine, stop=stop, session=self)

            async def handle_utterance(wav_path):
                log.info("vad: utterance %s (%d bytes)", wav_path.name,
                         wav_path.stat().st_size)
                # wake-word gate (proactive audio): if armed and the
                # transcript lacks a wake phrase, drop the turn.
                gate = _WAKE_GATE["gate"]
                if gate and not gate.is_active:
                    try:
                        from .provider_modular import ModularProvider
                        probe = ModularProvider().transcribe(wav_path)
                        ok, cleaned = gate.check_transcript(probe)
                        if not ok:
                            log.info("vad: dropped (no wake phrase): %r",
                                     probe[:60])
                            return
                        # re-synthesize cleaned text for providers that need audio
                        if (_active_name or "") in ("voicechat", "hybrid"):
                            wav_path = self._tts_input(cleaned) \
                                if cleaned else wav_path
                    except Exception as exc:  # noqa: BLE001
                        log.warning("wake gate check failed (%s); accepting", exc)

                engine.set_speaking(True)   # close mic gate during reply
                try:
                    await self._turn(wav_path)
                    gate.activate() if gate else None   # follow-ups stay open
                finally:
                    engine.set_speaking(False)

            def on_utterance(wav_path):
                if _MAIN_LOOP and _MAIN_LOOP.is_running():
                    asyncio.run_coroutine_threadsafe(
                        handle_utterance(wav_path), _MAIN_LOOP)

            def on_state(state):
                if _MAIN_LOOP and _MAIN_LOOP.is_running() and state == "speaking":
                    asyncio.run_coroutine_threadsafe(
                        self.send("vad", state="listening"), _MAIN_LOOP)

            hf["thread"] = threading.Thread(
                target=lambda: listen_continuous(
                    engine, on_utterance, on_state=on_state,
                    stop_flag=stop), daemon=True)
            hf["thread"].start()
            log.info("hands-free ON (%s VAD)",
                     type(engine).__name__)
            await self.send("hands_free", on=True,
                            engine=type(engine).__name__)
            await self.send("status",
                            text="👂 hands-free — just start talking")
        elif not on:
            if hf["thread"] is None:
                # already off — re-ack so the button state stays in sync
                await self.send("hands_free", on=False)
                return
            hf["stop"].set()
            hf["thread"].join(timeout=3)
            hf.update(thread=None, stop=None, engine=None, session=None)
            await self.send("hands_free", on=False)
            log.info("hands-free OFF")

    # -- audio plumbing ---------------------------------------------------------

    def _to_wav(self, blob: bytes) -> Path:
        import tempfile
        # sniff container: RIFF (wav), EBML (webm/matroska, what
        # MediaRecorder produces on Chrome/Edge), ID3/mp3 sync
        if blob[:4] == b"RIFF":
            suffix = ".wav"
        elif blob[:4] == b"\x1a\x45\xdf\xa3":
            suffix = ".webm"
        elif blob[:3] == b"ID3" or (len(blob) > 2 and blob[0] == 0xFF):
            suffix = ".mp3"
        else:
            suffix = ".webm"  # MediaRecorder default on Chromium
        p = Path(tempfile.gettempdir()) / f"pai_chunk{suffix}"
        p.write_bytes(blob)
        log.info("audio chunk: %d bytes, detected %s", len(blob), suffix)
        if suffix != ".wav":
            p = self._ffmpeg_to_wav(p)
        return p

    def _ffmpeg_to_wav(self, p: Path) -> Path:
        out = p.with_suffix(".wav")
        r = subprocess.run(["ffmpeg", "-y", "-i", str(p), "-ar", "16000",
                            "-ac", "1", str(out)],
                           capture_output=True, timeout=60)
        if r.returncode != 0 or not out.exists():
            log.error("ffmpeg convert failed: %s", r.stderr.decode()[-400:])
            raise RuntimeError("audio conversion failed (is ffmpeg on PATH?)")
        log.info("converted %s -> wav (%d bytes)",
                 p.suffix, out.stat().st_size)
        return out

    # -- turns -------------------------------------------------------------------

    async def _current_provider(self):
        if _active_provider is None:
            raise RuntimeError("no active provider — say hello first")
        return _active_provider

    async def _turn(self, wav: Path) -> None:
        provider = await self._current_provider()
        loop = asyncio.get_running_loop()
        if provider.name == "modular":
            await self.send("status", text="transcribing (Whisper)...")
            transcript = await loop.run_in_executor(
                None, provider.transcribe, wav)
            await self.send("transcript", text=transcript)
            # GPT-Live-style backchannel while the model thinks
            loop.run_in_executor(None, self._play_backchannel)
            # streaming path: sentence-chunked TTS as the LLM generates.
            # A StreamSanitizer FSM strips reasoning tags (<think>…) that
            # split across SSE chunks — they never reach the UI or TTS.
            await self.send("status", text="thinking...")
            t0 = time.time()
            messages = provider._build_messages(transcript, self.executor)
            got_tool_call = False

            def on_thinking(t):
                if _MAIN_LOOP and _MAIN_LOOP.is_running():
                    asyncio.run_coroutine_threadsafe(
                        self.send("thinking", delta=t), _MAIN_LOOP)

            from .stream_sanitizer import sanitize_stream
            raw = (piece for piece in provider._chat_stream(
                messages, deep=self.cfg.reasoning_effort == "deep"))

            def gen():
                nonlocal got_tool_call
                for clean_piece in sanitize_stream(raw, on_thinking=on_thinking):
                    s = clean_piece.lstrip()
                    if not got_tool_call and s[:1] in ("{", "`"):
                        got_tool_call = True   # tool-call reply — abort stream
                        return
                    yield clean_piece

            def on_sentence(sentence, audio_path):
                if audio_path and _MAIN_LOOP and _MAIN_LOOP.is_running():
                    asyncio.run_coroutine_threadsafe(
                        self._send_audio(audio_path), _MAIN_LOOP)

            full = await loop.run_in_executor(
                None, lambda: provider.speak_streaming(gen(), on_sentence))
            if got_tool_call:
                # classic tool loop handles JSON replies
                result = await loop.run_in_executor(
                    None, provider.think, transcript, self.executor)
                await self.send("reply", text=result["text"])
                if result["text"]:
                    spoken = await loop.run_in_executor(
                        None, provider.speak, result["text"],
                        LOG_DIR / "reply.wav")
                    await self._send_audio(spoken)
                return
            log.info("streamed turn finished in %.1fs", time.time() - t0)
            provider.memory.add("assistant", full)
            await self.send("reply", text=full)
        else:
            log.info("turn: provider=%s wav=%s (%d bytes) — processing",
                     provider.name, wav.name, wav.stat().st_size)
            await self.send("status", text="thinking (model is generating)...")
            t0 = time.time()
            result = await loop.run_in_executor(
                None, provider.turn, wav, self.executor)
            dt = time.time() - t0
            audio = result.get("audio")
            log.info("turn: finished in %.1fs, audio=%s", dt, audio)
            text = result.get("text") or ""
            if not text and audio:
                # voicechat replies are audio-only: transcribe so chat shows it
                await self.send("status", text="transcribing reply...")
                try:
                    text = await loop.run_in_executor(
                        None, self._transcribe, Path(audio))
                except Exception as exc:  # noqa: BLE001
                    log.warning("reply transcription failed: %s", exc)
                    text = "(voice reply)"
                log.info("turn: reply transcript: %s", text[:200])
            await self.send("reply", text=text or "(empty reply)")
            if audio:
                # optional re-voicing: swap the built-in Nemotron voice for
                # the user-selected edge-tts voice + speed
                if (getattr(self.cfg, "voicechat_tts_mode", "native") == "edge"
                        and text and text != "(voice reply)"):
                    await self.send(
                        "status", text="🎙 applying selected voice...")
                    try:
                        from .provider_modular import ModularProvider
                        mp = ModularProvider()
                        out = LOG_DIR / f"reply_voiced_{int(time.time())}.mp3"
                        audio = await loop.run_in_executor(
                            None, mp._edge_tts, text, out)
                        log.info("turn: re-voiced as %s @ %sx",
                                 self.cfg.tts_voice, self.cfg.tts_speed)
                    except Exception as exc:  # noqa: BLE001
                        log.warning(
                            "re-voice failed (%s) — keeping native voice",
                            exc)
                await self._send_audio(Path(audio))

    _ack_cache: dict = {}

    def _play_backchannel_sync(self) -> None:
        """Synthesize the ack clip if not cached (runs in worker thread)."""
        cfg = self.cfg
        key = f"{cfg.tts_voice}:{cfg.tts_speed}"
        mp3 = Path(__import__("tempfile").gettempdir()) / \
            f"pai_ack_{abs(hash(key)) % 99999}.mp3"
        if key in self._ack_cache and mp3.exists():
            return
        import edge_tts
        asyncio.run(edge_tts.Communicate(
            "Mm-hmm?", voice=cfg.tts_voice).save(str(mp3)))
        self._ack_cache[key] = mp3.read_bytes()

    async def _play_backchannel(self) -> None:
        """Play a short acknowledgement clip (cached, best-effort)."""
        try:
            cfg = self.cfg
            if not getattr(cfg, "backchannel", True):
                return
            key = f"{cfg.tts_voice}:{cfg.tts_speed}"
            mp3 = Path(__import__("tempfile").gettempdir()) / \
                f"pai_ack_{abs(hash(key)) % 99999}.mp3"
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._play_backchannel_sync)
            if key in self._ack_cache:
                await self.send("backchannel",
                                data=base64.b64encode(
                                    self._ack_cache[key]).decode(),
                                format="mp3")
        except Exception as exc:  # noqa: BLE001
            log.debug("backchannel skipped: %s", exc)

    def _transcribe(self, wav: Path) -> str:
        """Transcribe a reply wav via the local Whisper server (best-effort)."""
        try:
            import requests
            with open(wav, "rb") as f:
                r = requests.post(
                    config.WHISPER_SERVER_URL,
                    files={"file": (wav.name, f, "audio/wav")},
                    data={"model": "whisper-1"}, timeout=60)
            if r.ok:
                return (r.json().get("text") or "").strip()
        except Exception:  # noqa: BLE001
            pass
        return "(voice reply)"

    async def _turn_text(self, text: str) -> None:
        provider = await self._current_provider()
        if provider.name == "modular":
            loop = asyncio.get_running_loop()
            await self.send("transcript", text=text)
            # GPT-Live-style backchannel while the model thinks
            loop.run_in_executor(None, self._play_backchannel)
            await self.send("status", text="thinking...")
            result = await loop.run_in_executor(
                None, provider.think, text, self.executor)
            await self.send("reply", text=result["text"])
            if result["text"]:
                await self.send("status", text="speaking (TTS)...")
                spoken = await loop.run_in_executor(
                    None, provider.speak, result["text"],
                    LOG_DIR / "reply.wav")
                await self._send_audio(spoken)
        else:
            # voicechat/hybrid accept only audio turns: synthesize the text
            # to a wav first, then run it through the normal voice turn.
            loop = asyncio.get_running_loop()
            await self.send("status", text="synthesizing text input...")
            try:
                wav = await loop.run_in_executor(None, self._tts_input, text)
            except Exception as exc:  # noqa: BLE001
                await self.send("reply",
                                text=f"(text-to-speech failed: {exc})")
                return
            await self._turn(wav)

    def _tts_input(self, text: str) -> Path:
        """Speak the user's typed text into a 16k mono wav for the model."""
        import tempfile

        async def _synth():
            import edge_tts
            tmp = Path(tempfile.gettempdir()) / "pai_text_input.mp3"
            await edge_tts.Communicate(
                text, voice=self.cfg.tts_voice).save(str(tmp))
            return tmp

        mp3 = asyncio.run(_synth())
        wav = mp3.with_suffix(".wav")
        r = subprocess.run(["ffmpeg", "-y", "-i", str(mp3), "-ar", "16000",
                            "-ac", "1", str(wav)],
                           capture_output=True, timeout=60)
        if r.returncode != 0 or not wav.exists():
            raise RuntimeError("tts conversion failed (is ffmpeg on PATH?)")
        return wav

    async def _send_audio(self, p: Path) -> None:
        try:
            data = p.read_bytes()
            await self.send("audio", data=base64.b64encode(data).decode(),
                            format=p.suffix.lstrip("."),
                            seq=int(time.time() * 1000))
        except Exception as exc:  # noqa: BLE001
            log.warning("send audio failed: %s", exc)


async def handler(ws) -> None:
    session: AssistantSession | None = None
    log.info("dashboard client connected")
    _CLIENTS.add(ws)
    try:
        async for raw in ws:
            msg = json.loads(raw)
            if session is None:
                if msg.get("kind") == "hello":
                    name = msg.get("provider", "voicechat")
                    session = AssistantSession(ws)
                    await session.send("status", text="loading model...")
                    await _get_shared_provider(name)
                    await session.send("ready", provider=name)
                continue
            await session.handle(msg)
    except Exception as exc:  # noqa: BLE001
        log.warning("ws error: %s", exc)
    finally:
        _CLIENTS.discard(ws)
        # NOTE: the shared provider is intentionally NOT stopped on
        # disconnect — it is reused by the next dashboard connection.
        # It is stopped only on provider switch (_get_shared_provider)
        # or server shutdown.
        log.info("dashboard client disconnected")


async def _shutdown_global() -> None:
    """Full cleanup: stop the shared model, sweep orphans. Idempotent."""
    global _active_provider, _active_name
    prov = _active_provider
    _active_provider = None
    _active_name = None
    if prov is not None and hasattr(prov, "stop"):
        log.info("shutdown: stopping model provider...")
        try:
            await asyncio.get_running_loop().run_in_executor(None, prov.stop)
        except Exception as exc:  # noqa: BLE001
            log.warning("shutdown: provider stop failed: %s", exc)
    # belt-and-braces: kill any orphaned model processes
    try:
        subprocess.run(["taskkill", "/F", "/IM", "llama-voicechat.exe"],
                       capture_output=True, timeout=10)
    except Exception:  # noqa: BLE001
        pass
    log.info("shutdown: clean — models stopped, RAM/VRAM released")


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def release_port(port: int) -> None:
    """Free a port held by a stale process: find the PID and kill it."""
    try:
        out = subprocess.run(
            ["netstat", "-ano"], capture_output=True, text=True,
            timeout=15).stdout
    except Exception as exc:  # noqa: BLE001
        log.warning("release_port(%d): netstat failed: %s", port, exc)
        return
    pids = set()
    for line in out.splitlines():
        if f":{port}" in line and "LISTENING" in line.upper():
            parts = line.split()
            if parts:
                pids.add(parts[-1])
    for pid in pids:
        if pid == str(os.getpid()):
            continue
        log.info("release_port(%d): killing stale PID %s", port, pid)
        subprocess.run(["taskkill", "/F", "/PID", pid],
                       capture_output=True, timeout=10)
    if pids:
        time.sleep(0.5)  # let the OS release the socket


async def main() -> None:
    import signal
    import websockets
    global _MAIN_LOOP
    logging.basicConfig(level=logging.INFO)
    cfg = config.get_config()

    LOG_DIR.mkdir(exist_ok=True)

    def process_request(connection, request):
        """Serve the dashboard over HTTP on the same port as the WS."""
        path = request.path
        if path == "/ws" or path.startswith("/ws?"):
            return None
        from urllib.parse import unquote
        rel = unquote(path.lstrip("/")) if path not in ("/", "") else "index.html"
        fpath = (STATIC / rel).resolve()
        # path-traversal guard: only serve files inside static/
        if not str(fpath).startswith(str(STATIC.resolve())) or not fpath.is_file():
            fpath = STATIC / "index.html"
        body = fpath.read_bytes()
        ctype = ("text/html" if fpath.suffix == ".html"
                 else "application/javascript" if fpath.suffix == ".js"
                 else "text/css" if fpath.suffix == ".css"
                 else "image/png" if fpath.suffix == ".png"
                 else "application/octet-stream")
        from websockets.http11 import Response
        from websockets.datastructures import Headers
        return Response(200, "OK",
                        headers=Headers([("Content-Type", ctype),
                                         ("Content-Length", str(len(body)))]),
                        body=body)

    # If a stale instance holds our port (e.g. previous crash), free it.
    if _port_in_use(cfg.dashboard_port):
        log.warning("port %d in use — releasing stale instance first",
                    cfg.dashboard_port)
        release_port(cfg.dashboard_port)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    _MAIN_LOOP = loop  # worker threads use this to emit tool events

    # reminder scheduler → proactive speech via a lightweight session shim
    from .reminders import ReminderStore, start_scheduler
    rem_store = ReminderStore()

    async def _deliver(item):
        # synthesize + broadcast to all connected dashboards
        text = f"Reminder: {item['text']}"
        log.info("delivering reminder: %s", item["text"][:60])
        try:
            import edge_tts, tempfile
            mp3 = Path(tempfile.gettempdir()) / "pai_rem.mp3"
            await edge_tts.Communicate(text).save(str(mp3))
            data = base64.b64encode(mp3.read_bytes()).decode()
        except Exception as exc:  # noqa: BLE001
            log.warning("reminder TTS failed: %s", exc)
            data = None
        # broadcast to every connected dashboard (tracked globally)
        for cli in list(_CLIENTS):
            try:
                await cli.send("reply", text=f"⏰ {text}")
                if data:
                    await cli.send("audio", data=data, format="mp3",
                                   seq=int(time.time() * 1000))
            except Exception:  # noqa: BLE001
                pass

    def _on_due(item):
        if _MAIN_LOOP and _MAIN_LOOP.is_running():
            asyncio.run_coroutine_threadsafe(_deliver(item), _MAIN_LOOP)

    _SCHEDULER["stop"] = start_scheduler(rem_store, _on_due)
    log.info("reminder scheduler running (%d pending)",
             len(rem_store.list_pending()))

    sigs = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGBREAK"):
        sigs.append(signal.SIGBREAK)  # Windows Ctrl+Break / CTRL_BREAK_EVENT
    import threading as _threading
    if _threading.current_thread() is not _threading.main_thread():
        log.info("running in a worker thread — skipping signal handlers "
                 "(shutdown via process exit)")
    else:
        for sig in sigs:
            try:
                loop.add_signal_handler(sig, stop.set)
            except (NotImplementedError, RuntimeError):
                # Windows: loop.add_signal_handler unsupported — use signal.signal
                try:
                    signal.signal(sig, lambda *_: stop.set())
                except (ValueError, OSError):
                    log.warning("could not install handler for %s", sig)

    async with websockets.serve(handler, "127.0.0.1", cfg.dashboard_port,
                                process_request=process_request,
                                max_size=32 * 1024 * 1024):
        log.info("dashboard on http://127.0.0.1:%d (ws on same port)",
                 cfg.dashboard_port)
        await stop.wait()          # interrupted (Ctrl+C / SIGTERM / close)
    # leaving the `async with` closes the listener and frees the port
    await _shutdown_global()       # stop model, free RAM/VRAM, sweep orphans


if __name__ == "__main__":
    asyncio.run(main())

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
import time
from pathlib import Path

from . import config, input_control, providers, tools

log = logging.getLogger("pai.server")

STATIC = Path(__file__).resolve().parents[2] / "static"

# Single shared provider across dashboard connections — prevents model
# processes (and VRAM) piling up on browser refreshes/reconnects.
_active_provider = None
_active_name: str | None = None
_provider_lock = asyncio.Lock()


async def _get_shared_provider(name: str):
    """Return the shared provider instance, switching cleanly if needed.

    Guarantees the OLD model process is fully stopped (RAM/VRAM freed)
    before the new one loads.
    """
    global _active_provider, _active_name
    async with _provider_lock:
        if _active_provider is not None and _active_name == name:
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

    def __init__(self, ws, provider_name: str):
        self.ws = ws
        self.cfg = config.get_config()
        self.cfg.provider = provider_name
        self.provider = providers.get_provider(provider_name)
        self.executor = tools.ToolExecutor(
            autonomy=self.cfg.autonomy,
            on_screenshot=self._push_screenshot)
        self.history: list[dict] = []
        self.latest_png: bytes | None = None

    def _push_screenshot(self, png: bytes) -> None:
        self.latest_png = png

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
        if kind == "config":
            await self.send("config_ok", provider=self.provider.name,
                            autonomy=self.cfg.autonomy)
        elif kind == "switch":
            # switch the shared provider without dropping the connection
            name = msg.get("provider", "voicechat")
            await self.send("status", text=f"switching to {name}...")
            self.provider = await _get_shared_provider(name)
            self.cfg.provider = name
            await self.send("ready", provider=self.provider.name)
        elif kind == "audio_chunk":
            # browser sends base64 wav/webm blob of the utterance
            data = base64.b64decode(msg.get("data", ""))
            wav = self._to_wav(data)
            await self._turn(wav)
        elif kind == "text":
            await self._turn_text(msg.get("text", ""))
        elif kind == "screenshot":
            entry = self.executor.execute("screenshot", {})
            if self.latest_png:
                await self.send("screenshot",
                                image=base64.b64encode(self.latest_png).decode())
        elif kind == "kill":
            ks = input_control.get_kill_switch()
            ks._trigger()
            await self.send("kill_state", engaged=ks.engaged)

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
        import subprocess
        out = p.with_suffix(".wav")
        r = subprocess.run(["ffmpeg", "-y", "-i", str(p), "-ar", "16000",
                        "-ac", "1", str(out)],
                       capture_output=True, timeout=60)
        if r.returncode != 0 or not out.exists():
            log.error("ffmpeg convert failed: %s", r.stderr.decode()[-400:])
            raise RuntimeError("audio conversion failed (is ffmpeg on PATH?)")
        log.info("converted %s -> wav (%d bytes)", p.suffix, out.stat().st_size)
        return out

    async def _turn(self, wav: Path) -> None:
        if self.cfg.provider == "modular":
            loop = asyncio.get_running_loop()
            transcript = await loop.run_in_executor(
                None, self.provider.transcribe, wav)
            await self.send("transcript", text=transcript)
            result = await loop.run_in_executor(
                None, self.provider.think, transcript, self.executor)
            await self.send("reply", text=result["text"])
            out = Path("logs/reply.wav")
            out.parent.mkdir(exist_ok=True)
            if result["text"]:
                spoken = await loop.run_in_executor(
                    None, self.provider.speak, result["text"], out)
                await self._send_audio(spoken)
        else:
            loop = asyncio.get_running_loop()
            log.info("turn: provider=%s wav=%s (%d bytes) — processing",
                     self.cfg.provider, wav.name, wav.stat().st_size)
            await self.send("status", text="thinking (model is generating)...")
            t0 = time.time()
            result = await loop.run_in_executor(
                None, self.provider.turn, wav, self.executor)
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
                await self._send_audio(Path(audio))

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
        if self.cfg.provider == "modular":
            await self._turn_modular_text(text)
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
        import subprocess
        import tempfile
        import asyncio

        async def _synth():
            import edge_tts
            tmp = Path(tempfile.gettempdir()) / "pai_text_input.mp3"
            await edge_tts.Communicate(text, "en-US-AriaNeural").save(str(tmp))
            return tmp

        mp3 = asyncio.run(_synth())
        wav = mp3.with_suffix(".wav")
        subprocess.run(["ffmpeg", "-y", "-i", str(mp3), "-ar", "16000",
                        "-ac", "1", str(wav)],
                       capture_output=True, check=True)
        return wav

    async def _turn_modular_text(self, text: str) -> None:
        loop = asyncio.get_running_loop()
        await self.send("transcript", text=text)
        result = await loop.run_in_executor(
            None, self.provider.think, text, self.executor)
        await self.send("reply", text=result["text"])
        out = Path("logs/reply.wav")
        out.parent.mkdir(exist_ok=True)
        if result["text"]:
            spoken = await loop.run_in_executor(
                None, self.provider.speak, result["text"], out)
            await self._send_audio(spoken)

    async def _send_audio(self, p: Path) -> None:
        try:
            data = p.read_bytes()
            await self.send("audio", data=base64.b64encode(data).decode(),
                            format=p.suffix.lstrip("."))
        except Exception as exc:  # noqa: BLE001
            log.warning("send audio failed: %s", exc)


async def handler(ws) -> None:
    session: AssistantSession | None = None
    log.info("dashboard client connected")
    try:
        async for raw in ws:
            msg = json.loads(raw)
            if session is None:
                if msg.get("kind") == "hello":
                    name = msg.get("provider", "voicechat")
                    session = AssistantSession(ws, name)
                    await session.send("status", text="loading model...")
                    session.provider = await _get_shared_provider(name)
                    await session.send("ready",
                                       provider=session.provider.name)
                continue
            await session.handle(msg)
    except Exception as exc:  # noqa: BLE001
        log.warning("ws error: %s", exc)
    finally:
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
    logging.basicConfig(level=logging.INFO)
    cfg = config.get_config()

    def process_request(connection, request):
        """Serve the dashboard over HTTP on the same port as the WS."""
        import http.server
        path = request.path
        if path == "/ws" or path.startswith("/ws?"):
            return None
        fpath = STATIC / ("index.html" if path in ("/", "") else path.lstrip("/"))
        if not fpath.is_file():
            fpath = STATIC / "index.html"
        body = fpath.read_bytes()
        ctype = ("text/html" if fpath.suffix == ".html"
                 else "application/javascript" if fpath.suffix == ".js"
                 else "text/css" if fpath.suffix == ".css"
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
    sigs = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGBREAK"):
        sigs.append(signal.SIGBREAK)  # Windows Ctrl+Break / CTRL_BREAK_EVENT
    for sig in sigs:
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError):
            # Windows: loop.add_signal_handler unsupported — use signal.signal
            signal.signal(sig, lambda *_: stop.set())

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

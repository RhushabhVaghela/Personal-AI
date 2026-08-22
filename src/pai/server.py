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
import time
from pathlib import Path

from . import config, input_control, providers, tools

log = logging.getLogger("pai.server")

STATIC = Path(__file__).resolve().parents[2] / "static"


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
        if kind == "config":
            await self.send("config_ok", provider=self.provider.name,
                            autonomy=self.cfg.autonomy)
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
        suffix = ".wav"
        if blob[:4] == b"RIFF":
            suffix = ".wav"
        elif blob[:3] == b"IDC" or b"webm" in blob[:64]:
            suffix = ".webm"
        p = Path(tempfile.gettempdir()) / f"pai_chunk{suffix}"
        p.write_bytes(blob)
        if suffix == ".webm":
            p = self._ffmpeg_to_wav(p)
        return p

    def _ffmpeg_to_wav(self, p: Path) -> Path:
        import subprocess
        out = p.with_suffix(".wav")
        subprocess.run(["ffmpeg", "-y", "-i", str(p), "-ar", "16000",
                        "-ac", "1", str(out)],
                       capture_output=True, check=False)
        return out if out.exists() else p

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
            result = await loop.run_in_executor(
                None, self.provider.turn, wav, self.executor)
            await self.send("reply", text=result.get("text", ""))
            if result.get("audio"):
                await self._send_audio(Path(result["audio"]))

    async def _turn_text(self, text: str) -> None:
        if self.cfg.provider == "modular":
            await self._turn_modular_text(text)
        else:
            await self.send("reply",
                            text="(text input needs the modular provider)")

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
                    if hasattr(session.provider, "start"):
                        await session.send("status", text="loading model...")
                        loop = asyncio.get_running_loop()
                        await loop.run_in_executor(
                            None, session.provider.start)
                    await session.send("ready",
                                       provider=session.provider.name)
                continue
            await session.handle(msg)
    except Exception as exc:  # noqa: BLE001
        log.warning("ws error: %s", exc)
    finally:
        if session and hasattr(session.provider, "stop"):
            try:
                session.provider.stop()
            except Exception:  # noqa: BLE001
                pass
        log.info("dashboard client disconnected")


async def main() -> None:
    import websockets
    from websockets.legacy.server import serve as legacy_serve  # type: ignore
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

    async with websockets.serve(handler, "127.0.0.1", cfg.dashboard_port,
                                process_request=process_request,
                                max_size=32 * 1024 * 1024):
        log.info("dashboard on http://127.0.0.1:%d (ws on same port)",
                 cfg.dashboard_port)
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())

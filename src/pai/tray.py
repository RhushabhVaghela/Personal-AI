"""System-tray app — background continuation (ChatGPT desktop parity).

Runs the dashboard server in a background thread and puts an icon in the
Windows system tray:
  • left-click / Open  → launch the dashboard in the browser
  • Hands-free toggle  → starts/stops VAD listening (works headless —
    replies play through speakers, no browser needed)
  • Quit               → graceful shutdown (models stopped, ports freed)
  • kill-switch        → global hotkey Ctrl+Alt+Q works regardless

Requires: pip install pystray pillow  (in requirements.txt)
"""
from __future__ import annotations

import asyncio
import logging
import threading
import webbrowser

log = logging.getLogger("pai.tray")


def run_with_tray() -> None:
    try:
        import pystray
        from PIL import Image, ImageDraw
    except ImportError:
        log.error("tray: pystray/Pillow missing — running headless instead")
        from .server import main as server_main
        asyncio.run(server_main())
        return

    state = {"hands_free": False, "killed": False, "loop": None}

    def make_icon():
        # simple orb with a mic glyph; color reflects assistant state
        img = Image.new("RGB", (64, 64), (15, 17, 23))
        d = ImageDraw.Draw(img)
        if state["killed"]:
            color = (228, 72, 77)      # red — killed
        elif state["hands_free"]:
            color = (76, 195, 138)     # green — listening
        else:
            color = (124, 108, 240)    # purple — idle
        d.ellipse([8, 8, 56, 56], fill=color)
        d.rectangle([28, 18, 36, 34], fill=(15, 17, 23))
        d.arc([22, 26, 42, 44], 0, 180, fill=(15, 17, 23), width=3)
        d.line([32, 44, 32, 50], fill=(15, 17, 23), width=3)
        return img

    icon = pystray.Icon("pai", make_icon(), "PersonalAI Assistant")

    def refresh():
        icon.icon = make_icon()

    def on_open(_, item):
        import webbrowser as _wb
        from . import config as _config
        port = _config.get_config().dashboard_port
        _wb.open(f"http://127.0.0.1:{port}")

    def _handle_utterance_speakers(wav_path):
        """Headless utterance handling: reply through the speakers."""
        from . import server as srv
        prov = srv._active_provider
        if prov is None:
            log.info("tray: no provider loaded — say hello in the dashboard "
                     "once to load it")
            return
        if getattr(prov, "name", "") == "modular":
            text = prov.transcribe(wav_path)
            result = prov.think(text, None)   # think() tolerates executor=None
            if result.get("text"):
                out = srv.LOG_DIR / "tray_reply.wav"
                audio_p = prov.speak(result["text"], out)
                from .audio import play_any
                play_any(audio_p)
        else:
            # voice-model providers need a ToolExecutor for their tool loop;
            # build one with current autonomy instead of passing None
            from .tools import ToolExecutor
            cfg = srv.config.get_config()
            result = prov.turn(
                wav_path, ToolExecutor(autonomy=cfg.autonomy))
            if result.get("audio"):
                from .audio import play_any
                from pathlib import Path
                play_any(Path(result["audio"]))

    async def _toggle_hands_free_remote() -> bool:
        """Start/stop the shared hands-free loop without a browser session.

        Mirrors AssistantSession._toggle_hands_free but standalone.
        """
        from . import server as srv
        from .vad import get_engine, listen_continuous
        hf = srv._HANDSFREE

        if hf["thread"] is None:                       # → turn ON
            cfg = srv.config.get_config()
            engine = get_engine(
                prefer_webrtc=cfg.vad_engine in ("auto", "webrtc"),
                sample_rate=cfg.sample_rate,
                silence_hangover_ms=cfg.vad_silence_ms)
            stop = threading.Event()

            def on_utterance(wav_path):
                engine.set_speaking(True)   # echo-guard during reply
                try:
                    _handle_utterance_speakers(wav_path)
                finally:
                    engine.set_speaking(False)

            hf.update(engine=engine, stop=stop)
            hf["thread"] = threading.Thread(
                target=lambda: listen_continuous(
                    engine, on_utterance, stop_flag=stop), daemon=True)
            hf["thread"].start()
            log.info("tray: hands-free ON")
            return True

        # → turn OFF
        if hf["stop"]:
            hf["stop"].set()
        if hf["thread"]:
            hf["thread"].join(timeout=3)
        hf.update(thread=None, stop=None, engine=None, session=None)
        log.info("tray: hands-free OFF")
        return False

    def on_hands_free(_, item):
        loop = state.get("loop")
        if not loop or not loop.is_running():
            log.warning("tray: server loop not ready yet")
            return

        future = asyncio.run_coroutine_threadsafe(
            _toggle_hands_free_remote(), loop)

        def _done(fut):
            try:
                state["hands_free"] = bool(fut.result())
            except Exception as exc:              # noqa: BLE001
                log.error("tray toggle failed: %s", exc)
            refresh()

        future.add_done_callback(_done)

    def on_quit(_, item):
        log.info("tray quit requested")
        icon.stop()
        import os
        import signal
        # graceful: signal our own process; handlers stop models + free port
        threading.Thread(target=lambda: os.kill(
            os.getpid(), signal.SIGINT), daemon=True).start()

    icon.menu = pystray.Menu(
        pystray.MenuItem("Open Dashboard", on_open, default=True),
        pystray.MenuItem(
            lambda _: "👂 Hands-free: ON" if state["hands_free"]
            else "👂 Hands-free: OFF", on_hands_free),
        pystray.MenuItem("🛑 Kill-switch is global hotkey Ctrl+Alt+Q",
                         None, enabled=False),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit (stops models)", on_quit))

    def server_thread():
        from .server import main as server_main
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            state["loop"] = loop          # tray actions target this loop
            asyncio.run(server_main())
        except Exception as exc:  # noqa: BLE001
            log.error("server exited: %s", exc)

    t = threading.Thread(target=server_thread, daemon=True)
    t.start()
    log.info("tray icon up")
    icon.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_with_tray()

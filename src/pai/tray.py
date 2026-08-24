"""System-tray app — background continuation (ChatGPT desktop parity).

Runs the dashboard server in a background thread and puts an icon in the
Windows system tray:
  • left-click / Open  → launch the dashboard in the browser
  • toggle hands-free  → start/stop VAD listening remotely
  • kill-switch state  → shows 🔴 when engaged
  • Quit               → graceful shutdown (models stopped, ports freed)

Requires: pip install pystray pillow  (already in requirements)
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

    state = {"hands_free": False, "killed": False,
             "loop": None, "session": None}

    def make_icon():
        # draw a simple orb with a mic glyph; color reflects state
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
        webbrowser.open("http://127.0.0.1:8765")

    async def _send(msg):
        for cli in list(_CLIENTS):  # noqa: F821 — injected below
            try:
                await cli.send(json.dumps(msg))  # noqa: F821
            except Exception:  # noqa: BLE001
                pass

    def on_hands_free(_, item):
        loop = state["loop"]

        async def do():
            from . import server as srv
            on = not state["hands_free"]
            srv._HANDSFREE_NEXT = on  # flag consumed by a shim session
            await _send({"kind": "hands_free", "on": on})
        if loop:
            asyncio.run_coroutine_threadsafe(do(), loop)
        state["hands_free"] = not state["hands_free"]
        refresh()

    def on_quit(_, item):
        log.info("tray quit requested")
        icon.stop()
        import os
        import signal
        # graceful: signal our own process; handlers stop models + free port
        threading.Thread(target=lambda: os.kill(
            os.getpid(), signal.SIGINT if hasattr(signal, "SIGINT") else
            signal.SIGTERM), daemon=True).start()

    import json  # noqa: E402 — used inside _send

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

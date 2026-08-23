"""Integration test: boot dashboard, fetch page, exercise WS protocol."""
import asyncio
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import websockets

from pai import server as pai_server

threading.Thread(target=lambda: asyncio.run(pai_server.main()),
                 daemon=True).start()
time.sleep(2)

fail = 0

html = urllib.request.urlopen("http://127.0.0.1:8765/").read()
print("HTTP page:", "OK" if b"PersonalAI" in html else "FAIL", len(html), "bytes")
fail += 0 if b"PersonalAI" in html else 1


async def recv_until(ws, kinds, timeout=30):
    """Receive messages until one of `kinds` arrives; return it."""
    end = time.time() + timeout
    while time.time() < end:
        m = json.loads(await asyncio.wait_for(ws.recv(), timeout=end - time.time()))
        yield m
        if m.get("kind") in kinds:
            return


async def ws_test():
    global fail
    async with websockets.connect("ws://127.0.0.1:8765/ws",
                                  max_size=32 * 1024 * 1024) as ws:
        await ws.send(json.dumps({"kind": "hello", "provider": "modular"}))
        # drain status/loading messages until ready
        ready = None
        async for m in recv_until(ws, ("ready",)):
            print("  <-", m.get("kind"), str(m)[:60])
            ready = m
        print("WS hello reply:", ready)
        if not ready or ready.get("kind") != "ready":
            fail += 1

        await ws.send(json.dumps({"kind": "screenshot"}))
        shot = None
        async for m in recv_until(ws, ("screenshot", "error")):
            print("  <-", m.get("kind"))
            shot = m
        ok = shot and shot.get("kind") == "screenshot" \
            and len(shot.get("image", "")) > 1000
        print("WS screenshot:", "OK" if ok else "FAIL",
              len(shot.get("image", "")) if shot else 0, "b64 chars")
        fail += 0 if ok else 1

asyncio.run(ws_test())
print("INTEGRATION:", "PASS" if fail == 0 else f"FAIL({fail})")
sys.exit(1 if fail else 0)

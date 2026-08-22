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


async def ws_test():
    global fail
    async with websockets.connect("ws://127.0.0.1:8765/ws",
                                  max_size=32 * 1024 * 1024) as ws:
        await ws.send(json.dumps({"kind": "hello", "provider": "modular"}))
        # modular provider: ready without model load (lazy HTTP calls)
        m = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        print("WS hello reply:", m)
        if m.get("kind") != "ready":
            fail += 1
        await ws.send(json.dumps({"kind": "screenshot"}))
        m = json.loads(await asyncio.wait_for(ws.recv(), timeout=20))
        print("WS screenshot:", "OK" if m.get("kind") == "screenshot"
              and len(m.get("image", "")) > 1000 else "FAIL",
              len(m.get("image", "")), "b64 chars")
        fail += 0 if m.get("kind") == "screenshot" else 1

asyncio.run(ws_test())
print("INTEGRATION:", "PASS" if fail == 0 else f"FAIL({fail})")
sys.exit(1 if fail else 0)

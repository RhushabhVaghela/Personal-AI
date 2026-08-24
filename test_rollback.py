"""Verify switch-failure rollback: failed switch restores previous provider."""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


class FakeWS:
    def __init__(self):
        self.sent = []

    async def send(self, raw):
        self.sent.append(json.loads(raw))


async def main():
    import pai.server as srv

    ws = FakeWS()
    s = srv.AssistantSession(ws)
    # start on modular (loads instantly — no model process)
    await srv._get_shared_provider("modular")
    assert srv._active_name == "modular"

    # try to switch to moshi (backend not built -> pre-flight fails)
    await s._handle({"kind": "switch", "provider": "moshi"}, "switch")

    err = [m for m in ws.sent if m["kind"] == "error"]
    rdy = [m for m in ws.sent if m["kind"] == "ready" and m.get("restored")]
    assert err and "unavailable" in err[0]["text"], ws.sent[-3:]
    assert rdy and rdy[-1]["provider"] == "modular", ws.sent[-3:]
    assert srv._active_name == "modular", srv._active_name
    print("ROLLBACK PASS: error shown, ready(restored=modular), "
          "active provider intact")

    # caps message present after successful hello-switch path is covered
    # by integration tests; here we assert the error path leaves state clean.
    print("SWITCH ROLLBACK TEST PASS")


asyncio.run(main())

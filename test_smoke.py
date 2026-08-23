"""Smoke tests for PersonalAI-Assistant core (no model load)."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
ROOT = Path(__file__).parent

ok = fail = 0
def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1; print(f"  PASS {name}")
    else:
        fail += 1; print(f"  FAIL {name} {detail}")

print("== imports ==")
from pai import config, audio, input_control, providers, tools
from pai.screen_capture import get_capture
check("config loads", config.get_config().provider in providers.PROVIDERS)
check("assets exist: exe", config.VOICECHAT_EXE.exists())
check("assets exist: main gguf", config.GGUF_MAIN.exists())
check("assets exist: mmproj", config.GGUF_MMPROJ.exists())
check("assets exist: tts gguf", config.GGUF_TTS.exists())
check("assets exist: funchead gguf", config.GGUF_FUNCHEAD.exists())

print("== screen capture ==")
cap = get_capture()
png = cap.capture_png()
check("capture returns PNG", png[:8] == b"\x89PNG\r\n\x1a\n", f"len={len(png)}")
check("capture cached", cap.capture_png() is png)
check("stats tracked", cap.stats.n_captures >= 1)

print("== resize (max_width) ==")
from PIL import Image
import io as _io
w = Image.open(_io.BytesIO(png)).width
small = cap.capture_png(max_width=800)
sw = Image.open(_io.BytesIO(small)).width
check(f"max_width respected ({w}->{sw})", sw == min(800, w))
check("last_frame public api", cap.last_frame() is not None)

print("== tool parser ==")
calls = tools.parse_tool_calls('```json\n{"tool": "click", "params": {"x": 10, "y": 20}}\n```')
check("fenced json", calls == [("click", {"x": 10, "y": 20})], str(calls))
calls = tools.parse_tool_calls('blah {"tool":"scroll","params":{"amount":-3}} blah')
check("bare json", calls == [("scroll", {"amount": -3})], str(calls))
calls = tools.parse_tool_calls('<tool_call>press_key</tool_call><params>{"key":"enter"}</params>')
check("xml style", calls == [("press_key", {"key": "enter"})], str(calls))

print("== autonomy enforcement ==")
ex = tools.ToolExecutor(autonomy="auto_safe")
r = ex.execute("run_command", {"command": "echo hi"})
check("run_command blocked at auto_safe", r.get("blocked") is True)
r = ex.execute("screenshot", {})
check("screenshot allowed at auto_safe", r["ok"])
ex2 = tools.ToolExecutor(autonomy="confirm")
r = ex2.execute("click", {"x": 5, "y": 5})
check("click blocked at confirm", r.get("blocked") is True)
r = ex2.execute("screenshot", {})
check("screenshot allowed at confirm", r["ok"])

print("== kill-switch gating ==")
ks = input_control.get_kill_switch()
inp = input_control.InputController(ks)
engaged_before = ks.engaged
ks.toggle() if not engaged_before else None
check("toggle() works", ks.engaged is not engaged_before)
check("click refused when engaged", inp.click(5, 5) is False)
check("type refused when engaged", inp.type_text("x") is False)
check("refused counter", inp.stats.refused >= 2)
if ks.engaged != engaged_before:
    ks.toggle()

print("== hotkey normalization ==")
norm = input_control.normalize_hotkey("ctrl+alt+q")
check("hotkey normalized", norm == "<ctrl>+<alt>+q", norm)
check("listener started", input_control.get_kill_switch()._listener is not None)

print("== key table ==")
inp.press_key  # exists
if input_control.HAS_PYNPUT:
    check("f-keys in table", "f5" in input_control._KEY_ALIASES)
    check("pgup alias", "pgup" in input_control._KEY_ALIASES)

print("== provider registry ==")
for name in providers.PROVIDERS:
    p = providers.get_provider(name)
    check(f"provider {name} instantiates", p.name == name)

print("== hybrid has stop ==")
h = providers.get_provider("hybrid")
check("hybrid.stop exists", hasattr(h, "stop"))
check("hybrid.vc is voicechat", h.vc.name == "voicechat")

print("== server module parses ==")
import pai.server  # noqa: F401
import pai.terminal  # noqa: F401
check("server/terminal import", True)

print(f"\nRESULT: {ok} passed, {fail} failed")
sys.exit(1 if fail else 0)

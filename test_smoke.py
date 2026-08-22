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

print("== tool parser ==")
calls = tools.parse_tool_calls('```json\n{"tool": "click", "params": {"x": 10, "y": 20}}\n```')
check("fenced json", calls == [("click", {"x": 10, "y": 20})], str(calls))
calls = tools.parse_tool_calls('blah {"tool":"scroll","params":{"amount":-3}} blah')
check("bare json", calls == [("scroll", {"amount": -3})], str(calls))
calls = tools.parse_tool_calls('<tool_call>press_key</tool_call><params>{"key":"enter"}</params>')
check("xml style", calls == [("press_key", {"key": "enter"})], str(calls))

print("== tool executor (dry) ==")
ex = tools.ToolExecutor(autonomy="auto_safe")
r = ex.execute("screenshot", {})
check("screenshot tool", r["ok"] and r["result"]["bytes"] > 1000)
r = ex.execute("run_command", {"command": "echo hi"})
check("run_command blocked at auto_safe", not r["ok"] or "blocked" in str(r["result"]))
ex2 = tools.ToolExecutor(autonomy="confirm")
r = ex2.execute("run_command", {"command": "echo hello"})
check("run_command blocked at confirm", "blocked" in str(r["result"]))

print("== kill-switch gating ==")
ks = input_control.get_kill_switch()
inp = input_control.InputController(ks)
ks._engaged = True
check("click refused when engaged", inp.click(5, 5) is False)
check("type refused when engaged", inp.type_text("x") is False)
check("refused counter", inp.stats.refused >= 2)
ks._engaged = False

print("== provider registry ==")
for name in providers.PROVIDERS:
    p = providers.get_provider(name)
    check(f"provider {name} instantiates", p.name == name)

print("== server module parses ==")
import pai.server  # noqa: F401
import pai.terminal  # noqa: F401
check("server/terminal import", True)

print(f"\nRESULT: {ok} passed, {fail} failed")
sys.exit(1 if fail else 0)

"""Tool bridge: JSON tool-call protocol → real actions on the PC.

Tools exposed to the model (any provider):
  screenshot      — capture screen, return image (fed to vision) or summary
  click           — {x, y, button?, double?}
  drag            — {x1, y1, x2, y2}
  scroll          — {amount, x?, y?}
  type_text       — {text}
  press_key       — {key} e.g. "ctrl+s"
  open_app        — {name_or_path}
  run_command     — {command} (shell; gated by autonomy level)

Autonomy levels:
  confirm   — every action blocked except screenshot (assistant proposes,
              human acts or explicitly approves)
  auto_safe — input actions allowed, run_command blocked
  full      — everything allowed

Every execution is logged and broadcast via the on_event callback so the
dashboard can show live tool activity.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from . import config, input_control, screen_capture

log = logging.getLogger("pai.tools")


def tool_schema() -> list[dict]:
    return [
        {"name": "screenshot", "description": "Capture the current screen (returns image for vision).",
         "params": {"monitor": "int, default 0"}},
        {"name": "look_at_screen", "description": "Inspect the latest shared screen frame (use when screen sharing is active and the user asks about what they see).",
         "params": {}},
        {"name": "set_reminder", "description": "Set a reminder that will be SPOKEN ALOUD at the due time. Use for 'remind me...'. Natural time text goes in 'when_text', e.g. 'at 5pm', 'in 10 minutes'.",
         "params": {"when_text": "str like 'at 5pm' or 'in 10 minutes'", "text": "str what to remind"}},
        {"name": "list_reminders", "description": "List pending reminders.",
         "params": {}},
        {"name": "cancel_reminder", "description": "Cancel a reminder by id (from list_reminders).",
         "params": {"id": "str"}},
        {"name": "get_time", "description": "Current local date and time. Also emits a clock card to the dashboard.",
         "params": {}},
        {"name": "get_weather", "description": "Weather via wttr.in (no API key). Emits a weather card.",
         "params": {"location": "str, e.g. 'Mumbai' or '' for auto-IP"}},
        {"name": "click", "description": "Click the mouse.",
         "params": {"x": "int", "y": "int", "button": "left|right|middle", "double": "bool"}},
        {"name": "drag", "description": "Drag from (x1,y1) to (x2,y2).",
         "params": {"x1": "int", "y1": "int", "x2": "int", "y2": "int"}},
        {"name": "scroll", "description": "Scroll (positive=up).",
         "params": {"amount": "int", "x": "int?", "y": "int?"}},
        {"name": "type_text", "description": "Type a string of text.",
         "params": {"text": "str"}},
        {"name": "press_key", "description": "Press a key or combo, e.g. 'enter', 'ctrl+s'.",
         "params": {"key": "str"}},
        {"name": "open_app", "description": "Open an application by name or path.",
         "params": {"target": "str"}},
        {"name": "run_command", "description": "Run a shell command (blocked unless autonomy=full).",
         "params": {"command": "str"}},
    ]


class ToolExecutor:
    """Executes tool calls, records history, enforces autonomy + kill-switch."""

    # what each autonomy level allows
    SAFE_INFO_TOOLS = {"get_time", "get_weather"}
    LEVELS = {
        "confirm": {"screenshot", "look_at_screen", *SAFE_INFO_TOOLS},
        "auto_safe": {"screenshot", "look_at_screen", *SAFE_INFO_TOOLS,
                      "set_reminder", "list_reminders", "cancel_reminder",
                      "click", "drag", "scroll", "type_text", "press_key",
                      "open_app"},
        "full": {"screenshot", "look_at_screen", *SAFE_INFO_TOOLS,
                 "set_reminder", "list_reminders", "cancel_reminder",
                 "click", "drag", "scroll", "type_text", "press_key",
                 "open_app", "run_command"},
    }

    def __init__(self, autonomy: str = "full",
                 on_screenshot: Optional[Callable[[bytes], None]] = None,
                 on_event: Optional[Callable[[dict], None]] = None,
                 on_card: Optional[Callable[[str, dict], None]] = None):
        self.autonomy = autonomy if autonomy in self.LEVELS else "full"
        if autonomy not in self.LEVELS:
            log.warning("unknown autonomy %r, using 'full'", autonomy)
        self.history: list[dict] = []
        self._lock = threading.Lock()
        self.on_screenshot = on_screenshot
        self.on_event = on_event
        self.on_card = on_card          # answer cards → dashboard widgets
        self.cap = screen_capture.get_capture()
        self.inp = input_control.get_input()
        from .reminders import ReminderStore
        self.reminders = ReminderStore()

    @property
    def allowed(self) -> set[str]:
        return self.LEVELS[self.autonomy]

    def execute(self, name: str, params: dict) -> dict:
        t0 = time.perf_counter()
        entry = {"tool": name, "params": params, "ok": False, "result": None}
        if name not in self.allowed:
            entry["result"] = f"blocked: autonomy={self.autonomy} does not allow '{name}'"
            entry["blocked"] = True
            log.warning("tool %s %s", name, entry["result"])
        else:
            try:
                entry["result"] = self._dispatch(name, params)
                entry["ok"] = not isinstance(entry["result"], Exception)
            except Exception as exc:  # noqa: BLE001
                entry["result"] = f"error: {exc}"
        entry["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        with self._lock:
            self.history.append(entry)
        log.info("tool %s -> %s (%.0fms)", name, entry["result"], entry["ms"])
        if self.on_event:
            try:
                self.on_event(entry)
            except Exception:  # noqa: BLE001
                pass
        return entry

    def _emit_card(self, kind: str, data: dict) -> None:
        """Push an answer card to the dashboard (GPT-Live widgets parity)."""
        if self.on_card:
            try:
                self.on_card(kind, data)
            except Exception:  # noqa: BLE001
                pass

    def _dispatch(self, name: str, p: dict):
        if name in ("screenshot", "look_at_screen"):
            if name == "look_at_screen":
                from . import server as _srv  # late import (circular-safe)
                streamer = _srv._SHARE.get("streamer")
                if streamer and streamer.latest_png:
                    png = streamer.latest_png
                    if self.on_screenshot:
                        try:
                            self.on_screenshot(png)
                        except Exception:  # noqa: BLE001
                            pass
                    return {"ok": True, "source": "shared_stream",
                            "age_s": round(time.time() - getattr(
                                streamer, "_last_ts", time.time()), 1)}
                # fall through to a fresh capture when nothing is shared
            png = self.cap.capture_png(int(p.get("monitor", 0)),
                                       max_width=p.get("max_width"))
            if self.on_screenshot:
                try:
                    self.on_screenshot(png)
                except Exception:  # noqa: BLE001
                    pass
            return {"ok": True, "bytes": len(png),
                    "hash": self.cap.screenshot_hash()}
        if name == "set_reminder":
            when, cleaned = self.reminders.parse_when(
                str(p.get("when_text", "")))
            if when is None:
                return {"ok": False,
                        "error": "could not parse time — use 'at 5pm' "
                                 "or 'in 10 minutes'"}
            item = self.reminders.add(when, str(p.get("text", cleaned)))
            due = datetime.fromtimestamp(item["when"]).strftime("%H:%M")
            self._emit_card("reminder", {"text": p.get("text", cleaned),
                                         "due": due})
            return {"ok": True, "id": item["id"], "due": due}
        if name == "list_reminders":
            items = [{"id": i["id"],
                      "due": datetime.fromtimestamp(i["when"]).strftime(
                          "%a %H:%M"),
                      "text": i["text"]}
                     for i in self.reminders.list_pending()]
            return {"ok": True, "reminders": items}
        if name == "cancel_reminder":
            return {"ok": self.reminders.cancel(str(p.get("id", "")))}
        if name == "get_time":
            now = datetime.now()
            card = {"time": now.strftime("%H:%M"),
                    "date": now.strftime("%A, %d %B %Y")}
            self._emit_card("clock", card)
            return {"ok": True, **card}
        if name == "get_weather":
            import requests as _rq
            loc = str(p.get("location", "") or "")
            try:
                r = _rq.get(f"https://wttr.in/{loc}?format=j1",
                            timeout=15)
                cur = r.json()["current_condition"][0]
                card = {"location": loc or (r.json().get(
                            "nearest_area", [{}])[0].get(
                            "areaName", [{"value": "?"}])[0]["value"]),
                        "temp_c": cur["temp_C"], "feels_c": cur[
                            "FeelsLikeC"],
                        "desc": cur["weatherDesc"][0]["value"],
                        "humidity": cur["humidity"]}
                self._emit_card("weather", card)
                return {"ok": True, **card}
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "error": f"weather unavailable: {exc}"}
        if name == "click":
            ok = self.inp.click(p.get("x"), p.get("y"),
                                p.get("button", "left"),
                                2 if p.get("double") else 1)
            return {"ok": ok}
        if name == "drag":
            return {"ok": self.inp.drag(p["x1"], p["y1"], p["x2"], p["y2"])}
        if name == "scroll":
            return {"ok": self.inp.scroll(int(p["amount"]),
                                          p.get("x"), p.get("y"))}
        if name == "type_text":
            return {"ok": self.inp.type_text(p["text"])}
        if name == "press_key":
            return {"ok": self.inp.press_key(p["key"])}
        if name == "open_app":
            return self._open_app(p["target"])
        if name == "run_command":
            return self._run_command(p["command"])
        raise ValueError(f"unknown tool: {name}")

    def _open_app(self, target: str):
        known = {"notepad": "notepad.exe", "calc": "calc.exe",
                 "explorer": "explorer.exe", "terminal": "wt.exe",
                 "cmd": "cmd.exe", "paint": "mspaint.exe"}
        exe = known.get(str(target).lower(), target)
        # never let shell metacharacters through Popen(shell=True)
        if Path(exe).exists() or shutil.which(exe):
            subprocess.Popen([exe], shell=False,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return {"ok": True, "launched": exe}
        safe = "".join(ch for ch in exe
                       if ch.isalnum() or ch in " ._:\\/-")
        log.info("open_app: shell-start fallback for %r", safe)
        subprocess.Popen(["cmd", "/c", "start", "", safe], shell=False,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return {"ok": True, "launched": safe, "via": "shell-start"}

    def _run_command(self, command: str):
        r = subprocess.run(command, shell=True, capture_output=True,
                           text=True, timeout=60)
        return {"ok": r.returncode == 0, "code": r.returncode,
                "stdout": r.stdout[:2000], "stderr": r.stderr[:2000]}


def parse_tool_calls(text: str) -> list[tuple[str, dict]]:
    """Extract tool calls from model output. Supports:

    1. ```tool ... ``` / ```json fenced blocks containing {"tool": ..., "params": {...}}
    2. bare JSON {"tool": "...", "params": {...}} anywhere in the text
    3. <tool_call>name</tool_call><params>{...}</params> style
    """
    import re
    calls = []
    for m in re.finditer(r"```(?:tool|json)?\s*\n(.*?)```", text, re.S):
        _try_parse(m.group(1), calls)
    if not calls:
        dec = json.JSONDecoder()
        idx = 0
        while idx < len(text):
            start = text.find("{", idx)
            if start == -1:
                break
            try:
                obj, end = dec.raw_decode(text, start)
                idx = end
                if isinstance(obj, dict) and "tool" in obj:
                    calls.append((obj["tool"], obj.get("params", {})))
            except json.JSONDecodeError:
                idx = start + 1
    if not calls:
        for m in re.finditer(
                r"<tool_call>\s*(\w+)\s*</tool_call>\s*<params>(.*?)</params>",
                text, re.S):
            try:
                calls.append((m.group(1), json.loads(m.group(2))))
            except Exception:  # noqa: BLE001
                pass
    return calls


def _try_parse(s: str, out: list):
    try:
        obj = json.loads(s.strip())
    except Exception:  # noqa: BLE001
        return
    if isinstance(obj, dict) and "tool" in obj:
        out.append((obj["tool"], obj.get("params", {})))
    elif isinstance(obj, list):
        for o in obj:
            if isinstance(o, dict) and "tool" in o:
                out.append((o["tool"], o.get("params", {})))

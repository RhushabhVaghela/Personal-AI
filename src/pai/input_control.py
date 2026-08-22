"""Low-level mouse/keyboard control with a global kill-switch.

Kill-switch: a dedicated listener thread watches for the configured hotkey
(e.g. ctrl+alt+q). When triggered, ALL actions from this module are refused
and the agent loop is signalled to stop. Full-autonomy safety net.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

log = logging.getLogger("pai.input")

try:
    import pynput
    from pynput import keyboard, mouse
    HAS_PYNPUT = True
except ImportError:
    HAS_PYNPUT = False


@dataclass
class InputStats:
    actions: int = 0
    refused: int = 0


class KillSwitch:
    """Global hotkey kill-switch + pause gate."""

    def __init__(self, hotkey: str = "<ctrl>+<alt>+q"):
        self._engaged = False
        self._lock = threading.Lock()
        self._callbacks: list[Callable[[], None]] = []
        self._listener = None
        if HAS_PYNPUT:
            try:
                hk = keyboard.HotKey(
                    keyboard.HotKey.parse(hotkey), self._trigger
                )
                self._listener = keyboard.Listener(
                    on_press=self._proxy(hk, True),
                    on_release=self._proxy(hk, False),
                )
                self._listener.daemon = True
                self._listener.start()
            except Exception as exc:  # noqa: BLE001
                log.warning("kill-switch listener failed to start: %s", exc)

    @staticmethod
    def _proxy(hk: "keyboard.HotKey", press: bool):
        def handler(key):
            try:
                if press:
                    hk.press(key)
                else:
                    hk.release(key)
            except Exception:
                pass
        return handler

    def _trigger(self):
        with self._lock:
            self._engaged = not self._engaged  # toggle
            engaged = self._engaged
        log.warning("KILL-SWITCH toggled -> %s", "ENGAGED" if engaged else "released")
        for cb in list(self._callbacks):
            try:
                cb(engaged)
            except Exception:  # noqa: BLE001
                pass

    @property
    def engaged(self) -> bool:
        return self._engaged

    def on_toggle(self, cb: Callable[[bool], None]) -> None:
        self._callbacks.append(cb)

    def stop(self):
        if self._listener:
            self._listener.stop()


class InputController:
    """Mouse + keyboard control. Every action checks the kill-switch first."""

    def __init__(self, kill_switch: KillSwitch,
                 move_duration_ms: int = 150,
                 type_interval_sec: float = 0.02):
        self.ks = kill_switch
        self.move_duration_ms = move_duration_ms
        self.type_interval_sec = type_interval_sec
        self.stats = InputStats()

    def _gate(self) -> bool:
        if self.ks.engaged:
            self.stats.refused += 1
            log.warning("action refused: kill-switch engaged")
            return False
        return True

    # -- mouse ---------------------------------------------------------------

    def move(self, x: int, y: int, duration: bool = True):
        if not self._gate() or not HAS_PYNPUT:
            return False
        m = mouse.Controller()
        if duration and self.move_duration_ms > 0:
            cur = m.position
            steps = max(2, self.move_duration_ms // 10)
            for i in range(1, steps + 1):
                if self.ks.engaged:
                    return False
                t = i / steps
                m.position = (int(cur[0] + (x - cur[0]) * t),
                              int(cur[1] + (y - cur[1]) * t))
                time.sleep(self.move_duration_ms / 1000 / steps)
        else:
            m.position = (x, y)
        self.stats.actions += 1
        return True

    def click(self, x: Optional[int] = None, y: Optional[int] = None,
              button: str = "left", clicks: int = 1):
        if not self._gate() or not HAS_PYNPUT:
            return False
        m = mouse.Controller()
        if x is not None and y is not None:
            self.move(x, y, duration=False)
        btn = mouse.Button.left if button == "left" else (
            mouse.Button.right if button == "right" else mouse.Button.middle)
        m.click(btn, clicks)
        self.stats.actions += 1
        return True

    def double_click(self, x=None, y=None):
        return self.click(x, y, clicks=2)

    def drag(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 400):
        if not self._gate() or not HAS_PYNPUT:
            return False
        m = mouse.Controller()
        m.position = (x1, y1)
        time.sleep(0.05)
        with m.pressed(mouse.Button.left):
            steps = max(5, duration_ms // 20)
            for i in range(1, steps + 1):
                if self.ks.engaged:
                    return False
                t = i / steps
                m.position = (int(x1 + (x2 - x1) * t), int(y1 + (y2 - y1) * t))
                time.sleep(duration_ms / 1000 / steps)
        self.stats.actions += 1
        return True

    def scroll(self, amount: int, x=None, y=None):
        """amount > 0 scrolls up, < 0 scrolls down."""
        if not self._gate() or not HAS_PYNPUT:
            return False
        m = mouse.Controller()
        if x is not None and y is not None:
            m.position = (x, y)
        m.scroll(0, amount)
        self.stats.actions += 1
        return True

    # -- keyboard ------------------------------------------------------------

    def type_text(self, text: str):
        if not self._gate() or not HAS_PYNPUT:
            return False
        k = keyboard.Controller()
        k.type(text)
        self.stats.actions += 1
        return True

    def press_key(self, key: str):
        """key like 'enter', 'ctrl+s', 'win'."""
        if not self._gate() or not HAS_PYNPUT:
            return False
        k = keyboard.Controller()
        parts = [p.strip() for p in key.split("+")]
        special = {"enter": keyboard.Key.enter, "tab": keyboard.Key.tab,
                   "esc": keyboard.Key.esc, "escape": keyboard.Key.esc,
                   "space": keyboard.Key.space, "backspace": keyboard.Key.backspace,
                   "delete": keyboard.Key.delete, "up": keyboard.Key.up,
                   "down": keyboard.Key.down, "left": keyboard.Key.left,
                   "right": keyboard.Key.right, "home": keyboard.Key.home,
                   "end": keyboard.Key.end, "win": keyboard.Key.cmd,
                   "cmd": keyboard.Key.cmd, "alt": keyboard.Key.alt,
                   "ctrl": keyboard.Key.ctrl, "shift": keyboard.Key.shift,
                   "f5": keyboard.Key.f5}
        keys = []
        for p in parts:
            lk = p.lower()
            if lk in special:
                keys.append(special[lk])
            elif len(p) == 1:
                keys.append(p)
            else:
                log.warning("unknown key part %r", p)
                return False
        for kk in keys[:-1]:
            k.press(kk)
        k.press(keys[-1]); k.release(keys[-1])
        for kk in reversed(keys[:-1]):
            k.release(kk)
        self.stats.actions += 1
        return True


# singletons
_ks: Optional[KillSwitch] = None
_input: Optional[InputController] = None


def get_kill_switch(hotkey: str = "<ctrl>+<alt>+q") -> KillSwitch:
    global _ks
    if _ks is None:
        _ks = KillSwitch(hotkey)
    return _ks


def get_input() -> InputController:
    global _input
    if _input is None:
        _input = InputController(get_kill_switch())
    return _input

"""Continuous screen-share streaming — Gemini Live parity.

Server-side loop: capture → change-detect (pixel hash) → downscale →
base64-push to the dashboard. Latest frame is kept for the assistant:
  • `look_at_screen` tool lets the model inspect it on demand
  • optional auto-injection into modular turns ("see my screen" mode)
"""
from __future__ import annotations

import base64
import logging
import threading
import time

from . import screen_capture

log = logging.getLogger("pai.screenshare")


class ScreenShareStreamer:
    def __init__(self, max_width: int = 960,
                 min_interval: float = 0.7,
                 on_frame=None):
        self.max_width = max_width
        self.min_interval = min_interval   # seconds between pushed frames
        self.on_frame = on_frame           # callback(png_bytes)
        self._thread = None
        self._stop = None
        self.latest_png: bytes | None = None
        self._last_hash = ""
        self._last_ts = 0.0
        self.n_pushed = 0

    def start(self) -> bool:
        if self._thread:
            return True
        try:
            screen_capture.get_capture().capture_png(max_width=self.max_width)
        except Exception as exc:  # noqa: BLE001
            log.error("screen share: capture unavailable: %s", exc)
            return False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="pai-screenshare")
        self._thread.start()
        log.info("screen share started (%.1fs interval)", self.min_interval)
        return True

    def stop(self):
        if self._stop:
            self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        self._thread = None
        self._stop = None
        log.info("screen share stopped")

    @property
    def running(self) -> bool:
        return bool(self._thread)

    def _loop(self):
        while not self._stop.is_set():
            t0 = time.time()
            try:
                png = screen_capture.get_capture().capture_png(
                    max_width=self.max_width)
                h = __import__("hashlib").sha256(png).hexdigest()[:16]
                if h != self._last_hash:
                    self._last_hash = h
                    self.latest_png = png
                    self._last_ts = time.time()
                    self.n_pushed += 1
                    if self.on_frame:
                        try:
                            self.on_frame(png)
                        except Exception:  # noqa: BLE001
                            pass
            except Exception as exc:  # noqa: BLE001
                log.warning("screen share frame failed: %s", exc)
            elapsed = time.time() - t0 + self.min_interval
            self._stop.wait(min(2.0, max(0.2, elapsed)))

    def get_b64(self) -> str | None:
        if self.latest_png:
            return base64.b64encode(self.latest_png).decode()
        return None

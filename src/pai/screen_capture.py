"""Screen capture with MSS → PIL fallback chain.

Adapted from D:/Agents-and-other-repos/Computer-Use/src/screen_capture.py
"""
from __future__ import annotations

import hashlib
import io
import logging
import time
from dataclasses import dataclass, field

log = logging.getLogger("pai.capture")

try:
    import mss
    import mss.tools
    HAS_MSS = True
except ImportError:
    HAS_MSS = False

try:
    from PIL import ImageGrab, Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


@dataclass
class CaptureStats:
    n_captures: int = 0
    n_failures: int = 0
    backend_use: dict = field(default_factory=dict)
    last_capture_ms: float = 0.0


class ScreenCapture:
    """Multi-backend screen capture with caching + change detection."""

    def __init__(self, backend: str = "auto", cache_ttl: float = 0.1):
        self.backend = backend
        self.cache_ttl = cache_ttl
        self._cache = None          # (timestamp, bytes of full-res PNG)
        self.stats = CaptureStats()

    # -- public API ---------------------------------------------------------

    def capture_png(self, monitor: int = 0,
                    max_width: int | None = None) -> bytes:
        """Capture the screen as PNG bytes.

        monitor 0 = all monitors virtual desktop.
        max_width: downscale so the image is at most this many px wide
        (keeps VLM payloads small). Cache stores the FULL-RES frame;
        resizing is applied per call.
        """
        full = self._capture_full(monitor)

        if max_width:
            png = self._resize(full, int(max_width))
        else:
            png = full

        self.stats.n_captures += 1
        self.stats.last_capture_ms = self._capture_ms
        self.stats.backend_use[self._backend_used] = (
            self.stats.backend_use.get(self._backend_used, 0) + 1)
        return png

    def capture_pil(self, monitor: int = 0, max_width: int | None = None):
        """Capture as a PIL Image (None if PIL unavailable)."""
        if not HAS_PIL:
            return None
        return Image.open(io.BytesIO(self.capture_png(monitor, max_width)))

    def screenshot_hash(self) -> str:
        return hashlib.sha256(self._capture_full(0)).hexdigest()[:16]

    def last_frame(self) -> bytes | None:
        """The most recent full-res frame, without forcing a new capture."""
        if self._cache and (time.time() - self._cache[0]) < self.cache_ttl * 100:
            return self._cache[1]
        return None

    # -- internals ----------------------------------------------------------

    _backend_used = "none"
    _capture_ms = 0.0

    def _capture_full(self, monitor: int) -> bytes:
        # cache check (full-res frames only, short TTL)
        if (
            self._cache is not None
            and (time.time() - self._cache[0]) < self.cache_ttl
        ):
            return self._cache[1]

        order = [self.backend] if self.backend != "auto" else ["mss", "pil"]
        for backend in order:
            try:
                t0 = time.perf_counter()
                if backend == "mss" and HAS_MSS:
                    with mss.mss() as sct:
                        if monitor == 0:
                            region = sct.monitors[0]
                        else:
                            region = sct.monitors[min(monitor, len(sct.monitors) - 1)]
                        raw = sct.grab(region)
                        png = mss.tools.to_png(raw.rgb, raw.size)
                elif backend == "pil" and HAS_PIL:
                    img = ImageGrab.grab(all_screens=(monitor == 0))
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    png = buf.getvalue()
                else:
                    continue
                self._capture_ms = (time.perf_counter() - t0) * 1000
                self._backend_used = backend
                self._cache = (time.time(), png)
                return png
            except Exception as exc:  # noqa: BLE001
                log.warning("capture backend %s failed: %s", backend, exc)
                self.stats.n_failures += 1
        raise RuntimeError("all screen capture backends failed")

    def _resize(self, png: bytes, max_width: int) -> bytes:
        if not HAS_PIL:
            return png
        try:
            img = Image.open(io.BytesIO(png))
            if img.width <= max_width:
                return png
            ratio = max_width / img.width
            img = img.resize((max_width, int(img.height * ratio)),
                             Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            return buf.getvalue()
        except Exception as exc:  # noqa: BLE001
            log.warning("resize failed: %s", exc)
            return png


# module-level singleton
_default: ScreenCapture | None = None


def get_capture(backend: str = "auto") -> ScreenCapture:
    global _default
    if _default is None:
        _default = ScreenCapture(backend=backend)
    return _default

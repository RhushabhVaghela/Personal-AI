"""Webcam capture — pairs with ScreenShareStreamer for camera input.

Captures JPEG frames from a local camera via OpenCV (fallback: PIL/ESCAPE
not available on Windows, so OpenCV is the practical path). Frames feed
the same look_at_screen pipeline; dashboard shows a 📷 tab preview.
"""
from __future__ import annotations

import logging
import threading

log = logging.getLogger("pai.webcam")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class WebcamCapture:
    def __init__(self, device: int = 0, width: int = 960):
        self.device = device
        self.width = width
        self.cap = None
        self._lock = threading.Lock()

    def available(self) -> bool:
        return HAS_CV2

    def start(self) -> bool:
        if not HAS_CV2:
            log.warning("webcam: opencv-python not installed")
            return False
        if self.cap:
            return True
        try:
            cap = cv2.VideoCapture(self.device, cv2.CAP_DSHOW)  # windows-fast
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            ok, _ = cap.read()
            if not ok:
                cap.release()
                log.warning("webcam %d: no frame", self.device)
                return False
            self.cap = cap
            log.info("webcam started (device %d)", self.device)
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("webcam start failed: %s", exc)
            return False

    def read_jpeg(self, quality: int = 80) -> bytes | None:
        if not self.cap:
            return None
        with self._lock:
            ok, frame = self.cap.read()
        if not ok:
            return None
        ok, buf = cv2.imencode(".jpg", frame,
                               [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else None

    def stop(self):
        if self.cap:
            with self._lock:
                self.cap.release()
            self.cap = None
            log.info("webcam stopped")

    def __del__(self):
        try:
            self.stop()
        except Exception:  # noqa: BLE001
            pass

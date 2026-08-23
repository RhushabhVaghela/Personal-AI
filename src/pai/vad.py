"""VAD engine — hands-free always-listening (Silero-class behaviour).

Two implementations:
  • WebRtcVadEngine   — CPU-light, robust (requires `webrtcvad-wheels`)
  • EnergyVadEngine   — pure-python fallback (RMS threshold)

The state machine is shared:
  idle → speaking-detected → [silence for hangover ms] → utterance end
Echo guard: while the assistant speaks, the mic gate is CLOSED so the
model doesn't hear itself; a hangover tail prevents clipping the first
word after replies.
"""
from __future__ import annotations

import collections
import logging
import time

log = logging.getLogger("pai.vad")

try:
    import webrtcvad
    HAS_WEBRTC = True
except ImportError:
    HAS_WEBRTC = False


class VadState:
    IDLE = "idle"
    SPEAKING = "speaking"
    ECHO_GUARD = "echo_guard"


class BaseVadEngine:
    def __init__(self, sample_rate: int = 16000,
                 frame_ms: int = 30,
                 silence_hangover_ms: int = 700):
        assert sample_rate in (8000, 16000, 32000, 48000)
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_bytes = int(sample_rate * frame_ms / 1000) * 2  # 16-bit
        self.silence_hangover_ms = silence_hangover_ms
        self.state = VadState.IDLE
        self.buffer = bytearray()
        self._speech_frames = 0
        self._silence_frames = 0
        self._last_voice_ts = 0.0

    # -- to implement -----------------------------------------------------

    def is_speech(self, frame: bytes) -> bool:  # pragma: no cover
        raise NotImplementedError

    # -- state machine ------------------------------------------------------

    def feed(self, chunk: bytes) -> str | None:
        """Feed raw PCM16 mono @sample_rate. Returns:
           None          — still collecting
           "partial"     — speech detected (utterance in progress)
           "utterance"   — complete utterance ready in .buffer
        """
        now = time.time()

        # echo guard: ignore everything while assistant speaks + hangover
        if self.state == VadState.ECHO_GUARD:
            if now - self._last_voice_ts > 0.15:
                self.state = VadState.IDLE
            return None

        n = len(self.buffer)
        for i in range(0, len(chunk) - self.frame_bytes + 1,
                       self.frame_bytes):
            frame = chunk[i:i + self.frame_bytes]
            voiced = self.is_speech(frame)
            if self.state == VadState.IDLE:
                if voiced:
                    self._speech_frames += 1
                    if self._speech_frames >= 3:      # ~90ms of voice
                        self.state = VadState.SPEAKING
                        self.buffer.clear()
            elif self.state == VadState.SPEAKING:
                if voiced:
                    self._silence_frames = 0
                    self._last_voice_ts = now
                else:
                    self._silence_frames += 1
                    if self._silence_frames * self.frame_ms >= \
                            self.silence_hangover_ms:
                        buf = bytes(self.buffer)
                        self.buffer.clear()
                        self.state = VadState.IDLE
                        self._speech_frames = 0
                        self._silence_frames = 0
                        if len(buf) > self.frame_bytes * 6:   # >180ms audio
                            return "utterance"
        return ("partial" if self.state == VadState.SPEAKING and
                len(self.buffer) > self.frame_bytes * 20 else None)

    def set_speaking(self, speaking: bool) -> None:
        """Echo control: call when assistant playback starts/stops."""
        if speaking:
            self.state = VadState.ECHO_GUARD
            self.buffer.clear()
            self._speech_frames = self._silence_frames = 0
        else:
            self._last_voice_ts = time.time()   # hangover before re-arm
            self.state = VadState.IDLE

    def reset(self) -> None:
        self.buffer.clear()
        self.state = VadState.IDLE
        self._speech_frames = self._silence_frames = 0


class WebRtcVadEngine(BaseVadEngine):
    """Robust CPU VAD (webrtcvad). aggressiveness 0-3 (3 = most strict)."""

    def __init__(self, aggressiveness: int = 2, **kw):
        super().__init__(**kw)
        if not HAS_WEBRTC:
            raise RuntimeError("webrtcvad not installed "
                               "(pip install webrtcvad-wheels)")
        self.vad = webrtcvad.Vad(aggressiveness)

    def is_speech(self, frame: bytes) -> bool:
        try:
            return self.vad.is_speech(frame, self.sample_rate)
        except Exception:
            return False


class EnergyVadEngine(BaseVadEngine):
    """Fallback: RMS threshold with adaptive noise floor."""

    def __init__(self, threshold: float = 0.008, **kw):
        super().__init__(**kw)
        self.threshold = threshold
        self.noise_floor = threshold

    def is_speech(self, frame: bytes) -> bool:
        import array
        samples = array.array("h", frame)
        if not samples:
            return False
        rms = (sum(s * s for s in samples) / len(samples)) ** 0.5 / 32768
        self.noise_floor = 0.95 * self.noise_floor + 0.05 * min(rms, 0.05)
        thr = max(self.threshold, self.noise_floor * 2.2)
        return rms > thr


def get_engine(prefer_webrtc: bool = True, **kw) -> BaseVadEngine:
    if prefer_webrtc and HAS_WEBRTC:
        try:
            return WebRtcVadEngine(**kw)
        except Exception as exc:  # noqa: BLE001
            log.warning("webrtcvad unavailable (%s), using energy VAD", exc)
    return EnergyVadEngine(**kw)


# continuous mic capture → engine, callback on utterance
def listen_continuous(engine: BaseVadEngine, on_utterance,
                      on_state=None, stop_flag=None,
                      device: int | None = None):
    """Blocking loop: mic → VAD → on_utterance(wav_path). Run in a thread."""
    import numpy as np
    import sounddevice as sd
    import soundfile as sf
    import tempfile
    from pathlib import Path

    blocksize = int(engine.sample_rate * engine.frame_ms / 1000) * 4
    stopped = False

    def callback(indata, frames, time_info, status):
        nonlocal stopped
        if stop_flag is not None and stop_flag.is_set():
            stopped = True
            raise sd.CallbackStop
        event = engine.feed(bytes(indata))
        if on_state:
            on_state(engine.state)
        if event == "utterance":
            out = Path(tempfile.gettempdir()) / f"pai_vad_{int(time.time()*1000)}.wav"
            sf.write(out, np.frombuffer(engine.buffer, dtype=np.int16),
                     engine.sample_rate, subtype="PCM_16")
            engine.buffer.clear()
            on_utterance(out)

    with sd.InputStream(samplerate=engine.sample_rate, channels=1,
                        dtype="int16", blocksize=blocksize,
                        device=device, callback=callback):
        while not stopped:
            sd.sleep(100)
            if stop_flag is not None and stop_flag.is_set():
                break

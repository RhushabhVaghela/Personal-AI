"""Audio I/O: microphone recording (push-to-talk or fixed window) + playback."""
from __future__ import annotations

import logging
import threading
from pathlib import Path

log = logging.getLogger("pai.audio")

try:
    import sounddevice as sd
    import soundfile as sf
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False


def record_seconds(seconds: float, sample_rate: int = 16000) -> Path:
    """Record from the default mic for N seconds. Returns wav path."""
    import tempfile
    if not HAS_AUDIO:
        raise RuntimeError("sounddevice/soundfile not installed")
    print(f"🎙️ Recording for {seconds:.0f}s ... speak now!")
    audio = sd.rec(int(seconds * sample_rate), samplerate=sample_rate,
                   channels=1, dtype="float32")
    sd.wait()
    out = Path(tempfile.gettempdir()) / "pai_question.wav"
    sf.write(out, audio, sample_rate, subtype="PCM_16")
    print("🎙️ Done recording.")
    return out


def record_until_enter(sample_rate: int = 16000) -> Path:
    """Record until the user presses Enter in the terminal."""
    import tempfile
    if not HAS_AUDIO:
        raise RuntimeError("sounddevice/soundfile not installed")
    frames: list = []
    stop = threading.Event()

    def callback(indata, frame_count, time_info, status):
        if not stop.is_set():
            frames.append(indata.copy())

    with sd.InputStream(samplerate=sample_rate, channels=1,
                        dtype="float32", callback=callback):
        input("🎙️ Recording... press Enter to stop > ")
        stop.set()
    import numpy as np
    audio = np.concatenate(frames, axis=0) if frames else np.zeros(
        (sample_rate, 1), dtype="float32")
    out = Path(tempfile.gettempdir()) / "pai_question.wav"
    sf.write(out, audio, sample_rate, subtype="PCM_16")
    return out


def play_wav(path: Path) -> None:
    if not HAS_AUDIO:
        log.warning("cannot play audio (no sounddevice)")
        return
    data, sr = sf.read(str(path), dtype="float32")
    sd.play(data, sr)
    sd.wait()


def play_any(path: Path) -> None:
    """Play wav or mp3 (mp3 via pydub if available, else ffmpeg fallback)."""
    if path.suffix.lower() == ".wav":
        play_wav(path)
        return
    import subprocess, shutil  # noqa: E401
    if shutil.which("ffplay"):
        subprocess.run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",
                        str(path)], check=False)
    else:
        log.warning("cannot play %s (no ffplay)", path)

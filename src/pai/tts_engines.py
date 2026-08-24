"""Local TTS engines: OmniVoice, ZipVoice, VibeVoice — swappable backends.

Each engine exposes one function:
    synthesize(text, out_path, voice=None) -> Path

All run fully offline on your GPU (or CPU) and are tiny relative to the
16 GB budget:

  OmniVoice   k2-fsa/OmniVoice          0.6B fp16 ≈ 1.2 GB VRAM
              RTF 0.025 (40x realtime), 600+ languages, voice cloning +
              design ([laughter] etc). The default recommendation.
  ZipVoice    k2-fsa/ZipVoice(-Distill) flow-matching zero-shot TTS,
              Distill = fewer steps = faster; dialog variant for 2-speaker.
  VibeVoice   microsoft/VibeVoice-1.5B  next-token diffusion, up to 90 min
              multi-speaker audio; heavy (~4-6 GB) — best for long-form,
              plus VibeVoice-Realtime-0.5B for streaming.

Engines load lazily and unload via free() so you can A/B them without
restarting the server.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

log = logging.getLogger("pai.tts_engines")

_ENGINES: dict = {}          # name -> loaded model instance


def _get_omnivoice():
    if "omnivoice" in _ENGINES:
        return _ENGINES["omnivoice"]
    try:
        import torch  # noqa: F401
        from omnivoice import OmniVoice
    except ImportError as exc:
        raise RuntimeError(
            "OmniVoice not installed. In a separate venv (python 3.12):\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu128\n"
            "  pip install omnivoice soundfile") from exc
    m = OmniVoice.from_pretrained("k2-fsa/OmniVoice",
                                  device_map="cuda:0", dtype="float16")
    _ENGINES["omnivoice"] = m
    return m


def _get_zipvoice(distill: bool = True):
    key = f"zipvoice{'_distill' if distill else ''}"
    if key in _ENGINES:
        return _ENGINES[key]
    raise RuntimeError(
        "ZipVoice runs from its own repo (k2-fsa/ZipVoice). Start its "
        "inference server and set zipvoice_url in config.yaml, or use "
        "the omnivoice engine.")


def _get_vibevoice(streaming: bool = False):
    key = "vibevoice_rt" if streaming else "vibevoice"
    if key in _ENGINES:
        return _ENGINES[key]
    raise RuntimeError(
        "VibeVoice requires the microsoft/VibeVoice repo venv "
        "(heavy: ~4-6 GB VRAM for 1.5B). Use 'vibevoice' only for "
        "long-form generation.")


# ---------------------------------------------------------------- public API

def synthesize(text: str, out_path: Path, engine: str = "omnivoice",
               voice=None, ref_audio: str | None = None,
               **kwargs) -> Path:
    """Synthesize text → out_path with the chosen local engine."""
    t0 = time.time()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if engine == "omnivoice":
        m = _get_omnivoice()
        gen_kwargs = {"text": text}
        if ref_audio:                      # zero-shot voice cloning
            gen_kwargs["ref_audio"] = ref_audio
        elif voice:                        # voice-design attributes
            gen_kwargs["attributes"] = voice
        audio = m.generate(**gen_kwargs)
        import soundfile as sf
        sf.write(str(out_path.with_suffix(".wav")), audio,
                 getattr(m, "sample_rate", 24000))

    elif engine == "zipvoice":
        m = _get_zipvoice(distill=True)
        # (server-based path — see docstring)
        raise RuntimeError("zipvoice engine requires its inference server")

    elif engine == "vibevoice":
        m = _get_vibevoice()
        raise RuntimeError("vibevoice engine requires the VibeVoice repo venv")

    else:
        raise ValueError(f"unknown TTS engine: {engine}")

    log.info("tts[%s]: %.2fs, %d chars -> %s",
             engine, time.time() - t0, len(text), out_path.name)
    return out_path


def free(engine: str | None = None) -> None:
    """Unload one engine (or all) to reclaim VRAM."""
    keys = [engine] if engine else list(_ENGINES.keys())
    for k in keys:
        m = _ENGINES.pop(k, None)
        try:
            import torch
            del m
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
    log.info("tts engines freed: %s", keys)


def loaded() -> list[str]:
    return list(_ENGINES.keys())

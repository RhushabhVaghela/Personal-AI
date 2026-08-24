"""Provider: Moshi / MoshiVis (kyutai) — full-duplex realtime S2S + vision.

WHY: the ONLY open model where you can interrupt mid-sentence and it just
*flows* like a human call (~200 ms). MoshiVis adds PaliGemma2 vision so
you can point at things while talking. Trade-off: 7B 2024-era brain —
great conversationalist, weak tool-caller; use for natural chat, switch
to voicechat/qwen_omni for PC-control tasks.

VRAM BUDGET (16 GB) — quantization choice is critical:
  PyTorch bf16   ✗ needs ~24 GB (officially unsupported quant) — REJECTED
  Rust q8        ✓ moshika-vis-candle-q8 = 8.83 GB weights
                   + Mimi codec ~0.5 GB + KV ~1-2 GB → ~11-12 GB total ✓✓
  MLX            ✗ Apple-silicon only — N/A on your RTX 5080
→ WE USE THE RUST BACKEND WITH Q8. Leaves ~4-5 GB headroom.

Windows reality check (from kyutai's README): "no official Windows
support". The Rust backend compiles on Windows with CUDA toolchain
(nvcc required), or runs in WSL2. We support both via config:
  moshi_backend: rust | wsl
  moshi_repo: path to a clone of kyutai-labs/moshi (rust/ subdir)

Protocol: moshi-server exposes a WebSocket at :8998 streaming raw PCM16
(16 kHz mono) both ways. We bridge: mic/VAD utterances → WS in, WS out →
speakers. Full-duplex means we DON'T gate the mic during replies.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import threading
import time
from pathlib import Path

from . import config

log = logging.getLogger("pai.moshi")


class MoshiProvider:
    name = "moshi"

    WS_URL_DEFAULT = "wss://127.0.0.1:8998"

    def __init__(self):
        cfg = config.get_config()
        self.cfg = cfg
        self.backend = getattr(cfg, "moshi_backend", "rust")  # rust | wsl
        self.repo = getattr(cfg, "moshi_repo",
                            r"D:\Agents-and-other-repos\moshi")
        self.variant = getattr(cfg, "moshi_variant", "moshika-vis-q8")
        self.proc: subprocess.Popen | None = None
        self._ws = None

    # -- lifecycle -----------------------------------------------------------

    def _server_cmd(self) -> list[str]:
        """Build the launch command per backend."""
        if self.backend == "wsl":
            return ["wsl", "-e", "bash", "-lc",
                    f"cd /mnt/d/Agents-and-other-repos/moshi/rust && "
                    f"cargo run --release --features cuda --bin moshi-backend"
                    f" -- --config moshi-backend/config-{self.variant}.json "
                    f"standalone"]
        exe = Path(self.repo) / "rust" / "target" / "release" / \
            "moshi-backend.exe"
        if not exe.exists():
            raise RuntimeError(
                f"moshi-backend not built at {exe}.\n"
                "Build once (needs Rust + CUDA nvcc):\n"
                f"  cd {self.repo}\\rust\n"
                "  cargo run --release --features cuda --bin moshi-backend"
                " -- --config moshi-backend/config-moshika-vis-q8.json standalone")
        cfg_file = Path(self.repo) / "rust" / "moshi-backend" / \
            f"config-{self.variant}.json"
        return [str(exe), "--config", str(cfg_file), "standalone"]

    def start(self):
        if self.is_running():
            return True
        cmd = self._server_cmd()
        log.info("starting moshi (%s): %s", self.backend, " ".join(cmd))
        self.proc = subprocess.Popen(
            cmd, cwd=str(self.repo),
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        # wait for the ws port
        deadline = time.time() + 600    # first compile can take ages
        import socket
        while time.time() < deadline:
            if self.proc.poll() is not None:
                break
            with socket.socket() as s:
                s.settimeout(1)
                if s.connect_ex(("127.0.0.1", 8998)) == 0:
                    log.info("moshi server up on :8998")
                    return True
            time.sleep(3)
        raise RuntimeError(
            "moshi server did not start (see its console output). "
            "If this is a compile error, install Rust + CUDA toolkit and "
            "build manually inside the repo.")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
        self.proc = None

    def is_running(self) -> bool:
        return bool(self.proc and self.proc.poll() is None)

    # -- conversation -------------------------------------------------------------

    def turn(self, wav_path, executor=None, image_path=None,
             max_tool_rounds: int = 2) -> dict:
        """Moshi is session/full-duplex based; a 'turn' here bridges one VAD
        utterance through the websocket and collects the spoken response.

        NOTE: true full-duplex mode bypasses turn() entirely (the dashboard
        streams PCM straight through); this method exists so hands-free and
        terminal flows still work uniformly.
        """
        try:
            wav, txt = _moshi_exchange_sync(wav_path)
            if wav is None:
                return {"text": "(moshi silent)", "audio": None,
                        "tool_calls": []}
            return {"audio": str(wav), "text": txt, "tool_calls": []}
        except Exception as exc:  # noqa: BLE001
            log.error("moshi turn failed: %s", exc)
            return {"text": f"(moshi error: {exc})", "audio": None,
                    "tool_calls": []}


def _moshi_exchange_sync(wav_path):
    """Run one utterance through moshi-server's WebSocket.

    Creates its own event loop on a fresh thread because this is called
    from worker threads (asyncio.run would collide with any running loop).
    """
    import asyncio
    import websockets
    import soundfile as sf
    import numpy as np

    result = {"wav": None, "text": "", "error": None}

    async def _run():
        data, sr = sf.read(str(wav_path), dtype="int16")
        if sr != 16000:
            raise RuntimeError(f"need 16 kHz audio, got {sr}")
        pcm = data.tobytes()
        out_chunks = bytearray()
        # rust backend uses self-signed https by default; try ws first
        url = MoshiProvider.WS_URL_DEFAULT.replace("wss://", "ws://")
        async with websockets.connect(url, max_size=None) as ws:
            step = 1920 * 2                      # 120 ms of PCM16
            for i in range(0, len(pcm), step):
                await ws.send(pcm[i:i + step])
                await asyncio.sleep(0.01)
            last = time.time()
            while time.time() - last < 1.5:
                try:
                    msg = await asyncio.wait_for(ws.recv(), 0.5)
                    if isinstance(msg, (bytes, bytearray)):
                        out_chunks.extend(msg)
                        last = time.time()
                    else:
                        log.debug("moshi text: %s", str(msg)[:100])
                except asyncio.TimeoutError:
                    continue

        audio = np.frombuffer(bytes(out_chunks), dtype="int16")
        if len(audio) == 0:
            result["wav"], result["text"] = None, ""
            return
        out = Path(tempfile_dir()) / f"pai_moshi_{int(time.time()*1000)}.wav"
        sf.write(out, audio, 16000)
        result["wav"], result["text"] = out, ""

    def _target():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_run())
        except Exception as exc:  # noqa: BLE001
            result["error"] = exc
        finally:
            loop.close()

    t = threading.Thread(target=_target)
    t.start()
    t.join(timeout=120)
    if t.is_alive():
        raise RuntimeError("moshi exchange timed out")
    if result["error"]:
        raise result["error"]
    return result["wav"], result["text"]


def tempfile_dir() -> str:
    import tempfile
    return tempfile.gettempdir()

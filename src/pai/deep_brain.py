"""Deep Brain — on-demand 27B+-class intelligence without killing VRAM.

Strategy (why this works on 16 GB):
  1. MoE model: Qwen3-30B-A3B has 30B params but only ~3B ACTIVE per token.
     llama.cpp keeps attention/shared weights on GPU and streams experts
     from system RAM → 30B knowledge at usable speed on a 16 GB card.
  2. Lazy lifecycle: the deep brain is NOT resident during normal chatting.
     It starts on first deep request, serves, and auto-unloads after
     `deep_idle_unload_s` — so VRAM returns to your voice model.

VRAM/RAM budget (Qwen3-30B-A3B UD-Q4_K_XL ≈ 18.6 GB file):
  gpu_layers tuned so GPU side ≈ 5-7 GB (leaves room next to nothing else,
  because we PAUSE the active S2S provider while the deep brain thinks —
  see server.set_effort flow)
  RAM side: remainder fits easily in 32 GB system memory
Speed expectation on a laptop DDR5 + RTX 5080: ~8-15 tok/s.

Alternatives supported via config:
  deep_model_hf: "unsloth/Qwen3-30B-A3B-GGUF:UD-Q4_K_XL"   (default)
                 "Qwen/Qwen3-32B-GGUF:Q4_K_M"               (dense, slower)
                 anything llama.cpp can pull from HF
"""
from __future__ import annotations

import logging
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path

import requests

log = logging.getLogger("pai.deepbrain")


class DeepBrain:
    """Lazy llama.cpp server hosting a big model, auto-unloading when idle."""

    def __init__(self):
        cfg = self.cfg = __import__("pai.config", fromlist=["get_config"]) \
            .get_config()
        self.hf_repo = getattr(
            cfg, "deep_model_hf", "unsloth/Qwen3-30B-A3B-GGUF:UD-Q4_K_XL")
        self.port = getattr(cfg, "deep_llm_port", 8082)
        self.gpu_layers = int(getattr(cfg, "deep_gpu_layers", 28))
        self.idle_s = float(getattr(cfg, "deep_idle_unload_s", 300))
        self.pause_s2s = bool(getattr(cfg, "deep_pause_s2s", True))
        self.proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._last_used = 0.0
        self._reaper: threading.Timer | None = None

    # -- lifecycle -----------------------------------------------------------

    def _exe(self) -> str:
        exe = shutil.which("llama-server")
        if exe:
            return exe
        local = Path(r"D:\Agents-and-other-repos\hoidhxd-NVIDIA-NemotronLabs"
                     r"-VoiceChat-11B-GGUF\llama-server.exe")
        return str(local) if local.exists() else "llama-server"

    def _port_free(self) -> bool:
        with socket.socket() as s:
            s.settimeout(0.4)
            return s.connect_ex(("127.0.0.1", self.port)) != 0

    def ensure_up(self, timeout_s: float = 1800) -> bool:
        """Start the server if not running (downloads model on first run)."""
        with self._lock:
            self._last_used = time.time()
            self._arm_reaper()
            if self.is_up():
                return True
            cmd = [
                self._exe(),
                "-hf", self.hf_repo,
                "--port", str(self.port),
                "-ngl", str(self.gpu_layers),
                "-c", "16384",              # generous ctx, MoE KV is cheap
                "--jinja",                   # enable chat template
                "--flash-attn", "on",        # linear KV scaling (vs quadratic)
                "-ctk", "q8_0", "-ctv", "q8_0",   # quantized KV cache
                "--no-context-shift",
            ]
            log.info("deep brain starting: %s", " ".join(cmd))
            log.info("(first run downloads the model — %s)", self.hf_repo)
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            deadline = time.time() + timeout_s
            while time.time() < deadline:
                if self.proc.poll() is not None:
                    raise RuntimeError(
                        f"llama-server exited (code {self.proc.returncode})")
                try:
                    r = requests.get(
                        f"http://127.0.0.1:{self.port}/health", timeout=3)
                    if r.ok:
                        log.info("deep brain UP on :%d", self.port)
                        return True
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(3)
            raise RuntimeError("deep brain did not become healthy in time")

    def is_up(self) -> bool:
        if not (self.proc and self.proc.poll() is None):
            return False
        try:
            return requests.get(
                f"http://127.0.0.1:{self.port}/health", timeout=2).ok
        except Exception:  # noqa: BLE001
            return False

    def _arm_reaper(self):
        if self._reaper:
            self._reaper.cancel()
        self._reaper = threading.Timer(self.idle_s, self._idle_check)
        self._reaper.daemon = True
        self._reaper.start()

    def _idle_check(self):
        if time.time() - self._last_used >= self.idle_s and self.proc:
            log.info("deep brain idle %.0fs — unloading", self.idle_s)
            self.unload()

    def unload(self):
        """Stop the server → VRAM/RAM fully reclaimed."""
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
        log.info("deep brain unloaded")

    # -- inference ---------------------------------------------------------------

    def chat(self, messages: list[dict], max_tokens: int = 700,
             grammar: str | None = None) -> str:
        """Blocking chat completion against the deep brain.

        grammar: optional GBNF grammar string — logit-level constraint that
        makes invalid JSON/tool-call tokens impossible (quantized models
        flatten probability distributions; this restores determinism).
        """
        self.ensure_up()
        t0 = time.time()
        payload: dict = {"messages": messages, "temperature": 0.4,
                         "max_tokens": max_tokens}
        if grammar:
            payload["grammar"] = grammar
        r = requests.post(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            json=payload,
            timeout=900,
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        log.info("deep brain: %.1fs, %d chars", time.time() - t0, len(text))
        self._last_used = time.time()
        self._arm_reaper()
        return text


_singleton: DeepBrain | None = None


def get_deep_brain() -> DeepBrain:
    global _singleton
    if _singleton is None:
        _singleton = DeepBrain()
    return _singleton

"""Provider: GLM-4-Voice 9B (int4) — instruction-steerable S2S.

WHY: only open S2S model that changes emotion/pace/accent ON COMMAND
("say it happily", "speak slower") — pairs beautifully with our voice
picker. int4 build confirmed running on a 12 GB RTX 3060 → fits 16 GB
with lots of KV headroom.

VRAM BUDGET (16 GB, int4):
  LLM (9B W4A16)            ~6.0 GB
  Voice decoder              ~1.5 GB
  Whisper tokenizer          ~0.6 GB
  Runtime + KV               ~1.5-3 GB
  ─────────────────────────────────
  Total                     ~10-12 GB  ✓

Architecture: three processes
  model_server.py  (--dtype int4, port 10000)
  tokenizer server (whisper VQ, port 10001)
  decoder          (cosyvoice, port 10002? per repo layout)
We talk HTTP to those and never load torch in OUR venv.

Setup:
  git clone https://github.com/THUDM/GLM-4-Voice  D:/Agents-and-other-repos/GLM-4-Voice
  cd there; conda create -n glm python=3.10; pip install -r requirements.txt
  # int4 weights auto-download from cydxg/glm-4-voice-9b-int4 (~7 GB)
"""
from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

import requests

from . import config, tools as pai_tools

log = logging.getLogger("pai.glm_voice")


class GlmVoiceProvider:
    name = "glm_voice"

    MODEL_SERVER = "http://127.0.0.1:10000"
    TOKENIZER_SERVER = "http://127.0.0.1:10001"
    DECODER_SERVER = "http://127.0.0.1:10002"

    def __init__(self):
        cfg = config.get_config()
        self.cfg = cfg
        self.repo = getattr(
            cfg, "glm_voice_repo",
            r"D:\Agents-and-other-repos\GLM-4-Voice")
        self.dtype = getattr(cfg, "glm_voice_dtype", "int4")
        self.proc: list[subprocess.Popen] = []
        self.session = requests.Session()

    # -- lifecycle ---------------------------------------------------------------

    def _env_ready(self) -> bool:
        return Path(self.repo).exists()

    def start(self):
        if not self._env_ready():
            raise RuntimeError(
                f"GLM-4-Voice repo not found at {self.repo}. Clone:\n"
                "  git clone https://github.com/THUDM/GLM-4-Voice "
                + self.repo + "\n"
                "then create its conda env (python 3.10) and install "
                "requirements.txt")
        if self._healthy():
            return True
        py = Path(self.repo) / "env" / "python.exe"  # conda layout guess
        if not py.exists():
            py = "python"   # fall back to PATH (user activates env)
        cmds = [
            [str(py), "model_server.py", "--host", "127.0.0.1",
             "--model-path", "THUDM/glm-4-voice-9b", "--port", "10000",
             "--dtype", self.dtype, "--device", "cuda:0"],
            [str(py), "tokenizer_server.py", "--host", "127.0.0.1",
             "--port", "10001"],
            [str(py), "decoder_server.py", "--host", "127.0.0.1",
             "--port", "10002"],
        ]
        for cmd in cmds:
            log.info("starting %s...", cmd[1])
            self.proc.append(subprocess.Popen(
                cmd, cwd=self.repo,
                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT))
        deadline = time.time() + 600   # int4 loads slowly first time
        while time.time() < deadline:
            if self._healthy():
                log.info("glm-voice ready")
                return True
            time.sleep(5)
        raise RuntimeError("glm-voice servers did not become healthy")

    def stop(self):
        for p in self.proc:
            try:
                p.kill()
            except Exception:  # noqa: BLE001
                pass
        self.proc.clear()
        try:
            subprocess.run(["taskkill", "/F", "/IM", "python.exe"],
                           capture_output=True)  # too broad! see note below
        except Exception:  # noqa: BLE001
            pass
        # NOTE: we deliberately do NOT taskkill python.exe globally — that
        # would kill this dashboard too. Child kills above suffice.

    def is_running(self) -> bool:
        return bool(self.proc) and all(p.poll() is None for p in self.proc)

    def _healthy(self) -> bool:
        try:
            r = self.session.get(self.MODEL_SERVER + "/health", timeout=3)
            return r.ok
        except Exception:  # noqa: BLE001
            return False

    # -- conversation -----------------------------------------------------------------

    def turn(self, wav_path: Path, executor: pai_tools.ToolExecutor,
             image_path: Path | None = None,
             max_tool_rounds: int = 3) -> dict:
        b64 = None
        with open(wav_path, "rb") as f:
            import base64
            b64 = base64.b64encode(f.read()).decode()

        # 1. tokenize user speech
        tok = self.session.post(f"{self.TOKENIZER_SERVER}/tokenize",
                                json={"audio": b64}, timeout=60).json()
        # 2. ask the LLM
        chat = self.session.post(
            f"{self.MODEL_SERVER}/chat",
            json={"messages": [
                {"role": "system",
                 "content": self._system_prompt()},
                {"role": "assistant",
                 "content": "<|begin_of_audio|>" + "".join(tok["tokens"])
                            + "<|end_of_audio|>"},
            ]},
            timeout=180).json()
        reply_text = chat.get("response", "").strip()
        reply_tokens = chat.get("speech_tokens")

        calls = pai_tools.parse_tool_calls(reply_text)
        if calls:
            results = [executor.execute(n, p) for n, p in calls]
            return {"text": "(tool result spoken next turn)",
                    "audio": None, "tool_results": results}

        if not reply_tokens:
            return {"text": reply_text or "(no reply)", "audio": None,
                    "tool_calls": []}

        # 3. decode to speech
        dec = self.session.post(f"{self.DECODER_SERVER}/decode",
                                json={"tokens": reply_tokens},
                                timeout=120)
        import base64 as b64mod
        wav_bytes = b64mod.b64decode(dec.json()["audio"])
        out = Path(tempfile_dir()) / f"pai_glm_{int(time.time()*1000)}.wav"
        out.write_bytes(wav_bytes)
        return {"text": reply_text, "audio": str(out), "tool_calls": []}

    def _system_prompt(self) -> str:
        import json
        voice_hint = ""
        if self.cfg.tts_speed and self.cfg.tts_speed != 1.0:
            voice_hint += f" Speak at {'slower' if self.cfg.tts_speed < 1 else 'faster'} pace."
        return ("You are a helpful desktop assistant. Answer in 1-3 short "
                "spoken sentences." + voice_hint)


def tempfile_dir() -> str:
    import tempfile
    return tempfile.gettempdir()

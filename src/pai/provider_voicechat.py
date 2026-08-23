"""Provider: Nemotron VoiceChat 11B via llama-voicechat.exe --serve.

Single speech-to-speech model (STT+LLM+TTS in one). Communication is via
the process stdin/stdout JSON protocol (same as push_to_talk.py):

  -> {"cmd": "turn", "audio": "<in.wav>", "out": "<out.wav>"}
  <- reply WAV written to <out.wav> when generation finishes

The function-head GGUF is attempted first and auto-falls-back to the
standard STT-LLM model if the installed exe doesn't support it.
"""
from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from pathlib import Path

from . import config, tools as pai_tools

log = logging.getLogger("pai.voicechat")


class VoiceChatProvider:
    name = "voicechat"

    def __init__(self, use_funchead: bool | None = None, port: int | None = None):
        cfg = config.get_config()
        self.use_funchead = (cfg.voicechat_use_funchead if use_funchead is None
                             else use_funchead)
        self.port = port or cfg.voicechat_port
        self.proc: subprocess.Popen | None = None
        self._reader = None
        self._events: list[dict] = []
        self._ev_lock = threading.Lock()
        self._cv = threading.Condition(self._ev_lock)
        self._turn_seq = 0

    # -- lifecycle ------------------------------------------------------------

    def start(self):
        if self.proc and self.proc.poll() is None:
            return True
        if self.use_funchead:
            try:
                return self._launch(use_funchead=True)
            except RuntimeError:
                log.warning("funchead model failed to load "
                            "(exe may not support voicechat_function_head); "
                            "falling back to standard STT-LLM model")
                self.use_funchead = False
        return self._launch(use_funchead=False)

    def _launch(self, use_funchead: bool):
        main = config.GGUF_FUNCHEAD if use_funchead else config.GGUF_MAIN
        cmd = [
            str(config.VOICECHAT_EXE),
            "-m", str(main),
            "--mmproj", str(config.GGUF_MMPROJ),
            "--tts", str(config.GGUF_TTS),
            "-ngl", "99",
            "--serve",
        ]
        log.info("starting voicechat: %s", " ".join(cmd))
        with self._cv:
            self._events.clear()   # stale events from a previous launch
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            bufsize=1, cwd=str(config.VOICECHAT_DIR),
        )
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        ok = self._wait_ready(timeout=180)
        if not ok:
            raise RuntimeError(
                f"voicechat failed to become ready "
                f"(exit code {self.proc.poll() if self.proc else 'n/a'})")
        log.info("voicechat ready (%s)",
                 "funchead" if use_funchead else "standard STT-LLM")
        return True

    def stop(self):
        if self.proc and self.proc.poll() is None:
            # graceful: try JSON exit, then hard-kill after a grace period
            try:
                self.proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                self.proc.stdin.flush()
            except Exception:  # noqa: BLE001
                pass
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                log.warning("voicechat did not exit cleanly; killing")
                self.proc.kill()
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        # belt-and-braces: reap any orphaned llama-voicechat from OUR cwd
        try:
            subprocess.run(
                ["taskkill", "/F", "/IM", "llama-voicechat.exe"],
                capture_output=True, timeout=10)
        except Exception:  # noqa: BLE001
            pass
        self.proc = None

    def is_running(self) -> bool:
        return bool(self.proc and self.proc.poll() is None)

    def _read_loop(self):
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            if line.startswith("SERVER:") or line.startswith("cmn"):
                log.debug("%s", line[:200])
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                log.debug("voicechat out: %s", line[:200])
                continue
            with self._cv:
                self._events.append(ev)
                self._cv.notify_all()

    def _wait_ready(self, timeout: float = 180):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._cv:
                for ev in self._events:
                    if (ev.get("type") in ("ready", "server_ready", "listening")
                            or ev.get("kind") == "ready"):
                        return True
            if self.proc and self.proc.poll() is not None:
                return False
            time.sleep(0.5)
        return False

    # -- conversation ----------------------------------------------------------

    def turn(self, wav_path: Path, executor: pai_tools.ToolExecutor,
             image_path: Path | None = None,
             max_tool_rounds: int = 5) -> dict:
        """Send one voice turn using the exe's file-based protocol.

        Protocol (matches push_to_talk.py):
          -> {"cmd":"turn", "audio": <in.wav>, "out": <out.wav>}
          <- reply WAV written to <out.wav> when generation finishes
        """
        if not self.is_running():
            self.start()
        out_wav = config.VOICECHAT_DIR / f"pai_answer_{self._turn_seq % 4}.wav"
        self._turn_seq += 1
        if out_wav.exists():
            try:
                out_wav.unlink()
            except OSError:
                pass
        event = {"cmd": "turn", "audio": str(wav_path), "out": str(out_wav)}
        if image_path:
            event["image"] = str(image_path)
        self._send(event)
        log.info("turn: sent (%d bytes audio), waiting for reply...",
                 wav_path.stat().st_size)

        # wait for the reply wav to appear and stop growing
        deadline = time.time() + 180
        last_size = -1
        stable = 0
        while time.time() < deadline:
            if not self.is_running():
                return {"text": "(voicechat exited)", "tool_calls": [],
                        "audio": None}
            if out_wav.exists():
                size = out_wav.stat().st_size
                if size == last_size and size > 1000:
                    stable += 1
                    if stable >= 3:            # ~1.5s of no growth = done
                        return {"audio": str(out_wav), "text": "",
                                "tool_calls": []}
                else:
                    stable = 0
                last_size = size
            time.sleep(0.5)
        log.error("turn: timed out after %.0fs waiting for %s",
                  time.time() - (deadline - 180), out_wav.name)
        return {"text": "(timeout waiting for voicechat reply)",
                "tool_calls": [], "audio": None}

    def _send(self, event: dict):
        assert self.proc and self.proc.stdin
        self.proc.stdin.write(json.dumps(event) + "\n")
        self.proc.stdin.flush()

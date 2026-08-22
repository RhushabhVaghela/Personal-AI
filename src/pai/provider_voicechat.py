"""Provider: Nemotron VoiceChat 11B via llama-voicechat.exe --serve.

Single speech-to-speech model (STT+LLM+TTS in one). Communication is via
the process stdin/stdout JSON event protocol (same as push_to_talk.py):

  -> {"cmd": "turn", "input": "<wav-path>", ...}
  <- {"type": "audio", "audio_path": "..."} / {"type": "text", ...}

With the function-head GGUF the model can also emit tool calls; this
provider feeds tool results back and loops until a final audio answer.
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

    # -- lifecycle ------------------------------------------------------------

    def start(self):
        if self.proc and self.proc.poll() is None:
            return
        try:
            self._launch(use_funchead=self.use_funchead)
            self._wait_ready(timeout=180)
        except RuntimeError:
            if self.use_funchead:
                log.warning("funchead model failed to load "
                            "(exe may not support voicechat_function_head); "
                            "falling back to standard STT-LLM model")
                self.use_funchead = False
                self._launch(use_funchead=False)
                self._wait_ready(timeout=180)
            else:
                raise

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
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            bufsize=1, cwd=str(config.VOICECHAT_DIR),
        )
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self._wait_ready(timeout=180)

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.write(json.dumps({"cmd": "exit"}) + "\n")
                self.proc.stdin.flush()
            except Exception:  # noqa: BLE001
                self.proc.kill()
        self.proc = None

    def _read_loop(self):
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            if line.startswith("SERVER:"):
                log.debug("%s", line)
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                log.debug("voicechat out: %s", line[:200])
                continue
            with self._cv:
                self._events.append(ev)
                self._cv.notify_all()

    def _wait_ready(self, timeout: float = 120):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._cv:
                for ev in self._events:
                    if (ev.get("type") in ("ready", "server_ready", "listening")
                            or ev.get("kind") == "ready"):
                        return True
            if self.proc and self.proc.poll() is not None:
                raise RuntimeError("voicechat server exited during startup")
            time.sleep(0.5)
        log.warning("voicechat readiness not signalled; continuing anyway")
        return False

    def _wait_event(self, kinds: tuple[str, ...], timeout: float = 120) -> dict | None:
        deadline = time.time() + timeout
        seen = 0
        while time.time() < deadline:
            with self._cv:
                if len(self._events) > seen:
                    ev = self._events[seen]
                    seen += 1
                    if ev.get("type") in kinds:
                        return ev
                    continue
                self._cv.wait(min(1.0, deadline - time.time()))
        return None

    # -- conversation ----------------------------------------------------------

    def turn(self, wav_path: Path, executor: pai_tools.ToolExecutor,
             image_path: Path | None = None,
             max_tool_rounds: int = 5) -> dict:
        """Send one voice turn. Returns {"audio": path?, "text": str, "tool_calls": [...]}"""
        for round_no in range(max_tool_rounds + 1):
            event = {"cmd": "turn", "input": str(wav_path)}
            if image_path:
                event["image"] = str(image_path)
            if round_no > 0 and self._events:
                pass  # follow-up rounds carry tool results via transcript file
            self._send(event)
            ev = self._wait_event(
                ("audio", "text", "tool_call", "tool_calls", "final"), timeout=180)
            if ev is None:
                return {"text": "(timeout waiting for voicechat)", "tool_calls": []}
            calls = ev.get("tool_calls") or (
                [ev["tool_call"]] if ev.get("tool_call") else [])
            if calls:
                results = [executor.execute(c.get("name", c.get("tool", "")),
                                            c.get("params", c.get("arguments", {})))
                           for c in calls]
                # write results to transcript so the model sees them next round
                import tempfile
                rf = Path(tempfile.gettempdir()) / "pai_tool_results.json"
                rf.write_text(json.dumps(results, indent=2, default=str),
                              encoding="utf-8")
                self._send({"cmd": "tool_results", "path": str(rf)})
                continue
            return {
                "audio": ev.get("audio_path") or ev.get("audio"),
                "text": ev.get("text") or ev.get("transcript") or "",
                "tool_calls": [],
            }
        return {"text": "(max tool rounds reached)", "tool_calls": []}

    def _send(self, event: dict):
        assert self.proc and self.proc.stdin
        self.proc.stdin.write(json.dumps(event) + "\n")
        self.proc.stdin.flush()

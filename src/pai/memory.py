"""ConversationStore — hours-long memory for endless conversations.

Design (matches ChatGPT Voice behaviour):
  • every turn appended to a JSONL session file (survives restarts)
  • last N turns sent to the LLM as rolling context
  • when history grows past `memory_summarize_after`, older turns are
    folded into a compact summary (kept at the front of the prompt)
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("pai.memory")


class ConversationStore:
    def __init__(self, root: Path | None = None,
                 session: str = "default",
                 max_turns: int = 24,
                 summarize_after: int = 40):
        self.root = Path(root) if root else (
            Path(__file__).resolve().parents[3] / "sessions")
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / f"{session}.jsonl"
        self.max_turns = max_turns
        self.summarize_after = summarize_after
        self.turns: list[dict] = []      # [{"role","text","ts"}]
        self.summary: str = ""
        self._load()

    # -- persistence -----------------------------------------------------------

    def _load(self):
        if not self.path.exists():
            return
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                if rec.get("type") == "summary":
                    self.summary = rec.get("text", "")
                elif rec.get("role") in ("user", "assistant"):
                    self.turns.append(rec)
            # honor the rolling window on reload (file keeps full history)
            if len(self.turns) > self.max_turns:
                self.turns = self.turns[-self.max_turns:]
        except Exception as exc:  # noqa: BLE001
            log.warning("session load failed: %s", exc)

    def _append(self, rec: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # -- api ---------------------------------------------------------------------

    def add(self, role: str, text: str) -> None:
        if not text:
            return
        rec = {"role": role, "text": text, "type": "turn",
               "ts": round(time.time(), 3)}
        self.turns.append(rec)
        self._append(rec)
        if len(self.turns) > self.summarize_after:
            self._maybe_summarize()

    def context(self) -> list[dict]:
        """Rolling context: summary + last N turns, OpenAI message format."""
        msgs: list[dict] = []
        if self.summary:
            msgs.append({"role": "system",
                         "content": f"Earlier conversation summary: {self.summary}"})
        for t in self.turns[-self.max_turns:]:
            msgs.append({"role": t["role"], "content": t["text"]})
        return msgs

    def reset(self) -> None:
        """Start a fresh conversation (archives the old file)."""
        if self.path.exists():
            archive = self.path.with_suffix(
                f".{int(time.time())}.jsonl.bak")
            self.path.rename(archive)
        self.turns.clear()
        self.summary = ""

    def stats(self) -> dict:
        return {"turns": len(self.turns), "summarized": bool(self.summary),
                "file": self.path.name}

    # -- summarization -------------------------------------------------------------

    def _maybe_summarize(self) -> None:
        """Fold old turns into the running summary.

        Uses an extractive fallback by default (no LLM needed); providers can
        override via `set_summarizer` to use their LLM.
        """
        cut = len(self.turns) - self.max_turns
        if cut <= 0:
            return
        old, self.turns = self.turns[:cut], self.turns[cut:]

        # extractive fallback: keep first sentence of each old turn
        lines = []
        for t in old:
            s = t["text"].strip().replace("\n", " ")
            if s:
                lines.append(("U: " if t["role"] == "user" else "A: ")
                             + s[:120])
        fold = "\n".join(lines[-20:])
        self.summary = ((self.summary + "\n" if self.summary else "")
                        + fold)[:2000]
        self._append({"type": "summary", "text": self.summary})
        log.info("memory: summarized %d turns (history=%d)",
                 cut, len(self.turns))

    def set_summarizer(self, fn) -> None:
        """fn(old_summary, fold_text) -> new_summary (LLM-backed)."""
        self._summarizer = fn

    def attach_llm_summarizer(self, chat_fn) -> None:
        """Wire an LLM chat function for high-quality summaries.

        chat_fn(messages: list[dict]) -> str
        """
        def _summarize(old: str, fold: str) -> str:
            try:
                prompt = [
                    {"role": "system", "content":
                     "Summarize ongoing voice conversations into compact "
                     "notes. Keep facts, names, decisions and open items. "
                     "Merge with any prior summary."},
                    {"role": "user",
                     "content": f"Prior summary:\n{old or '(none)'}\n\nNew "
                                f"exchanges to fold in:\n{fold}\n\nUpdated "
                                f"summary:"},
                ]
                out = chat_fn(prompt).strip()
                if out:
                    return out[:2000]
            except Exception as exc:  # noqa: BLE001
                log.warning("LLM summarizer failed (%s); extractive fallback",
                            exc)
            # extractive fallback
            lines = [l for l in fold.splitlines() if l.strip()]
            return ((old + "\n" if old else "") + "\n".join(lines[-20:]))[:2000]
        self.set_summarizer(_summarize)

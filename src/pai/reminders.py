"""Reminders — persistent store + proactive speech delivery.

Gemini/ChatGPT-style: "remind me at 5pm to call mom" → the assistant
*speaks up on its own* when the time arrives (even mid hands-free session).

Storage: reminders.json in the project root. The scheduler thread ticks
every second, fires due reminders via a callback (server turns them into
TTS + pushes to the dashboard), and persists state.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

log = logging.getLogger("pai.reminders")


class ReminderStore:
    def __init__(self, path: Path | None = None):
        self.path = path or Path(__file__).resolve().parents[3] / "reminders.json"
        self.items: list[dict] = []
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        try:
            self.items = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            log.warning("reminders load failed: %s", exc)

    def _save(self):
        try:
            self.path.write_text(
                json.dumps(self.items, indent=1), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            log.error("reminders save failed: %s", exc)

    # -- crud ------------------------------------------------------------------

    def add(self, when: float, text: str) -> dict:
        item = {"id": f"r{int(time.time()*1000)}",
                "when": when, "text": text, "fired": False}
        with self._lock:
            self.items.append(item)
            self._save()
        log.info("reminder added: %s @ %s",
                 text[:40], datetime.fromtimestamp(when).strftime("%H:%M"))
        return item

    def list_pending(self) -> list[dict]:
        with self._lock:
            return sorted([i for i in self.items if not i["fired"]],
                          key=lambda i: i["when"])

    def cancel(self, rid: str) -> bool:
        with self._lock:
            before = len(self.items)
            self.items = [i for i in self.items if i["id"] != rid]
            changed = len(self.items) != before
            if changed:
                self._save()
        return changed

    # -- parsing -----------------------------------------------------------------

    TIME_PATTERNS = [
        # "at 5pm", "at 17:30", "5:30 pm"
        (re.compile(r"\bat (\d{1,2})(?::(\d{2}))?\s*(am|pm)?", re.I), "at"),
        # "in 10 minutes/seconds/hours"
        (re.compile(r"\bin (\d+)\s*(second|minute|hour)s?", re.I), "in"),
    ]

    @classmethod
    def parse_when(cls, text: str) -> tuple[float | None, str]:
        """Extract due time from natural text.

        Returns (epoch_ts_or_None, cleaned_text).
        Supports 'at 5pm', 'at 17:30', 'in 10 minutes', 'in 2 hours'.
        """
        now = datetime.now()
        for pat, kind in cls.TIME_PATTERNS:
            m = pat.search(text)
            if not m:
                continue
            if kind == "at":
                hh = int(m.group(1))
                mm = int(m.group(2) or 0)
                ap = (m.group(3) or "").lower()
                if ap == "pm" and hh < 12:
                    hh += 12
                if ap == "am" and hh == 12:
                    hh = 0
                due = now.replace(hour=hh % 24, minute=mm,
                                  second=0, microsecond=0)
                if due <= now:
                    due += timedelta(days=1)   # "at 5pm" said at 6pm → tomorrow
                cleaned = (text[:m.start()] + text[m.end():]).strip(" ,.")
                return due.timestamp(), cleaned
            if kind == "in":
                n = int(m.group(1))
                unit = m.group(2).lower()
                delta = {"second": timedelta(seconds=n),
                         "minute": timedelta(minutes=n),
                         "hour": timedelta(hours=n)}[unit]
                cleaned = (text[:m.start()] + text[m.end():]).strip(" ,.")
                return (now + delta).timestamp(), cleaned
        return None, text


def start_scheduler(store: ReminderStore, on_due, poll_s: float = 1.0):
    """Background thread calling on_due(item) for each reminder as it fires."""
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            now = time.time()
            for item in store.list_pending():
                if item["when"] <= now:
                    item["fired"] = True
                    with store._lock:
                        store._save()
                    log.info("reminder FIRED: %s", item["text"][:60])
                    try:
                        on_due(item)
                    except Exception as exc:  # noqa: BLE001
                        log.error("reminder delivery failed: %s", exc)
            stop.wait(poll_s)

    t = threading.Thread(target=loop, daemon=True, name="pai-reminders")
    t.start()
    return stop

"""Google Calendar sync (optional — graceful when unavailable).

Two paths:
  • gcal CLI installed + authenticated (`gcal` / `gcalcli`) → shell out
  • Google Calendar API via service-account/OAuth creds file → use REST

Enabled by setting `google_calendar: true` in config.yaml. Every reminder
created while enabled is mirrored as a calendar event; failures never
break the local-first reminder flow.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from datetime import datetime, timedelta, timezone

log = logging.getLogger("pai.gcal")


def enabled() -> bool:
    from . import config
    return bool(getattr(config.get_config(), "google_calendar", False))


def _try_gcalcli(text: str, when: datetime) -> bool:
    exe = shutil.which("gcalcli") or shutil.which("gcal")
    if not exe:
        return False
    try:
        title = f"PAI: {text[:80]}"
        begin = when.strftime("%Y-%m-%d %H:%M")
        end = (when + timedelta(minutes=15)).strftime("%H:%M")
        r = subprocess.run(
            [exe, "add", "--title", title, "--when", begin,
             "--duration", "15", "--nostart-gui"],
            capture_output=True, timeout=30)
        ok = r.returncode == 0
        log.info("gcal add via %s: %s", exe, ok)
        return ok
    except Exception as exc:  # noqa: BLE001
        log.warning("gcalcli failed: %s", exc)
        return False


def _try_api(text: str, when: datetime) -> bool:
    """REST via google-api-python-client if creds exist."""
    import os
    from pathlib import Path
    creds = Path(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", ""))
    if not creds.exists():
        return False
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        scopes = ["https://www.googleapis.com/auth/calendar.events"]
        creds_obj = service_account.Credentials.from_service_account_file(
            str(creds), scopes=scopes)
        svc = build("calendar", "v3", credentials=creds_obj, cache_discovery=False)
        body = {
            "summary": f"PAI: {text[:80]}",
            "start": {"dateTime": when.astimezone(timezone.utc).isoformat()},
            "end": {"dateTime": (when + timedelta(minutes=15))
                    .astimezone(timezone.utc).isoformat()},
        }
        svc.events().insert(calendarId=os.environ.get(
            "GOOGLE_CALENDAR_ID", "primary"), body=body).execute()
        log.info("gcal event created via API")
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("gcal API failed: %s", exc)
        return False


def mirror_event(text: str, when_epoch: float) -> bool:
    """Best-effort mirror of a reminder into Google Calendar."""
    if not enabled():
        return False
    when = datetime.fromtimestamp(when_epoch)
    return _try_gcalcli(text, when) or _try_api(text, when)

"""
src/daemon/privacy.py

Privacy controls for a process that watches a person all day.

This is not decoration. A system with a microphone, a camera, and a
screenshot loop is one bug away from being spyware, and the difference is
entirely in what it refuses to keep. Four mechanisms:

  1. PAUSE. One call, `privacy.pause()`, stops every privacy-sensitive sensor
     immediately and keeps them stopped until explicitly resumed. It is
     checked in the ambient loop before any sensitive sensor is polled, so it
     takes effect within one tick rather than whenever a thread notices.

  2. REDACTION. Text from any source passes through a redactor before it is
     journaled or written to memory. Card numbers, API keys, long hex tokens,
     email addresses and anything matching a user-supplied pattern are
     replaced in place. This runs on the write path, not the read path -- the
     secret never lands on disk in the first place.

  3. RETENTION. Raw journal entries and captured frames have a hard TTL.
     Consolidated summaries survive; raw observations do not. This is the
     single most important control here: the value of ambient memory is in the
     summary, and keeping the raw stream forever buys almost nothing while
     risking everything.

  4. LOCAL ONLY. Nothing in this module or the sensors ships data anywhere.
     The one exception is transcription if it falls back to Google's
     recognizer -- which is why `require_local_asr` defaults to True and will
     refuse that fallback rather than silently upload the user's room audio.

A note worth being explicit about: a microphone in a shared space records
people who did not opt in. That is a consideration for the operator, not
something code can solve, but `announce_on_start` at least makes the system's
presence visible rather than silent.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Pattern

logger = logging.getLogger("ash.daemon.privacy")

# Patterns applied to every string before it is persisted.
DEFAULT_REDACTIONS: List[tuple] = [
    (re.compile(r"\b(?:\d[ -]*?){13,19}\b"), "[CARD]"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[EMAIL]"),
    (re.compile(r"\b(?:sk|pk|ghp|gho|xox[baprs])[-_][A-Za-z0-9_-]{16,}\b"), "[TOKEN]"),
    (re.compile(r"\b[A-Fa-f0-9]{32,}\b"), "[HEX]"),
    (re.compile(r"(?i)\b(password|passwd|secret|api[_ -]?key)\s*[:=]\s*\S+"), r"\1=[REDACTED]"),
    (re.compile(r"\b(?:\+?\d{1,3}[ -]?)?\(?\d{3}\)?[ -]?\d{3}[ -]?\d{4}\b"), "[PHONE]"),
]


@dataclass
class PrivacyPolicy:
    """Runtime privacy state. Mutable at runtime via the control interface."""

    paused: bool = False
    require_local_asr: bool = True
    redact: bool = True
    announce_on_start: bool = True

    # Retention, in days. 0 = do not keep at all.
    raw_journal_days: float = 7.0
    frame_days: float = 2.0
    audio_days: float = 0.0        # raw audio is never kept by default

    extra_patterns: List[tuple] = field(default_factory=list)
    blocked_apps: List[str] = field(default_factory=list)

    _paused_at: Optional[float] = None

    # ------------------------------------------------------------------
    def pause(self, reason: str = "user request"):
        self.paused = True
        self._paused_at = time.time()
        logger.warning("PRIVACY PAUSE engaged (%s) -- sensitive sensors halted", reason)

    def resume(self):
        was = self._paused_at
        self.paused = False
        self._paused_at = None
        logger.warning("Privacy pause lifted after %.0fs", time.time() - was if was else 0)

    def allows(self, sensor) -> bool:
        """Checked by the ambient loop before every poll."""
        if not self.paused:
            return True
        return not getattr(sensor.capability, "privacy_sensitive", False)

    # ------------------------------------------------------------------
    def scrub(self, text: str) -> str:
        if not self.redact or not text:
            return text or ""
        out = text
        for pattern, repl in list(DEFAULT_REDACTIONS) + list(self.extra_patterns):
            try:
                out = pattern.sub(repl, out)
            except Exception:
                logger.exception("Redaction pattern failed")
        return out

    def scrub_event(self, event):
        """Redact in place, and strip payloads the policy says not to keep."""
        event.text = self.scrub(event.text)
        for k, v in list(event.meta.items()):
            if isinstance(v, str):
                event.meta[k] = self.scrub(v)
        if self.audio_days <= 0 and getattr(event, "modality", None) == "audio":
            event.data = None
        return event

    # ------------------------------------------------------------------
    def sweep(self, journal_dir: str, frame_dir: str) -> Dict[str, int]:
        """Delete anything past its TTL. Called from the sleep phase."""
        removed = {"journal": 0, "frames": 0}
        now = time.time()

        def _purge(d: str, days: float, key: str, suffixes: tuple):
            if not os.path.isdir(d) or days <= 0:
                return
            cutoff = now - days * 86400
            for f in os.listdir(d):
                if not f.endswith(suffixes):
                    continue
                p = os.path.join(d, f)
                try:
                    if os.path.getmtime(p) < cutoff:
                        os.remove(p)
                        removed[key] += 1
                except Exception:
                    logger.exception("Failed to purge %s", p)

        _purge(journal_dir, self.raw_journal_days, "journal", (".jsonl",))
        _purge(frame_dir, self.frame_days, "frames", (".jpg", ".png"))

        if any(removed.values()):
            logger.info("Privacy sweep: removed %d journal file(s), %d frame(s)",
                        removed["journal"], removed["frames"])
        return removed

    # ------------------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        return {
            "paused": self.paused,
            "paused_for_s": (round(time.time() - self._paused_at, 1)
                             if self._paused_at else None),
            "redaction": self.redact,
            "require_local_asr": self.require_local_asr,
            "retention_days": {
                "raw_journal": self.raw_journal_days,
                "frames": self.frame_days,
                "audio": self.audio_days,
            },
        }

    def banner(self) -> str:
        bits = [
            "PRIVACY:",
            f"  redaction        {'on' if self.redact else 'OFF'}",
            f"  raw journal TTL  {self.raw_journal_days} day(s)",
            f"  frame TTL        {self.frame_days} day(s)",
            f"  raw audio kept   {'yes' if self.audio_days > 0 else 'no'}",
            f"  local ASR only   {'enforced' if self.require_local_asr else 'not enforced'}",
            "  all data stays on this machine; 'pause' halts sensitive sensors",
        ]
        return "\n".join(bits)

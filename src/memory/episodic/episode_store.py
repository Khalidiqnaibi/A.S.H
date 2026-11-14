# episode_store.py

from dataclasses import asdict, fields
import json
import os

from ..paths import memory_file
from typing import Dict, List
from .episode_model import Episode

_EPISODE_FIELD_NAMES = {f.name for f in fields(Episode)}


class EpisodeStore:
    """
    Storage format: JSONL (one Episode per line), not a single JSON
    blob. add() used to call a full _save() (json.dump of the entire
    dict) on every single call -- O(n) per write, and it only gets
    worse as episodic memory grows into the hundreds/thousands of
    entries. JSONL makes add() an O(1) append: write one line, done.

    Full rewrites (json.dump of everything) still happen, but only
    where they're actually needed: prune() has to rewrite the file
    anyway since it's removing rows, so that's where compaction and
    any pending access_count/last_accessed updates get persisted --
    see EpisodicMemory.prune() and mark_accessed() below.
    """

    def __init__(self, path=None):
        self.path = memory_file("episodic_memory.jsonl", path)
        self.episodes: Dict[str, Episode] = {}
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            # Backward-compat: look for the old single-JSON-blob file
            # (default name from before this change) and migrate it.
            legacy_path = self.path.rsplit(".", 1)[0] + ".json"
            if os.path.exists(legacy_path):
                self._migrate_legacy_json(legacy_path)
            return

        bad = 0
        with open(self.path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    self._add_from_dict(json.loads(line))
                except Exception:               # a line torn by a crash or a full disk: skip it, keep the rest
                    bad += 1
        if bad:
            import logging
            logging.getLogger("ash.memory.episodic").warning("Skipped %d unreadable line(s) in %s", bad, self.path)

    def _add_from_dict(self, data: dict):
        # Filter to known Episode fields only, so a schema change (a
        # field added/removed later) never crashes loading old rows --
        # unknown keys are dropped, missing keys fall back to the
        # dataclass field defaults (e.g. access_count=0 for episodes
        # written before that field existed).
        filtered = {k: v for k, v in data.items() if k in _EPISODE_FIELD_NAMES}
        ep = Episode(**filtered)
        self.episodes[ep.episode_id] = ep

    def _migrate_legacy_json(self, legacy_path: str):
        """One-time migration from the old single-JSON-blob format to
        JSONL. Reads the old file, loads every episode (missing
        last_accessed/access_count fall back to Episode's dataclass
        defaults automatically), then writes them out as JSONL and
        leaves the old file in place untouched (renamed with .bak so
        nothing is silently lost if this migration needs re-running)."""
        with open(legacy_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for eid, data in raw.items():
            self._add_from_dict(data)
        self._save_all()
        os.rename(legacy_path, legacy_path + ".bak")

    def _save_all(self):
        """Full rewrite -- one JSON object per line. Used by prune()
        (which is removing rows and must rewrite anyway) and by the
        legacy migration above. NOT used by add() -- see append()."""
        from ..atomic import write_with

        def _rows(f):
            for ep in self.episodes.values():
                f.write(json.dumps(asdict(ep)) + "\n")
        write_with(self.path, _rows, "w")          # temp + fsync + replace

    def add(self, episode: Episode):
        """O(1): append one line, no full re-read/re-write."""
        self.episodes[episode.episode_id] = episode
        try:
            lead = ""
            if os.path.exists(self.path) and os.path.getsize(self.path) > 0:
                with open(self.path, "rb") as r:        # a torn last line has no newline: do not glue this one to it
                    r.seek(-1, os.SEEK_END)
                    if r.read(1) != b"\n":
                        lead = "\n"
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(lead + json.dumps(asdict(episode)) + "\n")
        except OSError as e:
            from ..atomic import log_io_failure
            import logging
            log_io_failure(logging.getLogger("ash.memory.episodic"), "an episode", e)

    def mark_accessed(self, episodes: List[Episode]):
        """Episodes are already mutated in place by the caller
        (episodic_manager.retrieve() bumps last_accessed/access_count
        directly on the Episode objects before calling this). This
        just keeps self.episodes pointing at the same objects --
        intentionally does NOT write to disk. Access-count updates
        live in RAM only until the next prune() sweep does a full
        _save_all(), which persists whatever's accumulated by then.
        Trade-off: an access bump can be lost on a crash before the
        next sweep -- acceptable, since access_count is a pruning
        heuristic, not state anything depends on being exact."""
        for ep in episodes:
            self.episodes[ep.episode_id] = ep

    def all(self) -> List[Episode]:
        return list(self.episodes.values())
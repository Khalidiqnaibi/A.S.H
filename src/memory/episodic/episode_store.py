# episode_store.py

from dataclasses import asdict
import json
from typing import Dict, List
from .episode_model import Episode


class EpisodeStore:

    def __init__(self, path="episodic_memory.json"):
        self.path = path
        self.episodes: Dict[str, Episode] = {}
        self._load()

    def _load(self):
        try:
            with open(self.path, "r") as f:
                raw = json.load(f)
                for eid, data in raw.items():
                    self.episodes[eid] = Episode(**data)
        except FileNotFoundError:
            self.episodes = {}

    def _save(self):
        with open(self.path, "w") as f:
            json.dump({eid: asdict(ep) for eid, ep in self.episodes.items()}, f, indent=2)

    def add(self, episode: Episode):
        self.episodes[episode.episode_id] = episode
        self._save()

    def all(self) -> List[Episode]:
        return list(self.episodes.values())
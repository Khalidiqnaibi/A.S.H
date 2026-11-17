"""
src/daemon/identity.py

What ASH calls the person. It is a setting (console settings -> "Your name", saved as daemon.json identity.user_name), so anyone who
runs ASH can set their own; until they do, ASH uses the developer's DEV_NAME if one is set, else a plain "User", which the prompt
reads as "the person you work with". The name used before is kept as an alias: what was said under it is still found.
"""

import os


def default_user() -> str:
    n = (os.getenv("ASH_USER_NAME") or os.getenv("DEV_NAME") or "").strip().strip("\"'")
    return n.title() if n else "User"


class NameMixin:
    user: str
    user_aliases: set

    def set_user_name(self, name: str) -> str:
        """Change what ASH calls the person, everywhere it is held (the brain too). Returns the name now in use."""
        name = " ".join(str(name or "").split())[:40]
        if not name:
            return self.user
        old = self.user
        if old and old.lower() != name.lower() and old.lower() not in ("user", "unknown"):
            self.user_aliases.add(old)
        self.user = name
        b = getattr(self, "brain", None)
        if b is not None:
            b.user = name
            b.user_aliases = set(self.user_aliases)
        return name

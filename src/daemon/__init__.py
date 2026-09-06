"""
ASH daemon: the always-on runtime.

    AmbientRuntime  -- the 24h loop (poll -> journal -> attend -> observe/respond)
    AttentionGate   -- observe vs respond; direct address always wins
    Journal         -- raw event log + deterministic rollup into episodic memory
    PrivacyPolicy   -- pause, redaction, retention, local-only
    service         -- config, single instance, signals, control socket, watchdog
"""

from .attention import AttentionGate, Disposition, AttentionDecision
from .journal import Journal
from .privacy import PrivacyPolicy
from .ambient import AmbientRuntime, AmbientConfig
from .service import (
    load_config, DEFAULT_CONFIG, SingleInstance, ControlServer,
    control_client, setup_logging, Watchdog, install_signal_handlers,
)

__all__ = [
    "AttentionGate", "Disposition", "AttentionDecision", "Journal",
    "PrivacyPolicy", "AmbientRuntime", "AmbientConfig", "load_config",
    "DEFAULT_CONFIG", "SingleInstance", "ControlServer", "control_client",
    "setup_logging", "Watchdog", "install_signal_handlers",
]

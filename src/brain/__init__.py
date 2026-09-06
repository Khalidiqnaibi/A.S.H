"""
ASH brain layer -- a human-brain-inspired cognitive architecture.

    Homeostatic Modulator (limbic)   homeostasis.HomeostaticModulator
    Sensory Extractors               sensory.SensoryCortex
    Executive Core                   executive.ExecutiveCore
    Predictive Model                 predictive.PredictiveModel
    System 1 Policy (fast reflexes)  system1.System1Policy
    System 2 (deliberation)          system2.System2Deliberator
    VLA channel (embodied action)    vla.VLAChannel
    Episodic/Semantic Memory Engine  memory_engine.MemoryEngine
    Cognitive cycle                  brain.Brain

Fast/slow execution is `signals.Pathway` and is decided in
`executive.ExecutiveCore.gate()`.
"""

from .signals import (
    ActionClass, ActionProposal, BrainResponse, BrainTrace, Drives,
    Pathway, Percept, PerceptBundle, Prediction, Verdict, Vote,
)
from .sensory import SensoryCortex, TextExtractor, VisionExtractor
from .homeostasis import HomeostaticModulator
from .predictive import PredictiveModel, LatentForwardModel
from .system1 import System1Policy, Critic, classify_action
from .system2 import System2Deliberator
from .vla import VLAChannel
from .executive import ExecutiveCore, BoardMember, Constitution
from .memory_engine import MemoryEngine
from .brain import Brain
from .consolidation import consolidate

__all__ = [
    "ActionClass", "ActionProposal", "BrainResponse", "BrainTrace", "Drives",
    "Pathway", "Percept", "PerceptBundle", "Prediction", "Verdict", "Vote",
    "SensoryCortex", "TextExtractor", "VisionExtractor",
    "HomeostaticModulator", "PredictiveModel", "LatentForwardModel",
    "System1Policy", "Critic", "classify_action", "System2Deliberator",
    "VLAChannel", "ExecutiveCore", "BoardMember", "Constitution",
    "MemoryEngine", "Brain", "consolidate",
]

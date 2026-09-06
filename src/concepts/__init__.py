"""
ASH associative layer: concept-based representation with Hebbian rewiring.

    model        Concept, Synapse (fast/slow weights), Activation
    network      k-WTA activation, spreading, resonance growth, priming
    plasticity   three-factor Hebbian: PMI gate, Oja bound, homeostatic
                 scaling, decay + pruning, sleep consolidation with rollback
    bindings     seeding, board proposer, associative recall, regression guard

The graph competes with the embedding classifier at the Executive board; it
does not replace it. See bindings.py for why that is the brain-faithful
arrangement rather than a hedge.
"""

from .model import (
    Activation, Concept, Origin, PROTECTED_ORIGINS, Synapse, SynType, TYPE_RULES,
)
from .hippocampus import Hippocampus, Trace
from .neuromod import Neuromodulators, NeuroState
from .sleep import SleepCycle, SleepConfig
from .network import ConceptNetwork
from .plasticity import Plasticity, PlasticityConfig
from .bindings import (
    ConceptMemoryIndex, ConceptProposer, ConceptSystem, RoutingEvaluator,
    seed_from_ash,
)

__all__ = [
    "Activation", "Concept", "Origin", "PROTECTED_ORIGINS", "Synapse",
    "SynType", "TYPE_RULES", "Hippocampus", "Trace",
    "Neuromodulators", "NeuroState", "SleepCycle", "SleepConfig",
    "ConceptNetwork", "Plasticity", "PlasticityConfig",
    "ConceptMemoryIndex", "ConceptProposer", "ConceptSystem",
    "RoutingEvaluator", "seed_from_ash",
]

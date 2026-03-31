"""
graph-glad-hep: Graph-level anomaly detection with contrastive learning
for LHC new physics searches (GSoC 2026 / ML4SCI GENIE).
"""
from .graph_builder import build_event_graph, MockEventGenerator
from .encoder import GraphEncoder, DualEncoder
from .losses import reconstruction_loss, contrastive_loss, representation_error, total_loss
from .decorrelation import dist_corr
from .evaluate import sic_curve, evaluate
from .model import GraphGLADC

__all__ = [
    "build_event_graph", "MockEventGenerator", "GraphEncoder", "DualEncoder",
    "reconstruction_loss", "contrastive_loss", "representation_error", "total_loss",
    "dist_corr", "sic_curve", "evaluate", "GraphGLADC",
]
__version__ = "0.1.0"

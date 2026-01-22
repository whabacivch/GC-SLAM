"""
Backend package for FL-SLAM.

Information-geometric fusion and state estimation.

Subpackages (flattened):
- gaussian_info/gaussian_geom/information_distances
- adaptive/birth/nig/process_noise/timestamp/weights
- dirichlet_routing
"""

from fl_slam_poc.backend.adaptive import AdaptiveParameter, OnlineStats
from fl_slam_poc.backend.timestamp import TimeAlignmentModel
from fl_slam_poc.backend.birth import StochasticBirthModel
from fl_slam_poc.backend.process_noise import AdaptiveProcessNoise
from fl_slam_poc.backend.nig import (
    NIGModel,
    NIG_PRIOR_KAPPA,
    NIG_PRIOR_ALPHA,
    NIG_PRIOR_BETA,
)
from fl_slam_poc.backend.weights import combine_independent_weights
from fl_slam_poc.backend.dirichlet_routing import DirichletRoutingModule

__all__ = [
    # Parameters
    "AdaptiveParameter",
    "OnlineStats",
    "TimeAlignmentModel",
    "StochasticBirthModel",
    "AdaptiveProcessNoise",
    "NIGModel",
    "NIG_PRIOR_KAPPA",
    "NIG_PRIOR_ALPHA",
    "NIG_PRIOR_BETA",
    "combine_independent_weights",
    # Routing
    "DirichletRoutingModule",
]

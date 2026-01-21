"""
Backend package for FL-SLAM.

Information-geometric fusion and state estimation.

Subpackages:
- fusion/: Gaussian and multi-modal fusion operators
- parameters/: Adaptive parameter estimation models
- routing/: Dirichlet-categorical routing for compositional inference
"""

from fl_slam_poc.backend.parameters import (
    AdaptiveParameter,
    OnlineStats,
    TimeAlignmentModel,
    StochasticBirthModel,
    AdaptiveProcessNoise,
    NIGModel,
    NIG_PRIOR_KAPPA,
    NIG_PRIOR_ALPHA,
    NIG_PRIOR_BETA,
    combine_independent_weights,
)

from fl_slam_poc.backend.routing import DirichletRoutingModule

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

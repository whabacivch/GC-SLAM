"""
Routing modules for compositional SLAM backend.

Implements Dirichlet-categorical routing with:
- Soft association (no hard gating)
- Frobenius retention (cubic contraction)
- Hellinger shift monitoring
"""

from fl_slam_poc.backend.routing.dirichlet_routing import DirichletRoutingModule

__all__ = ["DirichletRoutingModule"]

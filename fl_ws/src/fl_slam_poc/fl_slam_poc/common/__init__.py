"""
Common package for Geometric Compositional SLAM v2.

Contains shared utilities:
- belief: Gaussian belief representations (BeliefGaussianInfo)
- certificates: Certificate audit trail (CertBundle)
- constants: Configuration constants
- geometry/: SE(3) operations (JAX and NumPy)
- jax_init: JAX initialization
- primitives: Branch-free numeric primitives
"""

from fl_slam_poc.common import constants
from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common.belief import BeliefGaussianInfo
from fl_slam_poc.common.certificates import CertBundle

__all__ = [
    "constants",
    "jax",
    "jnp",
    "BeliefGaussianInfo",
    "CertBundle",
]

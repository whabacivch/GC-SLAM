"""
Common JAX Initialization Module.

This module initializes JAX once at import time with GPU configuration.
All other modules should import JAX from here instead of importing jax directly
to ensure consistent initialization and avoid "PJRT_Api already exists" errors.

Usage:
    from fl_slam_poc.common.jax_init import jax, jnp
    
    # JAX is already configured for GPU and x64 precision
    devices = jax.devices()  # Safe to call, JAX is already initialized
"""

from __future__ import annotations

import os

# Configure JAX environment variables BEFORE importing JAX.
# This must happen at module import time, before any JAX operations.
#
# IMPORTANT: Use "cuda" (not "gpu") to avoid JAX attempting ROCm first.
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# Import JAX (this will initialize PJRT with the environment variables set above)
import jax
import jax.numpy as jnp

# Configure JAX for x64 precision (required for numerical stability)
jax.config.update("jax_enable_x64", True)

# JAX is now initialized. Other modules can safely import from here.
__all__ = ["jax", "jnp"]

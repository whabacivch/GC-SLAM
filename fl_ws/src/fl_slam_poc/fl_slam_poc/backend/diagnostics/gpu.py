"""
GPU availability checking and IMU kernel warmup.

Handles early GPU detection and JAX kernel compilation warmup.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fl_slam_poc.common.jax_init import jnp

if TYPE_CHECKING:
    from fl_slam_poc.backend.backend_node import FLBackend


def check_gpu_availability(backend: "FLBackend") -> None:
    """
    Check GPU availability early at startup.
    
    Raises RuntimeError if GPU is required but not available.
    This ensures failures happen during node initialization, not when
    the first IMU message arrives.
    
    Args:
        backend: Backend node instance
        
    Raises:
        RuntimeError: If GPU is not available or JAX is not installed
    """
    try:
        import jax
        devices = jax.devices()
        has_gpu = any(d.platform == "gpu" for d in devices)
        
        if not has_gpu:
            available_devices = [f"{d.platform}:{d.device_kind}" for d in devices]
            raise RuntimeError(
                f"JAX GPU backend is required for IMU fusion but not available.\n"
                f"Available JAX devices: {available_devices}\n"
                f"To fix: Ensure CUDA is installed and JAX can detect GPU devices.\n"
                f"Note: 15D state with IMU fusion is always enabled - GPU is required."
            )
        
        backend.get_logger().info(
            f"GPU availability confirmed: {[f'{d.platform}:{d.device_kind}' for d in devices if d.platform == 'gpu']}"
        )
    except ImportError:
        raise RuntimeError(
            "JAX is required for IMU fusion but not installed.\n"
            "Install JAX with GPU support: pip install 'jax[cuda12]' or 'jax[cuda11]'"
        )
    except RuntimeError:
        # Re-raise RuntimeError (GPU not available)
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Failed to check GPU availability: {exc}\n"
            f"This may indicate a JAX configuration issue."
        ) from exc


def warmup_imu_kernel(backend: "FLBackend", gravity: list[float]) -> None:
    """
    Warm up JAX IMU kernel compilation to avoid first-call latency.
    
    Args:
        backend: Backend node instance
        gravity: Gravity vector (3,) for IMU integration
    """
    try:
        from fl_slam_poc.backend.math.imu_kernel import imu_batched_projection_kernel
        
        # Minimal dummy data for warmup
        anchor_mus = jnp.zeros((1, 15), dtype=jnp.float64)
        anchor_covs = jnp.eye(15, dtype=jnp.float64)[None, :, :] * 1e-3
        current_mu = jnp.zeros((15,), dtype=jnp.float64)
        current_cov = jnp.eye(15, dtype=jnp.float64) * 1e-3
        routing_weights = jnp.array([1.0], dtype=jnp.float64)
        imu_stamps = jnp.array([0.0, 0.001], dtype=jnp.float64)
        imu_accel = jnp.zeros((2, 3), dtype=jnp.float64)
        imu_gyro = jnp.zeros((2, 3), dtype=jnp.float64)
        imu_valid = jnp.array([True, True], dtype=bool)
        R_imu = jnp.eye(9, dtype=jnp.float64) * 1e-3
        R_nom = R_imu.copy()
        
        imu_batched_projection_kernel(
            anchor_mus=anchor_mus,
            anchor_covs=anchor_covs,
            current_mu=current_mu,
            current_cov=current_cov,
            routing_weights=routing_weights,
            imu_stamps=imu_stamps,
            imu_accel=imu_accel,
            imu_gyro=imu_gyro,
            imu_valid=imu_valid,
            R_imu=R_imu,
            R_nom=R_nom,
            dt_total=0.001,
            gravity=jnp.array(gravity, dtype=jnp.float64),
        )
        backend.get_logger().info("IMU kernel warmup complete.")
    except RuntimeError as exc:
        # Re-raise RuntimeError (GPU not available) - fail fast
        backend.get_logger().error(f"IMU kernel warmup failed: {exc}")
        raise
    except Exception as exc:
        # Other exceptions (e.g., compilation errors) are warnings
        backend.get_logger().warn(f"IMU kernel warmup failed (non-fatal): {exc}")

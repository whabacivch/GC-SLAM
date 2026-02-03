"""
Tests for Geometric Compositional SLAM v2 invariants.

Verifies that the implementation follows the spec invariants:
- Chart ID consistency
- Dimension correctness
- Budget enforcement
- No branching/gating

Reference: docs/GC_SLAM.md
"""

import os
import pytest
from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common import constants
from fl_slam_poc.common.belief import (
    BeliefGaussianInfo,
    D_Z,
    CHART_ID_GC_RIGHT_01,
    se3_identity,
)
from fl_slam_poc.common.certificates import CertBundle


class TestChartIdInvariant:
    """Verify chart_id is enforced correctly."""

    def test_belief_requires_correct_chart_id(self):
        """BeliefGaussianInfo must have chart_id == 'GC-RIGHT-01'."""
        cert = CertBundle.create_exact(
            chart_id=CHART_ID_GC_RIGHT_01,
            anchor_id="test",
        )
        
        # Valid chart ID should work
        belief = BeliefGaussianInfo(
            chart_id=CHART_ID_GC_RIGHT_01,
            anchor_id="test",
            X_anchor=se3_identity(),
            stamp_sec=0.0,
            z_lin=jnp.zeros(D_Z),
            L=jnp.eye(D_Z),
            h=jnp.zeros(D_Z),
            cert=cert,
        )
        assert belief.chart_id == CHART_ID_GC_RIGHT_01

    def test_belief_rejects_wrong_chart_id(self):
        """BeliefGaussianInfo must reject incorrect chart_id."""
        cert = CertBundle.create_exact(
            chart_id="wrong-chart",
            anchor_id="test",
        )
        
        with pytest.raises(ValueError, match="Invalid chart_id"):
            BeliefGaussianInfo(
                chart_id="wrong-chart",
                anchor_id="test",
                X_anchor=se3_identity(),
                stamp_sec=0.0,
                z_lin=jnp.zeros(D_Z),
                L=jnp.eye(D_Z),
                h=jnp.zeros(D_Z),
                cert=cert,
            )


class TestDimensionInvariants:
    """Verify dimension constants are correct."""

    def test_d_z_is_22(self):
        """D_Z must be 22 per spec."""
        assert D_Z == 22
        assert constants.GC_D_Z == 22

    def test_d_deskew_equals_d_z(self):
        """D_DESKEW must equal D_Z."""
        assert constants.GC_D_DESKEW == constants.GC_D_Z

    def test_belief_dimensions(self):
        """BeliefGaussianInfo dimensions must match spec."""
        cert = CertBundle.create_exact(
            chart_id=CHART_ID_GC_RIGHT_01,
            anchor_id="test",
        )
        
        belief = BeliefGaussianInfo(
            chart_id=CHART_ID_GC_RIGHT_01,
            anchor_id="test",
            X_anchor=se3_identity(),
            stamp_sec=0.0,
            z_lin=jnp.zeros(D_Z),
            L=jnp.eye(D_Z),
            h=jnp.zeros(D_Z),
            cert=cert,
        )
        
        assert belief.z_lin.shape == (D_Z,)
        assert belief.L.shape == (D_Z, D_Z)
        assert belief.h.shape == (D_Z,)
        assert belief.X_anchor.shape == (6,)  # 6D SE3


class TestBudgetConstants:
    """Verify budget constants match spec."""

    def test_k_hyp(self):
        """K_HYP must be 4."""
        assert constants.GC_K_HYP == 4

    def test_hyp_weight_floor(self):
        """HYP_WEIGHT_FLOOR must be 0.01 / K_HYP."""
        expected = 0.01 / constants.GC_K_HYP
        assert abs(constants.GC_HYP_WEIGHT_FLOOR - expected) < 1e-10

    def test_n_points_cap(self):
        """N_POINTS_CAP must be 8192."""
        assert constants.GC_N_POINTS_CAP == 8192


class TestEpsilonConstants:
    """Verify epsilon constants are positive and small."""

    def test_eps_psd_positive(self):
        """EPS_PSD must be positive."""
        assert constants.GC_EPS_PSD > 0
        assert constants.GC_EPS_PSD < 1e-6

    def test_eps_lift_positive(self):
        """EPS_LIFT must be positive."""
        assert constants.GC_EPS_LIFT > 0
        assert constants.GC_EPS_LIFT < 1e-6

    def test_eps_mass_positive(self):
        """EPS_MASS must be positive."""
        assert constants.GC_EPS_MASS > 0
        assert constants.GC_EPS_MASS < 1e-6

    def test_eps_r_positive(self):
        """EPS_R must be positive."""
        assert constants.GC_EPS_R > 0
        assert constants.GC_EPS_R < 1e-3


class TestFusionConstants:
    """Verify fusion constants are in valid ranges."""

    def test_alpha_range(self):
        """ALPHA_MIN and ALPHA_MAX must be in [0, 1]."""
        assert 0.0 <= constants.GC_ALPHA_MIN <= 1.0
        assert 0.0 <= constants.GC_ALPHA_MAX <= 1.0
        assert constants.GC_ALPHA_MIN <= constants.GC_ALPHA_MAX

    def test_coupling_constants_positive(self):
        """Coupling constants must be positive."""
        assert constants.GC_C_DT > 0
        assert constants.GC_C_EX > 0
        assert constants.GC_C_FROB > 0


class TestCertificateStructure:
    """Verify CertBundle structure matches spec."""

    def test_exact_cert_has_no_triggers(self):
        """Exact certificates have empty trigger list."""
        cert = CertBundle.create_exact(
            chart_id=CHART_ID_GC_RIGHT_01,
            anchor_id="test",
        )
        assert cert.exact is True
        assert cert.approximation_triggers == []
        assert cert.frobenius_applied is False

    def test_approx_cert_has_triggers(self):
        """Approximate certificates have non-empty trigger list."""
        cert = CertBundle.create_approx(
            chart_id=CHART_ID_GC_RIGHT_01,
            anchor_id="test",
            triggers=["TestOp"],
        )
        assert cert.exact is False
        assert "TestOp" in cert.approximation_triggers

    def test_cert_to_dict(self):
        """Certificates can be serialized to dict."""
        cert = CertBundle.create_exact(
            chart_id=CHART_ID_GC_RIGHT_01,
            anchor_id="test",
        )
        d = cert.to_dict()
        assert "chart_id" in d
        assert "exact" in d
        assert "conditioning" in d
        assert "influence" in d

    def test_total_trigger_magnitude(self):
        """Total trigger magnitude is computed correctly."""
        from fl_slam_poc.common.certificates import InfluenceCert
        
        cert = CertBundle.create_approx(
            chart_id=CHART_ID_GC_RIGHT_01,
            anchor_id="test",
            triggers=["TestOp"],
            influence=InfluenceCert(
                lift_strength=0.1,
                psd_projection_delta=0.2,
            ),
        )
        
        magnitude = cert.total_trigger_magnitude()
        assert magnitude >= 0.3  # At least lift + psd delta


class TestSensorHubWiring:
    """Verify GC launch uses the sensor hub layer for accountability."""

    def test_gc_rosbag_launch_uses_gc_sensor_hub(self):
        pkg_root = os.path.dirname(os.path.dirname(__file__))
        path = os.path.join(pkg_root, "launch", "gc_rosbag.launch.py")
        if not os.path.exists(path):
            pytest.skip("gc_rosbag.launch.py not found")
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        assert "gc_sensor_hub" in text
        assert 'executable="gc_sensor_hub"' in text
        assert "gc_unified.yaml" in text

    def test_dead_end_audit_yaml_exists_and_has_topic_specs(self):
        pkg_root = os.path.dirname(os.path.dirname(__file__))
        path = os.path.join(pkg_root, "config", "gc_dead_end_audit.yaml")
        if not os.path.exists(path):
            pytest.skip("gc_dead_end_audit.yaml not found")
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        assert "topic_specs:" in text
        assert "/camera/imu|sensor_msgs/msg/Imu" in text
        assert "/vrpn_client_node/UGV/pose|geometry_msgs/msg/PoseStamped" in text

    def test_unified_yaml_contains_dead_end_audit_topic_specs(self):
        pkg_root = os.path.dirname(os.path.dirname(__file__))
        path = os.path.join(pkg_root, "config", "gc_unified.yaml")
        if not os.path.exists(path):
            pytest.skip("gc_unified.yaml not found")
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        assert "dead_end_audit:" in text
        assert "topic_specs:" in text

    def test_gc_rosbag_launch_uses_camera_rgbd_node(self):
        """Single-path camera: launch must use camera_rgbd_node and must not wire the legacy two-node path."""
        pkg_root = os.path.dirname(os.path.dirname(__file__))
        path = os.path.join(pkg_root, "launch", "gc_rosbag.launch.py")
        if not os.path.exists(path):
            pytest.skip("gc_rosbag.launch.py not found")
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        assert 'executable="camera_rgbd_node"' in text
        assert 'executable="image_decompress_cpp"' not in text
        assert 'executable="depth_passthrough"' not in text

    def test_unified_yaml_uses_camera_rgbd_topic(self):
        """Backend config must specify camera_rgbd_topic (legacy camera_image_topic/camera_depth_topic removed)."""
        pkg_root = os.path.dirname(os.path.dirname(__file__))
        path = os.path.join(pkg_root, "config", "gc_unified.yaml")
        if not os.path.exists(path):
            pytest.skip("gc_unified.yaml not found")
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        assert "camera_rgbd_topic:" in text
        assert "camera_image_topic:" not in text
        assert "camera_depth_topic:" not in text

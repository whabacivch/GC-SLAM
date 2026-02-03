"""
Certificate schema tests for Geometric Compositional SLAM v2.

Reference: docs/GC_SLAM.md Section 2.3

These tests verify that:
1. All required cert fields exist with correct types
2. Aggregation preserves schema completeness
3. Determinism: same inputs -> same snapshot_id
"""

import pytest
import hashlib
import json

from fl_slam_poc.common.certificates import (
    CertBundle,
    ComputeCert,
    ScanIOCert,
    DeviceRuntimeCert,
    OverconfidenceCert,
    ConditioningCert,
    SupportCert,
    MismatchCert,
    ExcitationCert,
    InfluenceCert,
    ExpectedEffect,
    aggregate_certificates,
)


# =============================================================================
# Schema Completeness Tests
# =============================================================================


def test_certbundle_compute_schema_exists():
    """Verify ComputeCert has all required fields per spec §2.3."""
    cert = CertBundle.create_exact(chart_id="chart", anchor_id="anchor")
    assert hasattr(cert, "compute")
    assert isinstance(cert.compute, ComputeCert)
    assert isinstance(cert.compute.alloc_bytes_est, int)
    assert isinstance(cert.compute.largest_tensor_shape, tuple)
    assert len(cert.compute.largest_tensor_shape) == 2
    assert isinstance(cert.compute.segment_sum_k, int)
    assert isinstance(cert.compute.psd_projection_count, int)
    assert isinstance(cert.compute.chol_solve_count, int)

    assert isinstance(cert.compute.scan_io, ScanIOCert)
    assert isinstance(cert.compute.scan_io.scan_seq, int)
    assert isinstance(cert.compute.scan_io.scan_stamp_sec, float)
    assert isinstance(cert.compute.scan_io.streams, dict)

    assert isinstance(cert.compute.device_runtime, DeviceRuntimeCert)
    assert isinstance(cert.compute.device_runtime.host_sync_count_est, int)
    assert isinstance(cert.compute.device_runtime.device_to_host_bytes_est, int)
    assert isinstance(cert.compute.device_runtime.host_to_device_bytes_est, int)
    assert isinstance(cert.compute.device_runtime.jit_recompile_count, int)

    as_dict = cert.to_dict()
    assert "compute" in as_dict
    assert "scan_io" in as_dict["compute"]
    assert "device_runtime" in as_dict["compute"]


def test_certbundle_overconfidence_growth_sentinels():
    """Verify OverconfidenceCert has growth sentinel fields per spec §5.8."""
    cert = CertBundle.create_exact(chart_id="chart", anchor_id="anchor")
    assert hasattr(cert, "overconfidence")
    assert isinstance(cert.overconfidence, OverconfidenceCert)

    # Original fields
    assert hasattr(cert.overconfidence, "excitation_total")
    assert hasattr(cert.overconfidence, "ess_to_excitation")
    assert hasattr(cert.overconfidence, "cond_to_support")
    assert hasattr(cert.overconfidence, "dt_asymmetry")
    assert hasattr(cert.overconfidence, "z_to_xy_ratio")

    # Growth sentinels (new per spec)
    assert hasattr(cert.overconfidence, "ess_growth_rate")
    assert hasattr(cert.overconfidence, "excitation_growth_rate")
    assert hasattr(cert.overconfidence, "nullspace_energy_ratio")

    # Check types
    assert isinstance(cert.overconfidence.ess_growth_rate, float)
    assert isinstance(cert.overconfidence.excitation_growth_rate, float)
    assert isinstance(cert.overconfidence.nullspace_energy_ratio, float)

    # Check to_dict includes new fields
    oc_dict = cert.overconfidence.to_dict()
    assert "ess_growth_rate" in oc_dict
    assert "excitation_growth_rate" in oc_dict
    assert "nullspace_energy_ratio" in oc_dict


def test_certbundle_all_component_certs_present():
    """Verify CertBundle has all component certificates."""
    cert = CertBundle.create_exact(chart_id="chart", anchor_id="anchor")

    # Core fields
    assert hasattr(cert, "chart_id")
    assert hasattr(cert, "anchor_id")
    assert hasattr(cert, "exact")
    assert hasattr(cert, "approximation_triggers")
    assert hasattr(cert, "frobenius_applied")

    # Component certificates
    assert isinstance(cert.conditioning, ConditioningCert)
    assert isinstance(cert.support, SupportCert)
    assert isinstance(cert.mismatch, MismatchCert)
    assert isinstance(cert.excitation, ExcitationCert)
    assert isinstance(cert.influence, InfluenceCert)
    assert isinstance(cert.overconfidence, OverconfidenceCert)
    assert isinstance(cert.compute, ComputeCert)


def test_certbundle_to_dict_completeness():
    """Verify to_dict includes all required keys."""
    cert = CertBundle.create_approx(
        chart_id="test_chart",
        anchor_id="test_anchor",
        triggers=["test_trigger"],
        frobenius_applied=True,
    )

    d = cert.to_dict()

    # Required top-level keys
    required_keys = [
        "chart_id",
        "anchor_id",
        "exact",
        "approximation_triggers",
        "frobenius_applied",
        "conditioning",
        "support",
        "mismatch",
        "excitation",
        "influence",
        "overconfidence",
        "compute",
        "total_trigger_magnitude",
    ]
    for key in required_keys:
        assert key in d, f"Missing required key: {key}"

    # Check nested structures have expected keys
    assert "eig_min" in d["conditioning"]
    assert "eig_max" in d["conditioning"]
    assert "cond" in d["conditioning"]

    assert "ess_total" in d["support"]
    assert "support_frac" in d["support"]

    assert "ess_growth_rate" in d["overconfidence"]
    assert "excitation_growth_rate" in d["overconfidence"]
    assert "nullspace_energy_ratio" in d["overconfidence"]

    assert "alloc_bytes_est" in d["compute"]
    assert "scan_io" in d["compute"]
    assert "device_runtime" in d["compute"]


# =============================================================================
# Aggregation Tests
# =============================================================================


def test_certbundle_scan_seq_monotonic_when_present():
    """Verify aggregation picks highest scan_seq."""
    c1 = CertBundle.create_exact(chart_id="chart", anchor_id="anchor")
    c2 = CertBundle.create_exact(chart_id="chart", anchor_id="anchor")
    c1.compute.scan_io.scan_seq = 1
    c2.compute.scan_io.scan_seq = 2
    agg = aggregate_certificates([c1, c2])
    assert agg.compute.scan_io.scan_seq == 2


def test_aggregate_certificates_preserves_schema():
    """Verify aggregation produces valid schema."""
    c1 = CertBundle.create_exact(chart_id="chart", anchor_id="anchor")
    c2 = CertBundle.create_approx(
        chart_id="chart",
        anchor_id="anchor",
        triggers=["test"],
    )
    c1.overconfidence.ess_growth_rate = 0.1
    c2.overconfidence.excitation_growth_rate = 0.2

    agg = aggregate_certificates([c1, c2])

    # Check schema preserved
    assert hasattr(agg.overconfidence, "ess_growth_rate")
    assert hasattr(agg.overconfidence, "excitation_growth_rate")
    assert hasattr(agg.overconfidence, "nullspace_energy_ratio")

    # Check aggregation uses max
    assert agg.overconfidence.ess_growth_rate == 0.1
    assert agg.overconfidence.excitation_growth_rate == 0.2


def test_aggregate_empty_list():
    """Verify empty aggregation returns valid cert."""
    agg = aggregate_certificates([])
    assert agg.chart_id == "unknown"
    assert agg.anchor_id == "unknown"
    assert agg.exact is True


# =============================================================================
# Determinism Tests
# =============================================================================


def _cert_snapshot_id(cert: CertBundle) -> str:
    """Compute deterministic snapshot ID from cert."""
    d = cert.to_dict()
    # Sort keys for determinism
    json_str = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


def test_snapshot_determinism_exact():
    """Same inputs produce same snapshot_id for exact certs."""
    c1 = CertBundle.create_exact(chart_id="chart", anchor_id="anchor")
    c2 = CertBundle.create_exact(chart_id="chart", anchor_id="anchor")

    id1 = _cert_snapshot_id(c1)
    id2 = _cert_snapshot_id(c2)

    assert id1 == id2, "Determinism violated: same inputs -> different snapshot_id"


def test_snapshot_determinism_approx():
    """Same inputs produce same snapshot_id for approx certs."""
    c1 = CertBundle.create_approx(
        chart_id="chart",
        anchor_id="anchor",
        triggers=["test", "another"],
        frobenius_applied=True,
    )
    c2 = CertBundle.create_approx(
        chart_id="chart",
        anchor_id="anchor",
        triggers=["test", "another"],
        frobenius_applied=True,
    )

    id1 = _cert_snapshot_id(c1)
    id2 = _cert_snapshot_id(c2)

    assert id1 == id2, "Determinism violated: same inputs -> different snapshot_id"


def test_snapshot_different_inputs():
    """Different inputs produce different snapshot_id."""
    c1 = CertBundle.create_exact(chart_id="chart1", anchor_id="anchor")
    c2 = CertBundle.create_exact(chart_id="chart2", anchor_id="anchor")

    id1 = _cert_snapshot_id(c1)
    id2 = _cert_snapshot_id(c2)

    assert id1 != id2, "Different inputs should produce different snapshot_id"


# =============================================================================
# ExpectedEffect Tests
# =============================================================================


def test_expected_effect_schema():
    """Verify ExpectedEffect has required fields."""
    effect = ExpectedEffect(
        objective_name="test_objective",
        predicted=1.0,
        realized=0.9,
    )

    d = effect.to_dict()
    assert "objective_name" in d
    assert "predicted" in d
    assert "realized" in d
    assert d["objective_name"] == "test_objective"
    assert d["predicted"] == 1.0
    assert d["realized"] == 0.9

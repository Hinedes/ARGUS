"""SampledTrajectory: strict contract, transforms, CLI acceptance.

Tests that every filed requirement of the echos-argus v1 schema is enforced.
"""

import io
import json
import os
import sys
import tempfile

import numpy as np

from argus.acoustic_scene import Reflector, diamond_mics
from argus.motion import (
    SampledTrajectory, save_trajectory_npz, ConstantVelocity,
    _REQUIRED_KEYS_STRICT, _quat_multiply, _quat_to_rotmat,
    _slerp, _linear_interp, ECHOS_EMITTER_OFFSET, ECHOS_MIC_OFFSETS,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

MOCK_METADATA = {
    "schema": "echos-argus-trajectory",
    "version": 1,
    "physics_dt_s": 0.005,
    "control_dt_s": 0.01,
    "world_handedness": "right",
    "world_up_axis": "+Z",
    "body_forward_axis": "+X",
    "body_positive_y_semantics": "left",
    "quaternion_order": "wxyz",
    "quaternion_semantics": "active_body_to_world",
    "position_unit": "m",
    "time_unit": "s",
    "angular_unit": "rad",
    "physical_emitter_offset_body_m": list(ECHOS_EMITTER_OFFSET),
}


def _make_fixture_dict(n=100):
    """Return a dict of arrays matching the strict v1 schema."""
    rng = np.random.default_rng(42)
    t = np.linspace(0, 0.5, n)
    pos = np.column_stack([0.5 * t, 0.1 * np.sin(2 * t), 0.0 * t])
    # body quat: identity (no rotation)
    body_q = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    vel = np.tile(np.array([0.5, 0.0, 0.0]), (n, 1))
    omega_body = np.zeros((n, 3))
    omega_world = np.zeros((n, 3))
    gimbal_pitch = np.zeros(n)
    gimbal_body = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    gimbal_world = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    beam = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
    emitter = pos + ECHOS_EMITTER_OFFSET
    mic = np.zeros((n, 4, 3))
    for j in range(4):
        mic[:, j] = pos + ECHOS_MIC_OFFSETS[j]
    return dict(
        t_s=t, body_pos_world_m=pos, body_quat_world_wxyz=body_q,
        body_vel_world_mps=vel, body_omega_body_rps=omega_body,
        body_omega_world_rps=omega_world, gimbal_pitch_rad=gimbal_pitch,
        gimbal_quat_body_wxyz=gimbal_body, gimbal_quat_world_wxyz=gimbal_world,
        beam_axis_world=beam, emitter_pos_world_m=emitter,
        mic_pos_world_m=mic,
        # metadata as dict for in-memory tests (save_trajectory_npz serializes it)
        metadata_json=MOCK_METADATA,
    )


# ===========================================================================
# Strict contract tests
# ===========================================================================

def test_strict_rejects_missing_field():
    """Every required field individually missing triggers ValueError in strict mode."""
    full = _make_fixture_dict()
    for k in _REQUIRED_KEYS_STRICT:
        d = {kk: v for kk, v in full.items() if kk != k}
        try:
            SampledTrajectory(d, strict_contract=True)
            assert False, f"should reject missing {k!r}"
        except ValueError:
            pass


def test_strict_rejects_malformed_metadata():
    """Wrong metadata values are rejected."""
    full = _make_fixture_dict()
    for key, bad_val in [("world_handedness", "left"), ("version", 2),
                          ("quaternion_order", "xyzw")]:
        meta = dict(MOCK_METADATA)
        meta[key] = bad_val
        d = dict(full)
        d["metadata_json"] = json.dumps(meta)
        try:
            SampledTrajectory(d, strict_contract=True)
            assert False, f"should reject metadata {key}={bad_val!r}"
        except (ValueError, KeyError):
            pass


def test_non_strict_allows_missing_fields():
    """strict_contract=False accepts partial fixtures (synthetic tests)."""
    minimal = {
        "t_s": np.array([0.0, 0.01]),
        "body_pos_world_m": np.zeros((2, 3)),
        "body_quat_world_wxyz": np.tile([1, 0, 0, 0], (2, 1)),
        "gimbal_quat_body_wxyz": np.tile([1, 0, 0, 0], (2, 1)),
    }
    # Should not raise
    t = SampledTrajectory(minimal, strict_contract=False)
    assert t.emitter_pos(0.005) is not None


# ===========================================================================
# Transform cross-validation tests
# ===========================================================================

def test_q_vs_minus_q_identical_validation():
    """q and -q produce the same angular errors in composition check."""
    n = 50
    rng = np.random.default_rng(0)
    d = _make_fixture_dict(n)
    traj = SampledTrajectory(d, strict_contract=True)
    r1 = traj.validate_transforms()

    # flip every other body quaternion
    d2 = {k: v.copy() for k, v in d.items()}
    for i in range(0, n, 2):
        d2["body_quat_world_wxyz"][i] *= -1
    traj2 = SampledTrajectory(d2, strict_contract=True)
    r2 = traj2.validate_transforms()

    for k in r1:
        assert abs(r1[k] - r2[k]) < 1e-10, f"{k} differs after q flip"


def test_omega_body_to_world():
    """Transformed omega_body matches exported omega_world within tolerance."""
    d = _make_fixture_dict(50)
    # inject non-zero body rotation
    rng = np.random.default_rng(1)
    for i in range(50):
        omega_b = np.array([0.1, 0.2, 0.3])
        # body quat = identity, so omega_world == omega_body
        d["body_omega_body_rps"][i] = omega_b
        d["body_omega_world_rps"][i] = omega_b  # identity rotation
    traj = SampledTrajectory(d, strict_contract=True)
    r = traj.validate_transforms()
    assert r.get("omega_transform", 1e9) < 1e-8


def test_beam_axis_catches_sign_error():
    """A deliberate beam axis sign error (Z flipped after pitch) must be caught."""
    d = _make_fixture_dict(30)
    for i in range(len(d["t_s"])):
        # pitch around Y axis
        th = 0.5 * d["t_s"][i]
        q = np.array([np.cos(th / 2), 0, np.sin(th / 2), 0])
        d["body_quat_world_wxyz"][i] = q
        R = _quat_to_rotmat(q)
        d["beam_axis_world"][i] = R @ np.array([1.0, 0.0, 0.0])
        d["emitter_pos_world_m"][i] = d["body_pos_world_m"][i] + R @ ECHOS_EMITTER_OFFSET
        for j in range(4):
            d["mic_pos_world_m"][i, j] = d["body_pos_world_m"][i] + R @ ECHOS_MIC_OFFSETS[j]
    traj_ok = SampledTrajectory(d.copy(), strict_contract=True)
    r_ok = traj_ok.validate_transforms()
    d_bad = {k: v.copy() for k, v in d.items()}
    d_bad["beam_axis_world"][:, 2] *= -1  # flip Z
    traj_bad = SampledTrajectory(d_bad, strict_contract=True)
    r_bad = traj_bad.validate_transforms()
    ba_bad = r_bad.get("beam_axis_deg", 0)
    ba_ok = r_ok.get("beam_axis_deg", 0)
    assert ba_bad > 10.0, f"sign error should produce >10 deg error, got {ba_bad:.2f}"
    assert ba_ok < 1.0, f"correct beam axis should have <1 deg error, got {ba_ok:.2f}"


def test_emitter_position_agrees():
    """Derived emitter position matches exported emitter_pos_world_m."""
    d = _make_fixture_dict(50)
    traj = SampledTrajectory(d, strict_contract=True)
    r = traj.validate_transforms()
    assert "emitter_pos_mm" in r
    assert r["emitter_pos_mm"] < 1.0, f"emitter pos error {r['emitter_pos_mm']:.3f} mm"


def test_microphone_positions_agree():
    """All four derived mic positions match exported mic_pos_world_m."""
    d = _make_fixture_dict(50)
    traj = SampledTrajectory(d, strict_contract=True)
    r = traj.validate_transforms()
    assert "mic_pos_mm" in r
    assert r["mic_pos_mm"] < 1.0, f"mic pos error {r['mic_pos_mm']:.3f} mm"


def test_emitter_offset_4mm_rejected():
    """A 4 mm error in the derived emitter position should exceed tolerance."""
    d = _make_fixture_dict(30)
    # offset the exported emitter positions by 4 mm
    d2 = {k: v.copy() for k, v in d.items()}
    d2["emitter_pos_world_m"] = d2["emitter_pos_world_m"] + np.array([0.0, 0.004, 0.0])
    traj = SampledTrajectory(d2, strict_contract=True)
    r = traj.validate_transforms()
    assert r.get("emitter_pos_mm", 0) > 3.0, f"4 mm offset should be detectable, got {r.get('emitter_pos_mm', 0):.3f} mm"


# ===========================================================================
# Body frame / axis convention tests
# ===========================================================================

def test_body_frame_convention():
    """+X forward, +Y left, +Z up: a body-rotation-only trajectory transforms
    a body-forward vector to world +X when identity."""
    n = 10
    d = _make_fixture_dict(n)
    traj = SampledTrajectory(d, strict_contract=True)
    R = traj.body_rot(0.0)
    # identity rotation → body +X maps to world +X
    forward = R @ np.array([1.0, 0.0, 0.0])
    assert np.allclose(forward, [1.0, 0.0, 0.0], atol=1e-10)


# ===========================================================================
# Acceptance CLI tests
# ===========================================================================

def _write_fixture_npz(path, data):
    save_trajectory_npz(
        path,
        t_s=data["t_s"],
        body_pos=data["body_pos_world_m"],
        body_quat=data["body_quat_world_wxyz"],
        body_vel=data["body_vel_world_mps"],
        body_omega_body=data["body_omega_body_rps"],
        body_omega_world=data["body_omega_world_rps"],
        gimbal_pitch=data["gimbal_pitch_rad"],
        gimbal_quat_body=data["gimbal_quat_body_wxyz"],
        gimbal_quat_world=data["gimbal_quat_world_wxyz"],
        beam_axis_world=data["beam_axis_world"],
        emitter_pos_world=data["emitter_pos_world_m"],
        mic_pos_world=data["mic_pos_world_m"],
        metadata_json=data["metadata_json"],
    )


def test_acceptance_cli_passes_valid_fixture():
    """CLI exits 0 on a fully valid NPZ file."""
    d = _make_fixture_dict(50)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        _write_fixture_npz(f.name, d)
        from argus.validate_echos_trajectory import validate
        from argus.motion import _TOL_OMEGA_TRANSFORM, _TOL_GIMBAL_QUAT_DEG, _TOL_BEAM_AXIS_DEG, _TOL_POSITION_MM
        report = validate(f.name)
        assert report["n_samples"] == 50
        # transform checks must be within their declared tolerances
        xv = report.get
        assert xv("xval_omega_transform", 0) <= _TOL_OMEGA_TRANSFORM
        assert xv("xval_gimbal_quat_composition_deg", 0) <= _TOL_GIMBAL_QUAT_DEG
        assert xv("xval_beam_axis_deg", 0) <= _TOL_BEAM_AXIS_DEG
        assert xv("xval_emitter_pos_mm", 0) <= _TOL_POSITION_MM
        assert xv("xval_mic_pos_mm", 0) <= _TOL_POSITION_MM
    os.unlink(f.name)


def test_acceptance_cli_fails_corrupted_frame():
    """CLI raises or returns FAIL on a frame-corrupted fixture."""
    d = _make_fixture_dict(30)
    d["metadata_json"] = {"schema": "echos-argus-trajectory",
                           "version": 1, "world_handedness": "left",
                           "world_up_axis": "+Z"}
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        _write_fixture_npz(f.name, d)
        from argus.validate_echos_trajectory import validate
        try:
            validate(f.name)
            assert False, "should raise on corrupted frame"
        except (ValueError, KeyError):
            pass
    os.unlink(f.name)


def test_contract_report_includes_transforms():
    """contract_report() returns transform validation keys."""
    d = _make_fixture_dict(30)
    traj = SampledTrajectory(d, strict_contract=True)
    r = traj.contract_report()
    assert "xval_gimbal_quat_composition_deg" in r
    assert "xval_beam_axis_deg" in r

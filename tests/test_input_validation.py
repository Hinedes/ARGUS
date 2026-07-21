import json
import os
import tempfile

import numpy as np
import pytest

from argus.acoustic_scene import Reflector, diamond_mics
from argus.calibrate import probe
from argus.detect import phase_tdoa, arrival_times
from argus.motion import (
    ECHOS_EMITTER_OFFSET, ECHOS_MIC_OFFSETS, ConstantVelocity,
    SampledTrajectory, _slerp, path_lengths,
)
from argus.schedule import Segment, Schedule, build_schedule
from argus.sensitivity import _field_points
from argus.solve import solve_point
from argus.synthesize import synthesize


def _metadata():
    return {
        "schema": "echos-argus-trajectory", "version": 1,
        "physics_dt_s": 0.005, "control_dt_s": 0.01,
        "world_handedness": "right", "world_up_axis": "+Z",
        "body_forward_axis": "+X", "body_positive_y_semantics": "left",
        "quaternion_order": "wxyz", "quaternion_semantics": "active_body_to_world",
        "position_unit": "m", "time_unit": "s", "angular_unit": "rad",
        "physical_emitter_offset_body_m": list(ECHOS_EMITTER_OFFSET),
    }


def _trajectory_data(n=2):
    q = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    t = np.linspace(0.0, 0.1, n)
    pos = np.zeros((n, 3))
    return {
        "t_s": t,
        "body_pos_world_m": pos,
        "body_quat_world_wxyz": q,
        "body_vel_world_mps": np.zeros((n, 3)),
        "body_omega_body_rps": np.zeros((n, 3)),
        "body_omega_world_rps": np.zeros((n, 3)),
        "gimbal_pitch_rad": np.zeros(n),
        "gimbal_quat_body_wxyz": q.copy(),
        "gimbal_quat_world_wxyz": q.copy(),
        "beam_axis_world": np.tile([1.0, 0.0, 0.0], (n, 1)),
        "emitter_pos_world_m": np.tile(ECHOS_EMITTER_OFFSET, (n, 1)),
        "mic_pos_world_m": np.tile(ECHOS_MIC_OFFSETS, (n, 1, 1)),
        "metadata_json": json.dumps(_metadata()),
    }


def test_solver_rejects_bad_inputs_and_accepts_non_z_beam():
    mics = diamond_mics()
    point = np.array([2.0, 0.0, 0.0])
    tof = (np.linalg.norm(point) + np.linalg.norm(mics - point, axis=1)) / 343.0
    estimate = solve_point(tof, mics, 343.0, beam_axis=[4.0, 0.0, 0.0])
    assert np.linalg.norm(estimate - point) < 1e-6
    with pytest.raises(ValueError):
        solve_point([np.nan] * 4, mics)
    with pytest.raises(ValueError):
        solve_point(tof, mics, beam_axis=[0.0, 0.0, 0.0])


def test_schedule_validation_and_actual_end_time():
    with pytest.raises(ValueError):
        build_schedule([], 0.001, 100_000)
    schedule = Schedule([Segment(40_000, 0.5, 0.01)], 100_000)
    assert schedule.duration == pytest.approx(0.51)
    with pytest.raises(ValueError):
        Schedule([Segment(40_000, 0.0, 0.01), Segment(42_000, 0.005, 0.01)], 100_000)


def test_detector_rejects_empty_windows_and_bad_reference():
    schedule = build_schedule([40_000, 42_000], 0.001, 100_000)
    waveforms = np.zeros((4, 1))
    with pytest.raises(ValueError):
        arrival_times(waveforms, schedule)
    with pytest.raises(ValueError):
        phase_tdoa(np.zeros((4, 1000)), schedule.segments, 100_000,
                   np.ones(4) * 0.001, reference_mic=4)


def test_custom_speed_probe_does_not_mutate_calibration_inputs():
    result = probe([40_000, 42_000], 0.001, [Reflector([0.0, 0.0, 2.0])],
                   speed=300.0, sample_rate=100_000)
    assert np.all(np.isfinite(result["tof"]))


def test_moving_synthesis_buffer_covers_reception_time():
    mics = diamond_mics()
    schedule = build_schedule([40_000, 42_000], 0.002, 100_000, guard=0.02)
    traj = ConstantVelocity(mics, np.zeros(3), np.array([500.0, 0.0, 0.0]))
    reflector = Reflector([0.0, 0.0, 2.0])
    waveforms = synthesize(schedule, [reflector], mics, trajectory=traj)
    delay = np.max(path_lengths(reflector.pos, schedule.segments[-1].start,
                                traj)) / 343.0
    required = (schedule.segments[-1].start + delay +
                schedule.segments[-1].duration) * schedule.sample_rate
    assert waveforms.shape[1] > required


def test_slerp_returns_unit_quaternion():
    q = _slerp(np.array([2.0, 0.0, 0.0, 0.0]),
               np.array([0.0, 0.0, 0.0, 3.0]), 0.5)
    assert np.linalg.norm(q) == pytest.approx(1.0)


def test_non_strict_optional_transform_fields_are_optional():
    data = _trajectory_data()
    for key in ("gimbal_quat_world_wxyz", "beam_axis_world", "emitter_pos_world_m",
                "mic_pos_world_m", "metadata_json"):
        data.pop(key)
    traj = SampledTrajectory(data, strict_contract=False)
    assert traj.validate_transforms()["omega_transform"] == pytest.approx(0.0)
    assert np.all(np.isfinite(traj.contract_report()["pos_bounds_m"]))


def test_strict_trajectory_rejects_wrong_mic_shape_and_transform_tolerance():
    data = _trajectory_data()
    bad_shape = dict(data)
    bad_shape["mic_pos_world_m"] = np.zeros((2, 3))
    with pytest.raises(ValueError, match="mic_pos_world_m"):
        SampledTrajectory(bad_shape)

    bad_transform = dict(data)
    bad_transform["emitter_pos_world_m"] = data["emitter_pos_world_m"].copy()
    bad_transform["emitter_pos_world_m"][:, 1] += 0.004
    traj = SampledTrajectory(bad_transform)
    assert traj.validate_transforms()["emitter_pos_mm"] > 3.0


def test_validate_cli_enforces_transform_tolerance():
    from argus.validate_echos_trajectory import validate

    data = _trajectory_data()
    data["emitter_pos_world_m"] = data["emitter_pos_world_m"].copy()
    data["emitter_pos_world_m"][:, 1] += 0.004
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = f.name
    try:
        np.savez_compressed(path, **data)
        with pytest.raises(ValueError, match="transform tolerance"):
            validate(path)
    finally:
        os.unlink(path)


def test_fov_points_are_at_requested_line_of_sight_range():
    point = _field_points([2.0], [np.deg2rad(20)], [np.deg2rad(-10)])[0]
    assert np.linalg.norm(point) == pytest.approx(2.0)

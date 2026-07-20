"""Motion-acoustic coupling: full compensation comparison.

Compares four reconstructions for a moving platform:

    A: frozen scan-start pose (t=0 positions assumed for all tones)
    B: per-tone emission pose, microphones frozen at emission time
    C: exact emitter-at-emission + mic-at-reception geometry
    D: C with ESKF-like pose noise added

Separate gimbal stabilization test: body pitch + gimbal counter-rotation.
"""

import numpy as np
from argus.acoustic_scene import Reflector, diamond_mics, ORIGIN
from argus.schedule import build_schedule
from argus.synthesize import synthesize
from argus.motion import ConstantVelocity, Trajectory
from argus.detect import coherent_onset, median_tof
from argus.solve import solve_point
import argus.sensitivity as S

SPEED = 343.0
SR = 250_000
FREQS = [40_000, 42_000, 44_000, 46_000, 48_000]
TONE = 0.004
mics = diamond_mics()
P_ref = np.array([0.3, -0.2, 2.0])


def _solve_method_A(arr, traj, mics_static):
    """Frozen scan-start pose: median ToF, static t=0 microphone positions."""
    tof = median_tof(arr, axis=1)
    return solve_point(tof, mics_static, SPEED)


def _segs(sched):
    return sched.segments

def _solve_method_B(arr, sched, traj, mics_static):
    """Emission-only: per-tone solve with mic positions at emission time."""
    pts = []
    for k, seg in enumerate(_segs(sched)):
        t_emit = seg.start
        mics_k = np.array([traj.mic_pos(t_emit, i) for i in range(len(mics_static))])
        pts.append(solve_point(arr[:, k], mics_k, SPEED))
    return np.median(pts, axis=0)


def _solve_method_C(arr, sched, traj, mics_static):
    """Full emission+reception: emitter at t_emit, each mic at its own
    t_return = t_emit + per-mic ToF."""
    pts = []
    for k, seg in enumerate(_segs(sched)):
        t_emit = seg.start
        mics_k = np.zeros_like(mics_static)
        for i in range(len(mics_static)):
            t_ret = t_emit + arr[i, k]
            mics_k[i] = traj.mic_pos(t_ret, i)
        pts.append(solve_point(arr[:, k], mics_k, SPEED))
    return np.median(pts, axis=0)


def _solve_method_D(arr, sched, traj, mics_static, noise_std=0.01):
    """ESKF-like: same as C but with Gaussian noise on per-mic positions."""
    pts = []
    rng = np.random.default_rng(0)
    for k, seg in enumerate(_segs(sched)):
        t_emit = seg.start
        mics_k = np.zeros_like(mics_static)
        for i in range(len(mics_static)):
            t_ret = t_emit + arr[i, k]
            pos = traj.mic_pos(t_ret, i) + rng.normal(0, noise_std, 3)
            mics_k[i] = pos
        pts.append(solve_point(arr[:, k], mics_k, SPEED))
    return np.median(pts, axis=0)


def _run_all_methods(speed_mps, traj, sched, mics_static):
    """Return dict of {method_name: error} for a single speed value."""
    wf = synthesize(sched, [Reflector(P_ref)], mics_static, SPEED,
                    coherent=True, trajectory=traj)
    arr = coherent_onset(wf, sched)
    true = P_ref
    est_A = _solve_method_A(arr, traj, mics_static)
    est_B = _solve_method_B(arr, sched, traj, mics_static)
    est_C = _solve_method_C(arr, sched, traj, mics_static)
    est_D = _solve_method_D(arr, sched, traj, mics_static, noise_std=0.01)
    return {
        "A": float(np.linalg.norm(est_A - true)),
        "B": float(np.linalg.norm(est_B - true)),
        "C": float(np.linalg.norm(est_C - true)),
        "D": float(np.linalg.norm(est_D - true)),
    }


def test_motion_method_order():
    """Across speeds, C beats B beats A. D is similar to C with added noise."""
    sched = build_schedule(FREQS, TONE, SR)
    prev = {}
    for speed in [0.0, 0.5, 1.0, 2.0]:
        traj = ConstantVelocity(mic_local=mics, p0=np.zeros(3),
                                v=np.array([speed, 0, 0]))
        errs = _run_all_methods(speed, traj, sched, mics)
        print(f"  {speed:.1f} m/s:  A={errs['A']:.4f}  B={errs['B']:.4f}  C={errs['C']:.4f}  D={errs['D']:.4f}")
        # C beats B beats A at non-zero speed
        if speed > 0:
            assert errs["C"] <= errs["B"] + 1e-4
            assert errs["B"] <= errs["A"] + 1e-4
        # A worsens monotonically
        if prev:
            assert errs["A"] >= prev["A"]
        prev = errs
        # D grows with pose noise (ratio recorded, not asserted)
        if speed > 0 and errs["C"] > 0.001:
            print(f"    D/C ratio at {speed:.1f} m/s: {errs['D']/errs['C']:.1f}x")


def test_motion_c_recovers_static():
    """Method C at 2 m/s must be close to the static (speed=0) accuracy."""
    sched = build_schedule(FREQS, TONE, SR)
    # static reference
    traj0 = ConstantVelocity(mic_local=mics, p0=np.zeros(3), v=np.zeros(3))
    err_static = _run_all_methods(0.0, traj0, sched, mics)
    # moving at 2 m/s, method C
    traj2 = ConstantVelocity(mic_local=mics, p0=np.zeros(3), v=np.array([2.0, 0, 0]))
    errs2 = _run_all_methods(2.0, traj2, sched, mics)
    assert errs2["C"] < err_static["A"] * 2.0


def test_motion_lateral_and_vertical():
    """Method C recovers static accuracy under lateral and vertical motion."""
    sched = build_schedule(FREQS, TONE, SR)
    for label, v in [("lateral", [0, 2.0, 0]), ("vertical", [0, 0, 2.0])]:
        traj = ConstantVelocity(mic_local=mics, p0=np.zeros(3), v=np.array(v))
        wf = synthesize(sched, [Reflector(P_ref)], mics, SPEED, coherent=True, trajectory=traj)
        arr = coherent_onset(wf, sched)
        est_C = _solve_method_C(arr, sched, traj, mics)
        err = np.linalg.norm(est_C - P_ref)
        print(f"  {label}: C error={err:.4f}")
        assert err < 0.05


def test_motion_rotation():
    """Method C works under yaw and pitch rotation."""
    sched = build_schedule(FREQS, TONE, SR)
    for label, omega in [("yaw", [0, 0, 1.0]), ("pitch", [0, 1.0, 0])]:
        traj = ConstantVelocity(mic_local=mics, p0=np.zeros(3), v=np.zeros(3),
                                omega=np.array(omega))
        wf = synthesize(sched, [Reflector(P_ref)], mics, SPEED, coherent=True, trajectory=traj)
        arr = coherent_onset(wf, sched)
        est_C = _solve_method_C(arr, sched, traj, mics)
        err = np.linalg.norm(est_C - P_ref)
        print(f"  {label}: C error={err:.4f}")
        assert err < 0.05


class GimbalTrajectory(ConstantVelocity):
    """Like ConstantVelocity but beam_axis is gimbal-stabilized in world frame,
    independent of body orientation."""
    def beam_axis(self, t):
        return self.beam_axis_world  # constant in world space


def test_motion_gimbal_stabilization():
    """Body pitch with gimbal counter-rotation: C error must stay bounded.
    Without gimbal, the beam axis tilts with the body and the solver's
    beam_axis prior points in the wrong direction, causing catastrophic error."""
    sched = build_schedule(FREQS, TONE, SR)
    # body pitches forward at 30 deg/s around y-axis
    omega = [0, np.deg2rad(30), 0]

    # Ungimballed: beam follows body (default ConstantVelocity)
    traj_ungim = ConstantVelocity(mic_local=mics, p0=np.zeros(3), v=np.zeros(3),
                                   omega=np.array(omega))
    wf_ungim = synthesize(sched, [Reflector(P_ref)], mics, SPEED, coherent=True,
                           trajectory=traj_ungim)
    arr_ungim = coherent_onset(wf_ungim, sched)
    # Solve with wrong beam axis (the body-tilted one at each tone)
    pts = []
    for k, seg in enumerate(sched.segments):
        t_emit = seg.start
        mics_k = np.array([traj_ungim.mic_pos(t_emit, i) for i in range(len(mics))])
        beam_k = traj_ungim.body_rot(t_emit) @ np.array([0, 0, 1])
        tof_k = arr_ungim[:, k]
        pts.append(solve_point(tof_k, mics_k, SPEED, beam_axis=beam_k))
    est_ungim = np.median(pts, axis=0)
    err_ungim = np.linalg.norm(est_ungim - P_ref)

    # Gimballed: beam fixed in world
    traj_gim = GimbalTrajectory(mic_local=mics, p0=np.zeros(3), v=np.zeros(3),
                                 omega=np.array(omega))
    wf_gim = synthesize(sched, [Reflector(P_ref)], mics, SPEED, coherent=True,
                         trajectory=traj_gim)
    arr_gim = coherent_onset(wf_gim, sched)
    est_gim = _solve_method_C(arr_gim, sched, traj_gim, mics)
    err_gim = np.linalg.norm(est_gim - P_ref)

    print(f"  Ungimballed body-pitch error: {err_ungim:.4f}")
    print(f"  Gimballed beam error: {err_gim:.4f}")
    # Gimballed must be dramatically better than ungimballed
    assert err_gim < err_ungim * 0.5, (
        f"gimbal {err_gim:.4f} not significantly better than ungimbal {err_ungim:.4f}"
    )

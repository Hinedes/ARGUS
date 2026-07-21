"""Recover a 3D reflector point from ToF / TDOA.

Each mic gives a round-trip distance via ToF: d_i = c * t_i. The reflector
sits on the ellipsoid defined by |P - O| + |P - M_i| = d_i. Solving all four
simultaneously recovers P = (x, y, z), including signed z because the tilted
diamond breaks the up/down symmetry a flat square would have.
"""

import numpy as np

from .acoustic_scene import ORIGIN, diamond_mics


def tof_to_distance(tof: np.ndarray, speed: float = 343.0) -> np.ndarray:
    tof = np.asarray(tof, dtype=float)
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError("speed must be finite and positive")
    if np.any(~np.isfinite(tof)) or np.any(tof < 0):
        raise ValueError("tof must contain finite, non-negative values")
    return speed * tof


def _residual(P, mics, origin, d):
    rO = np.linalg.norm(P - origin)
    res = np.zeros(mics.shape[0])
    J = np.zeros((mics.shape[0], 3))
    gO = (P - origin) / (rO + 1e-12)
    for i, M in enumerate(mics):
        rM = np.linalg.norm(P - M)
        res[i] = rO + rM - d[i]
        gM = (P - M) / (rM + 1e-12)
        J[i] = gO + gM
    return res, J


def solve_point(
    tof: np.ndarray,
    mics: np.ndarray | None = None,
    speed: float = 343.0,
    origin: np.ndarray | None = None,
    beam_axis: np.ndarray | None = None,
    cone_halfangle: float = np.pi / 2.0,
) -> np.ndarray:
    """Least-squares estimate of reflector position from per-mic ToF.

    Minimizes sum_i (|P-O| + |P-M_i| - c*t_i)^2 via Levenberg-Marquardt,
    which damps the step when the geometry is ill-conditioned.

    ARGUS knows the gimbal beam axis, so we use it as a prior: the initial
    guess lies on that axis at the median range, and any converged solution
    behind the emitter (outside the forward hemisphere / beam cone) is rejected
    as a mirror ambiguity. This is the Throw-mode constraint the raw unconstrained
    solve would otherwise ignore.
    """
    tof = np.asarray(tof, dtype=float)
    if tof.ndim != 1:
        raise ValueError(f"tof must be 1-D, got shape {tof.shape}")
    if mics is None:
        mics = diamond_mics()
    mics = np.asarray(mics, dtype=float)
    if mics.ndim != 2 or mics.shape[1] != 3 or len(mics) != len(tof):
        raise ValueError("mics must have shape (len(tof), 3)")
    if len(mics) < 3 or np.any(~np.isfinite(mics)):
        raise ValueError("mics must contain at least three finite positions")
    if origin is None:
        origin = ORIGIN
    origin = np.asarray(origin, dtype=float)
    if origin.shape != (3,) or np.any(~np.isfinite(origin)):
        raise ValueError("origin must be a finite 3-vector")
    if beam_axis is None:
        beam_axis = np.array([0.0, 0.0, 1.0])
    beam_axis = np.asarray(beam_axis, dtype=float)
    beam_norm = np.linalg.norm(beam_axis)
    if beam_axis.shape != (3,) or not np.isfinite(beam_norm) or beam_norm == 0:
        raise ValueError("beam_axis must be a finite, non-zero 3-vector")
    beam_axis = beam_axis / beam_norm
    if not np.isfinite(cone_halfangle) or not 0 <= cone_halfangle <= np.pi:
        raise ValueError("cone_halfangle must be finite and in [0, pi]")

    d = tof_to_distance(tof, speed)
    R = np.median(d) / 2.0  # coarse range from median absolute ToF

    # start points: on the beam axis, plus small lateral offsets
    starts = [origin + beam_axis * R]
    lateral_ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(lateral_ref, beam_axis)) > 0.9:
        lateral_ref = np.array([0.0, 1.0, 0.0])
    lateral_x = lateral_ref - beam_axis * np.dot(lateral_ref, beam_axis)
    lateral_x /= np.linalg.norm(lateral_x)
    lateral_y = np.cross(beam_axis, lateral_x)
    for s in [(0.1, 0.1), (-0.1, 0.1), (0.1, -0.1), (-0.1, -0.1)]:
        lateral = s[0] * lateral_x + s[1] * lateral_y
        starts.append(origin + beam_axis * R + lateral)

    best = None
    best_cost = np.inf
    for P0 in starts:
        P = P0.copy()
        lam = 1e-3
        for _ in range(100):
            res, J = _residual(P, mics, origin, d)
            cost = float(res @ res)
            A = J.T @ J + lam * np.eye(3)
            try:
                step = np.linalg.solve(A, J.T @ -res)
            except np.linalg.LinAlgError:
                lam = min(lam * 10.0, 1e12)
                continue
            if np.any(~np.isfinite(step)):
                lam = min(lam * 10.0, 1e12)
                continue
            Pn = P + step
            resn, _ = _residual(Pn, mics, origin, d)
            if resn @ resn < cost:
                P = Pn
                lam = max(lam * 0.7, 1e-9)
                if np.linalg.norm(step) < 1e-10:
                    break
            else:
                lam = min(lam * 2.0, 1e6)
        res, _ = _residual(P, mics, origin, d)
        cost = float(res @ res)
        if not np.all(np.isfinite(P)) or not np.isfinite(cost):
            continue
        # reject mirror solutions outside the known forward beam cone
        if np.dot(P - origin, beam_axis) <= 0:
            continue
        ang = np.arccos(
            np.clip(np.dot((P - origin) / (np.linalg.norm(P - origin) + 1e-12),
                            beam_axis), -1.0, 1.0))
        if ang > cone_halfangle:
            continue
        if cost < best_cost:
            best_cost = cost
            best = P
    if best is None:
        raise RuntimeError("no valid reflector solution in the beam cone")
    return best

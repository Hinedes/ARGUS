"""Recover a 3D reflector point from ToF / TDOA.

Each mic gives a round-trip distance via ToF: d_i = c * t_i. The reflector
sits on the ellipsoid defined by |P - O| + |P - M_i| = d_i. Solving all four
simultaneously recovers P = (x, y, z), including signed z because the tilted
diamond breaks the up/down symmetry a flat square would have.
"""

import numpy as np

from .acoustic_scene import ORIGIN, diamond_mics


def tof_to_distance(tof: np.ndarray, speed: float = 343.0) -> np.ndarray:
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
    if mics is None:
        mics = diamond_mics()
    if origin is None:
        origin = ORIGIN
    if beam_axis is None:
        beam_axis = np.array([0.0, 0.0, 1.0])
    beam_axis = beam_axis / np.linalg.norm(beam_axis)

    d = tof_to_distance(tof, speed)
    R = np.median(d) / 2.0  # coarse range from median absolute ToF

    # start points: on the beam axis, plus small lateral offsets
    starts = [origin + beam_axis * R]
    for s in [(0.1, 0.1), (-0.1, 0.1), (0.1, -0.1), (-0.1, -0.1)]:
        lateral = np.array([s[0], s[1], 0.0])
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
            step = np.linalg.solve(A, J.T @ -res)
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
        # fall back to the lowest-cost start if all were outside the cone
        best = starts[0]
    return best

"""Numerical correctness: exact (unquantized) ToF must invert to the exact
point everywhere in the field of view. This separates solver correctness from
measurement sensitivity -- a 70 cm error under perfect data is a bug, not
hardware. Also inspect the Jacobian so the "ill-conditioned" claim is checked,
not asserted.
"""

import numpy as np
import pytest

from argus.acoustic_scene import diamond_mics, ORIGIN
from argus.solve import solve_point, _residual


MICS = diamond_mics()
SPEED = 343.0

# points spanning the intended field of view (ahead, off-axis, left/right, up)
POINTS = [
    np.array([0.0, 0.0, 2.0]),
    np.array([0.3, -0.2, 2.0]),
    np.array([-1.5, 0.0, 2.0]),
    np.array([0.0, 1.5, 2.0]),
    np.array([1.5, 1.0, 1.5]),
]


@pytest.mark.parametrize("pos", POINTS, ids=lambda p: str(p.tolist()))
def test_exact_tof_inverts_exactly(pos):
    """Same bistatic model for synthesis and solving -> exact recovery."""
    tof = (np.linalg.norm(pos) + np.linalg.norm(MICS - pos, axis=1)) / SPEED
    est = solve_point(tof, MICS, SPEED)
    assert np.linalg.norm(est - pos) < 1e-6


def test_residual_zero_at_true_point():
    pos = np.array([0.3, -0.2, 2.0])
    tof = (np.linalg.norm(pos) + np.linalg.norm(MICS - pos, axis=1)) / SPEED
    res, _ = _residual(pos, MICS, ORIGIN, SPEED * tof)
    assert np.allclose(res, 0.0, atol=1e-9)


@pytest.mark.parametrize("pos", POINTS, ids=lambda p: str(p.tolist()))
def test_jacobian_well_conditioned(pos):
    """Ill-conditioned != singular: rank 3, finite condition number."""
    tof = (np.linalg.norm(pos) + np.linalg.norm(MICS - pos, axis=1)) / SPEED
    _, J = _residual(pos, MICS, ORIGIN, SPEED * tof)
    sv = np.linalg.svd(J, compute_uv=False)
    assert np.linalg.matrix_rank(J, tol=1e-6) == 3
    assert sv.max() / sv.min() < 1e3


def test_mirror_rejected_by_beam_cone():
    """A point behind the emitter (mirror ambiguity) is rejected when the beam
    axis is known; the forward solution is returned instead."""
    pos = np.array([0.0, 0.0, 2.0])
    tof = (np.linalg.norm(pos) + np.linalg.norm(MICS - pos, axis=1)) / SPEED
    est = solve_point(tof, MICS, SPEED, beam_axis=np.array([0.0, 0.0, 1.0]))
    assert est[2] > 0.0
    assert np.linalg.norm(est - pos) < 1e-6

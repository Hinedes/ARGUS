import numpy as np

from argus.acoustic_scene import Reflector, diamond_mics
from argus.detect import arrival_times, median_tof
from argus.schedule import build_schedule
from argus.solve import solve_point
from argus.synthesize import synthesize

SPEED = 343.0
SAMPLE_RATE = 250_000
FREQS = [40_000, 42_000, 44_000, 46_000, 48_000]
TONE = 0.004


def test_out_of_order_echoes():
    """An off-axis reflector makes the four mics receive echoes in mixed
    (non-monotonic) order. The pipeline must still resolve a bounded point in
    the forward hemisphere -- the out-of-order arrival pattern is a geometry
    fact the solver handles, not a failure mode.
    """
    pos = np.array([1.5, 1.0, 1.5])
    sched = build_schedule(FREQS, TONE, SAMPLE_RATE)
    mics = diamond_mics()
    wf = synthesize(sched, [Reflector(pos)], mics, SPEED)
    arr = arrival_times(wf, sched)
    tof = median_tof(arr, axis=1)

    # confirm arrivals really are out of natural mic order
    assert not np.all(np.diff(tof) >= 0)

    est = solve_point(tof, mics, SPEED, beam_axis=np.array([0.0, 0.0, 1.0]))
    # bounded by the 12 cm array's angular ceiling, not diverging to nonsense
    assert np.linalg.norm(est - pos) < 1.0
    assert est[2] > 0.0  # forward hemisphere (mirror rejected)

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
POS = np.array([0.3, -0.2, 2.0])


def run_with_order(order):
    sched = build_schedule(FREQS, TONE, SAMPLE_RATE)
    mics = diamond_mics()
    # emit the five tones in a different order; arrival_times keys each ToF by
    # its frequency, so the recovered point must be order-independent
    sched.segments = [sched.segments[i] for i in order]
    wf = synthesize(sched, [Reflector(POS)], mics, SPEED)
    arr = arrival_times(wf, sched)
    tof = median_tof(arr, axis=1)
    return solve_point(tof, mics, SPEED)


def test_method_c_order_invariant():
    """Method C: reordered tone emission must recover the SAME point. The
    property under test is order-independence of the recovered location, not
    a 1 cm accuracy claim (that is set by the 12 cm array's angular ceiling)."""
    est_natural = run_with_order([0, 1, 2, 3, 4])
    est_shuffled = run_with_order([4, 2, 0, 3, 1])
    # both runs agree with each other (order doesn't matter)
    assert np.linalg.norm(est_natural - est_shuffled) < 1e-3
    # both stay within the physical ceiling, not diverging to nonsense
    assert np.linalg.norm(est_natural - POS) < 0.1
    assert np.linalg.norm(est_shuffled - POS) < 0.1


def test_method_c_five_estimates():
    """All five segments yield usable, consistent per-mic ToF."""
    sched = build_schedule(FREQS, TONE, SAMPLE_RATE)
    mics = diamond_mics()
    wf = synthesize(sched, [Reflector(POS)], mics, SPEED)
    arr = arrival_times(wf, sched)
    assert arr.shape == (4, 5)
    # each segment's arrival should be close to the median
    med = median_tof(arr, axis=1)
    for j in range(5):
        assert np.allclose(arr[:, j], med, atol=1e-4)

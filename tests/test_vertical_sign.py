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


def reconstruct(pos):
    sched = build_schedule(FREQS, TONE, SAMPLE_RATE)
    mics = diamond_mics()
    wf = synthesize(sched, [Reflector(np.asarray(pos, float))], mics, SPEED)
    arr = arrival_times(wf, sched)
    tof = median_tof(arr, axis=1)
    # ARGUS knows the gimbal beam points along +z; pass it so mirror
    # (behind-emitter) solutions are rejected
    return solve_point(tof, mics, SPEED, beam_axis=np.array([0.0, 0.0, 1.0]))


def test_vertical_sign():
    """Tilted diamond must recover SIGNED vertical direction in the forward
    field of view (above vs below the boresight plane)."""
    above = reconstruct([0.0, 0.5, 2.0])
    assert above[1] > 0.0
    below = reconstruct([0.0, -0.5, 2.0])
    assert below[1] < 0.0


def test_horizontal_sign():
    """Left/right must recover signed x; the array has lateral leverage
    off-axis. Assert the SIGN is right and the error stays within the 12 cm
    array's angular ceiling (not a 1 cm fantasy)."""
    left = reconstruct([-1.5, 0.0, 2.0])
    assert left[0] < 0.0
    right = reconstruct([1.5, 0.0, 2.0])
    assert right[0] > 0.0
    up = reconstruct([0.0, 1.5, 2.0])
    assert up[1] > 0.0
    down = reconstruct([0.0, -1.5, 2.0])
    assert down[1] < 0.0
    # bounded by the physical ceiling: no divergence to km-scale garbage
    for p, e in [([-1.5, 0.0, 2.0], left), ([1.5, 0.0, 2.0], right),
                  ([0.0, 1.5, 2.0], up), ([0.0, -1.5, 2.0], down)]:
        assert np.linalg.norm(e - np.array(p)) < 1.0

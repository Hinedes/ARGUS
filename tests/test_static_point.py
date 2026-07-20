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


def make_waveforms(refl_pos):
    sched = build_schedule(FREQS, TONE, SAMPLE_RATE)
    mics = diamond_mics()
    wf = synthesize(sched, [Reflector(np.asarray(refl_pos, float))], mics, SPEED)
    return wf, sched, mics


def test_static_point():
    """Gate: one known reflector reconstructed within 1 cm."""
    pos = np.array([0.0, 0.0, 2.0])
    wf, sched, mics = make_waveforms(pos)
    arr = arrival_times(wf, sched)
    tof = median_tof(arr, axis=1)
    est = solve_point(tof, mics, SPEED)
    assert np.linalg.norm(est - pos) < 0.01

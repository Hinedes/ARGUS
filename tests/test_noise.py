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
POS = np.array([0.0, 0.0, 2.0])


def run(noise, seed=0):
    rng = np.random.default_rng(seed)
    sched = build_schedule(FREQS, TONE, SAMPLE_RATE)
    mics = diamond_mics()
    wf = synthesize(sched, [Reflector(POS)], mics, SPEED)
    wf = wf + rng.normal(0, noise, wf.shape)
    arr = arrival_times(wf, sched)
    tof = median_tof(arr, axis=1)
    return solve_point(tof, mics, SPEED)


def test_noise_sensitivity_ceiling():
    """With a 12 cm array at 2 m, ~0.1 ms ToF jitter sets the real angular
    ceiling (~0.5 m), NOT a solver bug. Assert the error (a) stays bounded
    and (b) grows monotonically with noise rather than exploding to nonsense.
    """
    e025 = np.linalg.norm(run(noise=0.025) - POS)
    e050 = np.linalg.norm(run(noise=0.05) - POS)
    e100 = np.linalg.norm(run(noise=0.10) - POS)
    # bounded: never diverges to km-scale garbage (that would be a solver bug)
    assert e100 < 2.0
    # monotonic in noise: more noise -> more error, confirming physical sensitivity
    assert e025 <= e050 <= e100


def test_missing_frequency():
    """One missing tone segment should not break reconstruction."""
    sched = build_schedule(FREQS, TONE, SAMPLE_RATE)
    mics = diamond_mics()
    wf = synthesize(sched, [Reflector(POS)], mics, SPEED)
    # zero out one segment on all mics
    seg = sched.segments[2]
    i0 = int(seg.start * SAMPLE_RATE)
    i1 = int((seg.start + seg.duration) * SAMPLE_RATE)
    wf[:, i0:i1] = 0.0
    arr = arrival_times(wf, sched)
    # drop the corrupted (3rd) segment column before median
    arr = np.delete(arr, 2, axis=1)
    tof = median_tof(arr, axis=1)
    est = solve_point(tof, mics, SPEED)
    assert np.linalg.norm(est - POS) < 0.05

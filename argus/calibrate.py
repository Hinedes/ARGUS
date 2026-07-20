"""Adaptive accommodation: choose frequency / range preferences.

While stationary, ARGUS probes candidate schedules and scores them on the
metrics the rest of the chain can observe: received SNR, phase stability,
leakage, multipath decay, and agreement between repeated ToF measurements and
between microphones. The 40-50 kHz schedule is one candidate, not a law.
"""

import numpy as np

from .acoustic_scene import ORIGIN, diamond_mics
from .detect import arrival_times, median_tof
from .schedule import build_schedule
from .synthesize import synthesize


def probe(
    freqs,
    tone_duration,
    reflectors,
    sample_rate=250_000,
    speed=343.0,
    noise=0.0,
    rng=None,
):
    """Run one candidate schedule end to end and return observable metrics."""
    rng = rng or np.random.default_rng(0)
    sched = build_schedule(freqs, tone_duration, sample_rate)
    mics = diamond_mics()
    wf = synthesize(sched, reflectors, mics, speed)
    if noise:
        wf = wf + rng.normal(0, noise, wf.shape)

    arr = arrival_times(wf, sched)
    tof = median_tof(arr, axis=1)
    dist = speed * tof

    # SNR proxy: peak correlation amplitude vs residual energy
    peak = np.max(np.abs(wf), axis=1)
    snr = 20 * np.log10(peak / (np.std(wf, axis=1) + 1e-9))

    # agreement between mics: spread of solved-consistent distances
    mic_agreement = float(np.std(dist))

    # phase stability proxy: spread of per-frequency arrival across mics
    phase_stability = float(np.std(arr, axis=0).mean())

    return {
        "snr_db": float(np.mean(snr)),
        "mic_agreement": mic_agreement,
        "phase_stability": phase_stability,
        "tof": tof,
        "dist": dist,
    }


def search(
    candidates,
    reflectors,
    sample_rate=250_000,
    speed=343.0,
    noise=0.0,
):
    """Score each candidate (list of (freqs, tone_duration)) and rank by SNR."""
    scored = []
    for freqs, dur in candidates:
        m = probe(freqs, dur, reflectors, sample_rate, speed, noise)
        # prefer high SNR, low mic disagreement; weight SNR primarily
        score = m["snr_db"] - 10.0 * np.log10(m["mic_agreement"] + 1e-6)
        scored.append((score, freqs, dur, m))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored

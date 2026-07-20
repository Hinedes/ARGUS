"""Integration tests for detect.phase_tdoa on REAL synthesize.py waveforms.

Pipeline under test:
    synthesize (coherent carrier) -> envelope detector (coarse ToF)
    -> phase_tdoa (sub-sample TDOA via cross-frequency differential phase)

These do NOT modify arrival_times / median_tof / the solver, and they use the
per-mic coarse onset from the envelope detector as the common alignment anchor
(no five independently detected phase centres).
"""

import numpy as np
import pytest

from argus.acoustic_scene import Reflector, diamond_mics, ORIGIN
from argus.detect import arrival_times, median_tof, phase_tdoa, coherent_onset
from argus.schedule import build_schedule
from argus.synthesize import synthesize

SPEED = 343.0
SR = 250_000
FREQS = [40_000, 42_000, 44_000, 46_000, 48_000]
TONE = 0.004
POS = np.array([0.3, -0.2, 2.0])


def _scene(order=None, coherent=True, noise=0.0, seed=0, pos=POS):
    sched = build_schedule(FREQS, TONE, SR)
    if order is not None:
        sched.segments = [sched.segments[i] for i in order]
    mics = diamond_mics()
    wf = synthesize(sched, [Reflector(pos)], mics, SPEED, coherent=coherent)
    if noise > 0.0:
        rng = np.random.default_rng(seed)
        wf = wf + rng.normal(0.0, noise, wf.shape)
    # phase-coherent carrier needs a phase-coherent onset anchor; for the
    # default (non-coherent) carrier the existing envelope detector is used.
    if coherent:
        arr = coherent_onset(wf, sched)
    else:
        arr = arrival_times(wf, sched)
    coarse = median_tof(arr, axis=1)
    return wf, sched, mics, coarse


def _true_tdoa(pos, mics, ref=0):
    tau = np.array([
        (np.linalg.norm(pos - ORIGIN) + np.linalg.norm(m - pos)) / SPEED
        for m in mics
    ])
    return tau - tau[ref]


def _apply_common_phase(wf, sched, refl_phase):
    """Multiply every tone segment by cos(refl_phase[k]) -- a per-frequency
    phase shift IDENTICAL across all microphones (common reflection response)."""
    out = wf.copy()
    for k, seg in enumerate(sched.segments):
        s = int(seg.start * SR)
        e = s + int(seg.duration * SR)
        out[:, s:e] = out[:, s:e] * np.cos(refl_phase[k])
    return out


def test_exact_sampled_waveform():
    """Fractional delays not on ADC samples; TDOA recovered well below 4 us."""
    wf, sched, mics, coarse = _scene()
    delays, valid = phase_tdoa(wf, sched.segments, SR, coarse, reference_mic=0)
    true = _true_tdoa(POS, mics)
    err = np.abs(delays - true)
    assert delays[0] == 0.0 and valid[0]
    for i in range(1, len(mics)):
        assert valid[i]
        assert err[i] < 4e-6


def test_coarse_alignment_tolerance():
    """Perturb the coarse onset by +-1 sample. Phase refinement must still pick
    the correct branch (TDOA unchanged to sub-sample)."""
    wf, sched, mics, coarse = _scene()
    true = _true_tdoa(POS, mics)
    base, _ = phase_tdoa(wf, sched.segments, SR, coarse, reference_mic=0)
    for sign in (-1, +1):
        pert = coarse + sign / SR
        delays, valid = phase_tdoa(wf, sched.segments, SR, pert, reference_mic=0)
        for i in range(1, len(mics)):
            assert valid[i]
            assert abs(delays[i] - true[i]) < 4e-6
            assert abs(delays[i] - base[i]) < 4e-6


def test_common_reflection_phase():
    """A per-frequency phase shift shared by all mics cancels in the
    differential and leaves TDOA unchanged."""
    wf, sched, mics, coarse = _scene()
    true = _true_tdoa(POS, mics)
    refl_phase = np.array([0.3, -0.7, 1.1, -1.9, 2.4])
    wf2 = _apply_common_phase(wf, sched, refl_phase)
    arr = coherent_onset(wf2, sched)
    coarse2 = median_tof(arr, axis=1)
    delays, valid = phase_tdoa(wf2, sched.segments, SR, coarse2, reference_mic=0)
    for i in range(1, len(mics)):
        assert valid[i]
        assert abs(delays[i] - true[i]) < 4e-6


def test_tone_order_invariance():
    """Shuffle Method C transmission order; recovered TDOA must be unchanged."""
    wfA, schedA, micsA, coarseA = _scene(order=[0, 1, 2, 3, 4])
    wfB, schedB, _, coarseB = _scene(order=[4, 2, 0, 3, 1])
    true = _true_tdoa(POS, micsA)
    dA, vA = phase_tdoa(wfA, schedA.segments, SR, coarseA, 0)
    dB, vB = phase_tdoa(wfB, schedB.segments, SR, coarseB, 0)
    for i in range(1, len(micsA)):
        assert vA[i] and vB[i]
        assert abs(dA[i] - true[i]) < 4e-6
        assert abs(dB[i] - true[i]) < 4e-6
        assert abs(dA[i] - dB[i]) < 4e-6


def test_snr_sweep_reports_distribution():
    """Report median and p95 TDOA error across an SNR sweep (not just pass)."""
    true = _true_tdoa(POS, diamond_mics())
    rows = []
    for noise in (0.0, 0.01, 0.02, 0.05):
        errs = []
        for seed in range(20):
            wf, sched, mics, coarse = _scene(noise=noise, seed=seed)
            delays, valid = phase_tdoa(wf, sched.segments, SR, coarse, 0)
            for i in range(1, len(mics)):
                if valid[i]:
                    errs.append(abs(delays[i] - true[i]) * 1e6)  # us
        errs = np.array(errs)
        rows.append((noise, float(np.median(errs)), float(np.percentile(errs, 95))))
    for noise, med, p95 in rows:
        print(f"noise={noise:.3f} median_err={med:.3f}us p95_err={p95:.3f}us")
    assert rows[0][1] < 4.0   # clean: median below one sample
    assert rows[0][2] < 4.0   # clean: p95 below one sample


def test_failure_detection_on_destroyed_coherence():
    """When phase coherence is destroyed (independent per-mic per-tone carrier
    phase shift), the refinement must reject (valid=False), not return a
    precise-looking lie."""
    sched = build_schedule(FREQS, TONE, SR)
    mics = diamond_mics()
    # build a corrupted waveform: same amplitudes/timing, but each (mic, tone)
    # gets an INDEPENDENT carrier phase shift -> no shared linear trend.
    wf_bad = np.zeros((mics.shape[0], int((sched.duration + 0.1) * SR) + 1))
    rng = np.random.default_rng(7)
    for k, seg in enumerate(sched.segments):
        for i in range(mics.shape[0]):
            ph = rng.uniform(-np.pi, np.pi)
            delay = (np.linalg.norm(POS) + np.linalg.norm(mics[i] - POS)) / SPEED
            a = int((seg.start + delay) * SR)
            b = a + int(seg.duration * SR)
            tt = np.arange(a, b) / SR
            wf_bad[i, a:b] = np.hanning(b - a) * np.cos(
                2.0 * np.pi * seg.freq * (tt - seg.start - delay) + ph)
    arr = coherent_onset(wf_bad, sched)
    coarse_bad = median_tof(arr, axis=1)
    delays, valid = phase_tdoa(wf_bad, sched.segments, SR, coarse_bad, 0)
    for i in range(1, len(mics)):
        assert not valid[i]

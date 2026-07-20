"""Isolated cross-frequency phase model.

This test proves the phase-slope estimator in isolation, BEFORE it is wired
into ARGUS. It contains NO envelope detection, NO 3D solving, NO Monte Carlo,
NO beam cones, NO noise, NO multipath.

Key architecture (the part the earlier attempt got wrong): for one microphone
we obtain ONE common coarse delay tau_pred for the whole five-tone set, then
evaluate every tone's complex response at that SAME predicted alignment
(known transmit segment start + global ADC clock + common coarse delay). We do
NOT localize each tone at its own independently detected envelope peak -- that
would inject a different detector error 2*pi*f_k*eps_k into every tone and
poison the slope fit.

Phase convention: received tone k at absolute time t is
    cos(2*pi*f_k*(t - seg.start_k - tau))
(transmit phase is 2*pi*f_k*seg.start_k; we know it). Demodulating at the burst
CENTRE t_c = seg.start_k + tau_pred + dur/2 and removing the deterministic
2*pi*f_k*dur/2 term leaves
    phi_k = 2*pi*f_k*(tau_pred - tau)        (mod 2*pi)
so a weighted fit of phi_k vs f gives slope 2*pi*(tau_pred - tau) ->
tau = tau_pred - slope/(2*pi). For two microphones the differential phase
    phi2_k - phi1_k = 2*pi*f_k*(tau1 - tau2)
recovers the exact TDOA tau2 - tau1.
"""

import numpy as np
import pytest


SR = 250_000
FREQS = [40_000, 42_000, 44_000, 46_000, 48_000]
TONE = 0.004          # seconds
GUARD = 0.002         # seconds between segments (keeps bursts separated)


def _seg_starts(order):
    t = 0.0
    starts = []
    for _ in order:
        starts.append(t)
        t += TONE + GUARD
    return starts


def synthesize(taus, order, refl_phase=None, amplitude=1.0):
    slots = list(order)
    starts = _seg_starts(order)
    fk = [FREQS[s] for s in slots]
    sk = [starts[s] for s in slots]
    n = int((max(sk) + TONE + max(taus) + 0.005) * SR) + 1
    wf = np.zeros((len(taus), n))
    for i, tau in enumerate(taus):
        for kk in range(len(slots)):
            f = fk[kk]
            s = sk[kk]
            a = int((s + tau) * SR)
            b = a + int(TONE * SR)
            tt = np.arange(a, b) / SR
            win = np.hanning(b - a)
            ph = 0.0 if refl_phase is None else refl_phase[slots[kk]]
            wf[i, a:b] += amplitude * win * np.cos(2 * np.pi * f * (tt - s - tau) + ph)
    return wf, fk, sk


def _mic_phases(wf, mic, fk, sk, tau_pred):
    """Complex phase of each tone for one mic, demodulated at that mic's OWN
    common coarse delay tau_pred (the per-mic envelope onset estimate, which
    is within one sample of the true delay so tau_pred - tau is small)."""
    phi = np.zeros(len(fk))
    for kk in range(len(fk)):
        f = fk[kk]
        s = sk[kk]
        a = int((s + tau_pred) * SR)
        b = a + int(TONE * SR)
        seg = wf[mic, a:b]
        tc = s + tau_pred + TONE / 2.0
        tt = (a + np.arange(b - a)) / SR - tc
        z = np.sum(seg * np.exp(-1j * 2 * np.pi * f * tt))
        ang = np.angle(z)
        # True phase is 2*pi*f*(tau_pred - tau). Removing the known 2*pi*f*(s +
        # dur/2) transmit/centre term leaves 2*pi*f*(tau_pred - tau); the per-
        # tone wrap is undone relative to that. Assumes tau_pred - tau is small
        # (it is: coarse onset is within one ADC sample of the true delay).
        offset = f * (s + TONE / 2.0)
        theta = ang - 2 * np.pi * offset + 2 * np.pi * np.round(
            offset - ang / (2 * np.pi))
        phi[kk] = theta
    return phi


def _fit_delay(f, phi, tau_pred):
    # phi = 2*pi*f*(tau_pred - tau); slope fit recovers tau.
    slope = np.polyfit(f, phi, 1)[0]
    return tau_pred - slope / (2 * np.pi)


def _fit_tdoa(f, phi_a, phi_b, tau_pred_a, tau_pred_b):
    # diff = 2*pi*f*((tau_pred_b - tau_pred_a) - (tau_b - tau_a))
    slope = np.polyfit(f, phi_b - phi_a, 1)[0]
    return (tau_pred_b - tau_pred_a) - slope / (2 * np.pi)


def _coarse(tau):
    """Realistic envelope onset anchor: the true delay rounded to the nearest
    ADC sample (the envelope detector's integer-sample resolution)."""
    return round(tau * SR) / SR


def test_single_mic_recovered_delay_is_exact_no_noise():
    tau = 12.3456e-6
    tau_pred = _coarse(tau)
    wf, fk, sk = synthesize([tau], range(5))
    f = np.array(fk, dtype=float)
    phi = _mic_phases(wf, 0, fk, sk, tau_pred)
    tau_hat = _fit_delay(f, phi, tau_pred)
    assert abs(tau_hat - tau) < 1e-9


def test_delay_recovery_below_envelope_floor():
    for frac in np.linspace(0.0, 0.9, 10) * (1.0 / SR):
        tau = 20e-6 + frac
        tau_pred = _coarse(tau)
        wf, fk, sk = synthesize([tau], range(5))
        f = np.array(fk, dtype=float)
        phi = _mic_phases(wf, 0, fk, sk, tau_pred)
        tau_hat = _fit_delay(f, phi, tau_pred)
        assert abs(tau_hat - tau) < 1e-9


def test_two_mics_recovered_tdoa_is_exact():
    tau1 = 11.234e-6
    tau2 = tau1 + 53.7e-6
    wf, fk, sk = synthesize([tau1, tau2], range(5))
    f = np.array(fk, dtype=float)
    # each mic gets its OWN coarse onset (as the real envelope detector gives)
    p1 = _mic_phases(wf, 0, fk, sk, _coarse(tau1))
    p2 = _mic_phases(wf, 1, fk, sk, _coarse(tau2))
    tdoa = _fit_tdoa(f, p1, p2, _coarse(tau1), _coarse(tau2))
    assert abs(tdoa - 53.7e-6) < 1e-9


def test_common_reflection_phase_cancels():
    rng = np.random.default_rng(3)
    refl_phase = rng.uniform(-np.pi, np.pi, len(FREQS))
    tau1 = 11.234e-6
    tau2 = tau1 + 53.7e-6
    wf, fk, sk = synthesize([tau1, tau2], range(5), refl_phase=refl_phase)
    f = np.array(fk, dtype=float)
    p1 = _mic_phases(wf, 0, fk, sk, _coarse(tau1))
    p2 = _mic_phases(wf, 1, fk, sk, _coarse(tau2))
    tdoa = _fit_tdoa(f, p1, p2, _coarse(tau1), _coarse(tau2))
    assert abs(tdoa - 53.7e-6) < 1e-9


def test_frequency_order_invariance():
    tau1 = 11.234e-6
    tau2 = tau1 + 53.7e-6
    wfA, fkA, skA = synthesize([tau1, tau2], range(5))
    wfB, fkB, skB = synthesize([tau1, tau2], [4, 2, 0, 3, 1])
    fA = np.array(fkA, dtype=float)
    fB = np.array(fkB, dtype=float)
    pA0 = _mic_phases(wfA, 0, fkA, skA, _coarse(tau1))
    pA1 = _mic_phases(wfA, 1, fkA, skA, _coarse(tau2))
    pB0 = _mic_phases(wfB, 0, fkB, skB, _coarse(tau1))
    pB1 = _mic_phases(wfB, 1, fkB, skB, _coarse(tau2))
    tdoaA = _fit_tdoa(fA, pA0, pA1, _coarse(tau1), _coarse(tau2))
    tdoaB = _fit_tdoa(fB, pB0, pB1, _coarse(tau1), _coarse(tau2))
    assert abs(tdoaA - 53.7e-6) < 1e-9
    assert abs(tdoaA - tdoaB) < 1e-9

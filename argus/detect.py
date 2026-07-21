"""Matched-filter arrival-time detection.

Each frequency segment is a different tone, so we correlate each mic channel
with a short tone-burst template at that frequency (carrier-coherent matched
filter, frequency-selective so overlapping Method-C bursts at nearby tones stay
separated). The correlation peaks at the burst onset in the channel; converting
the peak index back to a time gives the arrival (ToF) for that mic on that
frequency. Method C yields 5 independent ToF estimates per mic; we median them
to reject unstable echoes.
"""

import numpy as np

from .schedule import Schedule


def _matched_env(sig: np.ndarray, freq: float, sr: float, tone: float) -> np.ndarray:
    """Carrier-coherent matched filter: correlate the channel with a short tone
    burst template of length `tone` seconds at `freq`. The correlation peaks at
    the burst onset in the channel. Returned as magnitude (frequency-selective,
    so overlapping Method-C bursts at nearby tones stay separated)."""
    nref = max(1, int(tone * sr))
    t = np.arange(nref) / sr
    ref = np.exp(1j * 2.0 * np.pi * freq * t)
    corr = np.correlate(sig, ref, mode="full")
    return np.abs(corr)


def _energy(sig: np.ndarray, win: int) -> np.ndarray:
    """Short-time energy: moving average of the squared signal. For a rectangular
    tone burst this is a clean step (high inside, ~0 outside), so the rising
    edge marks the true burst onset. (The Hilbert envelope of a tone ripples at
    2f and is useless for edge detection -- energy is not.)"""
    sq = sig * sig
    if win < 1:
        win = 1
    k = np.ones(win) / win
    return np.convolve(sq, k, mode="same")


def arrival_times(
    waveforms: np.ndarray,
    schedule: Schedule,
    speed: float = 343.0,
    range_window: float = 20.0,
) -> np.ndarray:
    """Return arrival times (seconds) for each mic/segment, relative to emission.

    Each segment is a different tone, so we correlate each mic channel with a
    short tone-burst template at that frequency (carrier-coherent matched
    filter, frequency-selective so overlapping Method-C bursts at nearby tones
    stay separated). The correlation peaks at the burst onset in the channel;
    converting the peak index back to a time gives the arrival. Time-
    unambiguous and degrades gracefully under noise (peak survives mild
    broadband noise).
    """
    n_mics = waveforms.shape[0]
    arrivals = np.zeros((n_mics, len(schedule.segments)))
    sr = schedule.sample_rate
    max_rt = (range_window * 2.0) / speed + 0.002
    for j, seg in enumerate(schedule.segments):
        i0 = int(seg.start * sr)  # emission time; arrival must be at/after this
        nref = max(1, int(seg.duration * sr))
        for i in range(n_mics):
            corr = _matched_env(waveforms[i], seg.freq, sr, seg.duration)
            # search the peak only within this segment's post-emission window so
            # a stronger later burst cannot be mistaken for this one
            lo = i0 + nref - 1
            hi = min(len(corr), i0 + int(max_rt * sr) + nref - 1)
            pk = lo + int(np.argmax(corr[lo:hi]))
            onset_idx = pk - (nref - 1)
            arrivals[i, j] = onset_idx / sr - seg.start
    return arrivals


def median_tof(arrivals: np.ndarray, axis: int = 1) -> np.ndarray:
    """Median ToF across frequency segments for each mic."""
    return np.median(arrivals, axis=axis)


def phase_tdoa(
    waveforms: np.ndarray,
    segments: list,
    sample_rate: float,
    coarse_tof_per_mic: np.ndarray,
    reference_mic: int = 0,
    phase_correction: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sub-sample TDOA from the cross-frequency differential phase slope.

    Pipeline (does NOT replace coarse ToF, does NOT re-detect per-tone peaks):

        envelopes -> one coarse delay per mic (caller-supplied)
        -> all five tones evaluated at that mic's SAME predicted alignment
        -> complex coefficient per (mic, tone)
        -> differential phase vs the reference mic, slope over frequency
        -> sub-sample TDOA.

    For each tone k at frequency f_k, segment start s_k, duration d, microphone
    i with coarse delay tau_c[i], we demodulate at the predicted burst centre
    t_c = s_k + tau_c[i] + d/2 using the ACTUAL waveform:

        z[i,k] = sum_m wf[i, a:b] * exp(-j*2*pi*f_k*(t_m - t_c))

    where a:b = [s_k+tau_c[i], s_k+tau_c[i]+d] in samples. Removing the known
    2*pi*f_k*(s_k + d/2) transmit/centre term leaves

        phi[i,k] = 2*pi*f_k*(tau_c[i] - tau_true[i])      (unwrapped)

    The differential vs the reference mic removes the common mode, and the slope
    of phi[i,k] - phi[ref,k] over f_k recovers tau_true[i] - tau_true[ref].

    Returns (delays, valid) where `delays[i] = tau_true[i] - tau_true[ref]`
    and `valid[i]` is True when the per-tone phases fit the linear model well
    (coherence held). When coherence is destroyed the delays are rejected.
    """
    sr = float(sample_rate)
    n_mics = waveforms.shape[0]
    n_seg = len(segments)
    freqs = np.array([seg.freq for seg in segments], dtype=float)
    starts = np.array([seg.start for seg in segments], dtype=float)
    durs = np.array([seg.duration for seg in segments], dtype=float)
    tau_c = np.asarray(coarse_tof_per_mic, dtype=float)

    phi = np.zeros((n_mics, n_seg))
    for i in range(n_mics):
        for k in range(n_seg):
            f = freqs[k]
            s = starts[k]
            d = durs[k]
            a = int((s + tau_c[i]) * sr)
            b = a + int(d * sr)
            if b > waveforms.shape[1]:
                b = waveforms.shape[1]
            if b <= a:
                phi[i, k] = 0.0
                continue
            tm = (a + np.arange(b - a)) / sr
            tc = s + tau_c[i] + d / 2.0
            z = np.sum(waveforms[i, a:b] * np.exp(-1j * 2.0 * np.pi * f * (tm - tc)))
            ang = np.angle(z)
            offset = f * (s + d / 2.0)
            # unalias to the unwrapped 2*pi*f*(tau_c - tau_true)
            theta = ang - 2.0 * np.pi * offset + 2.0 * np.pi * np.round(
                offset - ang / (2.0 * np.pi))
            phi[i, k] = theta

    # apply known per-(mic, freq) phase correction (calibration)
    if phase_correction is not None:
        phi = phi - phase_correction

    ref = phi[reference_mic]
    delays = np.zeros(n_mics)
    valid = np.zeros(n_mics, dtype=bool)
    for i in range(n_mics):
        if i == reference_mic:
            delays[i] = 0.0
            valid[i] = True
            continue
        diff = phi[i] - ref                       # = 2*pi*f*((tau_c[i]-tau_c[ref]) - d_tau)
        slope, intercept = np.polyfit(freqs, diff, 1)
        d_tau = (tau_c[i] - tau_c[reference_mic]) - slope / (2.0 * np.pi)
        delays[i] = d_tau
        # coherence check: per-tone deviation from the fitted line
        fit = slope * freqs + intercept
        res = diff - fit
        # residual should be < ~half a cycle rms; large residual => incoherent
        valid[i] = np.sqrt(np.mean(res ** 2)) < 0.6

    return delays, valid


def coherent_onset(
    waveforms: np.ndarray,
    schedule: Schedule,
    speed: float = 343.0,
    range_window: float = 20.0,
) -> np.ndarray:
    """Per-mic coarse ToF using a PHASE-COHERENT matched filter.

    Unlike `arrival_times` (which correlates with exp(j*2*pi*f*t)), this
    references each tone to its emission time s_k via exp(j*2*pi*f*(t - s_k)).
    That makes the correlation peak independent of the carrier's transmit
    phase, so it works for delay-coherent waveforms (synthesize(coherent=True))
    and lands the onset within ~1 sample -- the anchor `phase_tdoa` needs.

    `arrival_times` is left unchanged; this is an additive sibling used only
    when the phase-refinement path is active.
    """
    n_mics = waveforms.shape[0]
    arrivals = np.zeros((n_mics, len(schedule.segments)))
    sr = schedule.sample_rate
    max_rt = (range_window * 2.0) / speed + 0.002
    for j, seg in enumerate(schedule.segments):
        i0 = int(seg.start * sr)
        nref = max(1, int(seg.duration * sr))
        win_ref = np.hanning(nref)
        for i in range(n_mics):
            t_ref = np.arange(nref) / sr
            # Hann-windowed reference: correlates to true alignment (a rectangular
            # reference instead peaks at the burst's high-amplitude middle and is
            # systematically late by several samples).
            ref = win_ref * np.exp(1j * 2.0 * np.pi * seg.freq * (t_ref - seg.start))
            corr = np.abs(np.correlate(waveforms[i], ref, mode="full"))
            lo = i0 + nref - 1
            hi = min(len(corr), i0 + int(max_rt * sr) + nref - 1)
            pk = lo + int(np.argmax(corr[lo:hi]))
            onset_idx = pk - (nref - 1)
            arrivals[i, j] = onset_idx / sr - seg.start
    return arrivals

"""Second-stage phase-consistency validator.

Keeps ``phase_tdoa`` unchanged.  Runs four independent checks on the
demodulated phases using evidence the primary fit does not use:

1. *Cross-band agreement* — lower vs upper tone subsets must give the same TDOA.
2. *Leave-one-tone-out (LOTO) stability* — removing any one tone must not
   change the recovered slope significantly.
3. *Microphone-subset geometry* — 3-of-4 reconstructions must cluster
   (requires ``mics`` argument).
4. *Envelope compatibility* — phase-refined differential delays must stay
   within a physically plausible interval around the coarse envelope TDOA.

Thresholds are derived from a clean/calibrated single-reflector scene and
then frozen for evaluation on held-out multipath data.

Any single check failure → reject the *entire* phase solution
(``valid = False`` for all microphones and fall back to envelope).
"""

import itertools

import numpy as np

from .schedule import Schedule


# ---------------------------------------------------------------------------
# Computing the four metrics  (no thresholds, just raw disagreement values)
# ---------------------------------------------------------------------------

def _cross_band_metric(waveforms, segments, sr, coarse, ref):
    """Maximum per-mic TDOA disagreement between lower and upper tone subsets."""
    lo = [s for i, s in enumerate(segments) if i < 2]   # 40k, 42k
    hi = [s for i, s in enumerate(segments) if i >= 3]  # 46k, 48k
    d_lo, _ = phase_tdoa(waveforms, lo, sr, coarse, ref)
    d_hi, _ = phase_tdoa(waveforms, hi, sr, coarse, ref)
    return np.max(np.abs(d_lo - d_hi))


def _loto_metric(waveforms, segments, sr, coarse, ref):
    """Maximum per-mic standard deviation of TDOA across leave-one-out refits."""
    n_seg = len(segments)
    all_d = []
    for omit in range(n_seg):
        sub = [s for i, s in enumerate(segments) if i != omit]
        d, _ = phase_tdoa(waveforms, sub, sr, coarse, ref)
        all_d.append(d)
    return np.max(np.std(all_d, axis=0))


def _envelope_metric(delays, coarse, ref):
    """Maximum per-mic absolute deviation of phase TDOA from coarse envelope
    differential (in seconds)."""
    coarse_diff = coarse - coarse[ref]
    return np.max(np.abs(delays - coarse_diff))


# ---------------------------------------------------------------------------
# Threshold calibration  (run once on clean data)
# ---------------------------------------------------------------------------

def calibrate_validator_thresholds(
    waveforms: np.ndarray,
    segments: list,
    sample_rate: float,
    coarse_tof_per_mic: np.ndarray,
    reference_mic: int = 0,
    mics: np.ndarray | None = None,
    speed: float = 343.0,
    beam_axis: np.ndarray | None = None,
    n_trials: int = 30,
) -> dict:
    """Derive 99th-percentile thresholds for every check from clean/calibrated
    single-reflector data.  Returns a dict like::

        {"cross_band": 4.2e-6, "loto": 3.1e-6, "envelope": 8.0e-6,
         "geom_spread": 0.012}
    """
    metrics = {"cross_band": [], "loto": [], "envelope": [], "geom_spread": []}
    sr = float(sample_rate)
    ref = reference_mic
    for t in range(n_trials):
        coarse = coarse_tof_per_mic
        d_prim, _ = phase_tdoa(waveforms, segments, sr, coarse, ref)
        metrics["cross_band"].append(_cross_band_metric(waveforms, segments, sr, coarse, ref))
        metrics["loto"].append(_loto_metric(waveforms, segments, sr, coarse, ref))
        metrics["envelope"].append(_envelope_metric(d_prim, coarse, ref))
        if mics is not None:
            from .solve import solve_point
            tof_ph = coarse[ref] + d_prim
            pts = []
            for sub in itertools.combinations(range(len(mics)), 3):
                pt = solve_point(tof_ph[list(sub)], mics[list(sub)], speed, beam_axis=beam_axis)
                pts.append(pt)
            spread = np.max([np.linalg.norm(p - np.mean(pts, axis=0)) for p in pts])
            metrics["geom_spread"].append(spread)
    # add a noise floor so the validator does not reject trials for
    # microscopically small deviations (clean calibration gives near-zero
    # 99th-percentile values that make the validator pathologically strict).
    _floors = {"cross_band": 2e-6, "loto": 2e-6, "envelope": 4e-6, "geom_spread": 0.005}
    return {k: max(float(np.percentile(v, 99)), _floors.get(k, 0.0))
            for k, v in metrics.items()}


# ---------------------------------------------------------------------------
# The validator itself
# ---------------------------------------------------------------------------

def validate_phase_consistency(
    waveforms: np.ndarray,
    segments: list,
    sample_rate: float,
    coarse_tof_per_mic: np.ndarray,
    thresholds: dict,
    reference_mic: int = 0,
    mics: np.ndarray | None = None,
    speed: float = 343.0,
    beam_axis: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Second-stage consistency checks on the phase-derived TDOA.

    Parameters
    ----------
    thresholds : dict  from ``calibrate_validator_thresholds()``.
        Must contain at least ``cross_band``, ``loto``, ``envelope``.
        May optionally contain ``geom_spread``.

    Returns
    -------
    delays : (n_mic,)  the primary phase_tdoa estimate.
    valid : (n_mic,)  False for ALL mics if *any* check fails.
    diagnostics : dict  the raw metric values for this trial.
    """
    sr = float(sample_rate)
    ref = reference_mic
    n_mic = waveforms.shape[0]

    d_prim, v_prim = phase_tdoa(waveforms, segments, sr, coarse_tof_per_mic, ref)
    diag = {}

    # 1  cross-band
    diag["cross_band"] = _cross_band_metric(waveforms, segments, sr, coarse_tof_per_mic, ref)
    cb_pass = diag["cross_band"] <= thresholds.get("cross_band", 1e9)

    # 2  LOTO
    diag["loto"] = _loto_metric(waveforms, segments, sr, coarse_tof_per_mic, ref)
    loto_pass = diag["loto"] <= thresholds.get("loto", 1e9)

    # 3  envelope compatibility
    diag["envelope"] = _envelope_metric(d_prim, coarse_tof_per_mic, ref)
    env_pass = diag["envelope"] <= thresholds.get("envelope", 1e9)

    # 4  microphone-subset geometry
    geom_pass = True
    if mics is not None and "geom_spread" in thresholds:
        from .solve import solve_point
        tof_ph = coarse_tof_per_mic[ref] + d_prim
        pts = []
        for sub in itertools.combinations(range(len(mics)), 3):
            pt = solve_point(tof_ph[list(sub)], mics[list(sub)], speed, beam_axis=beam_axis)
            pts.append(pt)
        spread = np.max([np.linalg.norm(p - np.mean(pts, axis=0)) for p in pts])
        diag["geom_spread"] = spread
        geom_pass = spread <= thresholds.get("geom_spread", 1e9)
    else:
        diag["geom_spread"] = 0.0

    diag["pass"] = all([cb_pass, loto_pass, env_pass, geom_pass])

    valid_out = np.full(n_mic, diag["pass"], dtype=bool)
    return d_prim, valid_out, diag

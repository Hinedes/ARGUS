"""Sensor sensitivity map: convert "the array is ill-conditioned" into a real
hardware requirement.

Run Monte Carlo trials over the field of view (range x horizontal angle x
vertical angle) and inject controlled timing uncertainty, then measure the 3D
position-error distribution. This answers:

   1. Required effective ToF precision for 5 / 10 / 20 cm mapping accuracy.
   2. Effect of increasing the microphone baseline.
   3. Effect of ADC rate -- achievable delay precision may be SUB-SAMPLE
      (interpolation, carrier phase, cross-frequency), so the sample interval
      is a raw grid, not a hard floor.
   4. How common-mode vs differential timing error hit range vs direction.
   5. The real chain: waveform conditions -> measured ToF error -> spatial error.

Two timing-error classes are separated because ARGUS is sensitive to them
differently:

  * common-mode : all mics delayed by ~the same amount -> corrupts RANGE,
                  not direction (a shared-clock ADC removes clock drift
                  between channels, but not independent arrival-estimation noise).
  * differential: each mic's arrival jitters independently -> corrupts
                  DIRECTION, and therefore creates the large lateral errors.

The reflector is always in the forward hemisphere; the gimbal beam axis is a
prior, so the mirror solution is removed before scoring.
"""

from dataclasses import dataclass, field

import numpy as np

from .acoustic_scene import ORIGIN, diamond_mics, Reflector
from .detect import arrival_times, median_tof, coherent_onset, phase_tdoa
from .schedule import build_schedule
from .solve import solve_point
from .synthesize import synthesize


SPEED = 343.0
FREQS = [40_000, 42_000, 44_000, 46_000, 48_000]
TONE = 0.004
SAMPLE_RATE = 250_000
BEAM_AXIS = np.array([0.0, 0.0, 1.0])


@dataclass
class MismatchConfig:
    """Controlled departures from the ideal signal model.

    Each field is a N(0, sigma) magnitude for drawing per-(mic,freq) values,
    *except* tone_amp_skew (fractional per-freq amplitude variation from
    equal weighting) and emitter_ring (resonator damping ratio, lower = more
    ring). Set all to 0.0 for the ideal model.
    """
    channel_phase_sigma: float = 0.0        # rad per (mic, freq)
    delay_skew_sigma: float = 0.0           # seconds per mic
    emitter_ring: float = 0.0               # damping ratio z (>0, lower = more ring)
    tone_amp_skew: float = 0.0              # relative std of per-freq amplitude
    reflection_phase_common_sigma: float = 0.0   # rad per freq, same on all mics
    reflection_phase_per_mic_sigma: float = 0.0  # rad per (mic, freq), independent


@dataclass
class CalibrationConfig:
    """How well the receiver chain is known and compensated.

    * perfect : the exact injected mismatch is subtracted.
    * stale   : calibration was taken earlier and mismatch has drifted since
                (drift_fraction is the fraction of original mismatch still
                uncompensated, e.g. 0.2 = 20 % of the true mismatch remains).
    * known_phase : shape (n_mic, n_freq) rad, subtracted from phi before fit.
    * known_skew  : shape (n_mic,) seconds, subtracted from coarse ToF.
    """
    known: bool = False
    stale_drift: float = 0.0
    known_phase: np.ndarray | None = None
    known_skew: np.ndarray | None = None


@dataclass
class MultipathConfig:
    """One secondary path: x_i(t) = a0*s(t-tau0,i) + a1*s(t-tau1,i).

    Sweeping these parameters tests the valid_fraction gate: as multipath
    becomes destructive, valid_fraction must fall BEFORE phase error spikes.
    ``delay_extra`` is the common extra delay beyond the direct path.
    ``delay_per_mic_sigma`` adds independent per-microphone variation.
    """
    amp_ratio: float = 0.0           # a1/a0 (0 = no multipath)
    delay_extra: float = 0.0         # seconds, common extra delay
    delay_per_mic_sigma: float = 0.0 # seconds per-mic variation in extra delay


def _field_points(ranges, h_angles, v_angles):
    pts = []
    for r in ranges:
        for ha in h_angles:
            for va in v_angles:
                x = r * np.tan(ha)
                z = r / np.cos(ha)
                y = z * np.tan(va)
                pts.append(np.array([x, y, z]))
    return pts


def _draw_phase_bias(n_mic, n_freq, sigma, rng):
    """Return (n_mic, n_freq) array of N(0, sigma) rad phase biases."""
    return rng.normal(0.0, sigma, (n_mic, n_freq)) if sigma > 0.0 else np.zeros((n_mic, n_freq))


def _draw_delay_skew(n_mic, sigma, rng):
    """Return (n_mic,) array of N(0, sigma) second delay skews."""
    return rng.normal(0.0, sigma, n_mic) if sigma > 0.0 else np.zeros(n_mic)


def _draw_tone_amps(n_freq, skew, rng):
    """Return (n_freq,) per-tone amplitude multipliers; mean 1.0, std `skew`."""
    if skew <= 0.0:
        return np.ones(n_freq)
    a = rng.lognormal(mean=-0.5 * skew ** 2, sigma=skew, size=n_freq)
    return a / a.mean()


def _emitter_burst(f, dur, sr, start, delay, damping=0.1):
    """Return the waveform of a single tone burst emitted by a resonant source.

    The emitter is modelled as a 2nd-order LTI system (damping ratio `damping`,
    resonant frequency = the tone frequency). The excitation is a rectangular
    pulse of length `dur` seconds starting at `start`. The step response of a
    2nd-order system has a frequency-dependent ring-up transient; phase and
    envelope both deviate from the ideal Hann-gated tone. Attenuation from
    the reflector is excluded (applied by the caller).

    This produces the *reverberant* burst shape, not a windowed sinusoid.
    The matched filter in the estimator assumes a Hann envelope, so this
    injection is a *model mismatch*.
    """
    z = max(1e-6, damping)                     # damping ratio
    sr_f = float(sr)
    om = 2.0 * np.pi * f                       # natural freq (rad/s)
    om_d = om * np.sqrt(1.0 - z * z)           # damped freq
    n = int(np.ceil((dur + 0.008) * sr_f)) + 1 # buffer with ring-down margin
    t = np.arange(n) / sr_f

    # step response of 2nd-order system: y_step(t) = 1 - exp(-z*om*t) *
    #   (cos(om_d*t) + z / sqrt(1-z^2) * sin(om_d*t))
    a = np.exp(-z * om * t)
    b = np.cos(om_d * t) + (z / np.sqrt(max(1e-12, 1.0 - z * z))) * np.sin(om_d * t)
    step = 1.0 - a * b

    # burst = step(t - start) - step(t - start - dur)
    s = int(start * sr_f)
    sig = np.zeros(int(start * sr_f + n) + 1)
    seg = np.zeros(n)
    seg[:n] = step[:n]
    sig[s:s + n] += seg
    e = s + int(dur * sr_f)
    if e + 1 < len(sig):
        sig[e:e + n] -= seg[:min(n, len(sig) - e)]
    # remove DC offset (step settles to 1, but we want zero-mean carrier)
    sig -= np.mean(sig[s:min(e + n, len(sig))])
    # normalise peak to 1
    peak = np.max(np.abs(sig))
    if peak > 0:
        sig /= peak
    return sig[:min(len(sig), int(start * sr_f) + n)]


def _draw_mismatch_params(mismatch, n_mic, n_freq, rng):
    """Draw a SINGLE set of mismatch parameters (session-fixed biases).

    Physical channel phase offsets, delay skew, reflection phases are constant
    across pulses (hardware is stable within one measurement session); only
    waveform noise varies per trial.  This function is called ONCE per
    ``run_waveform_chain`` call, and the returned params are applied
    identically to every trial.
    """
    if mismatch is None:
        return {}
    return {
        "tone_amps": _draw_tone_amps(n_freq, mismatch.tone_amp_skew, rng),
        "channel_phase": _draw_phase_bias(n_mic, n_freq, mismatch.channel_phase_sigma, rng),
        "delay_skew": _draw_delay_skew(n_mic, mismatch.delay_skew_sigma, rng),
        "emitter_ring": mismatch.emitter_ring,
        "reflection_phase": (
            _draw_phase_bias(1, n_freq, mismatch.reflection_phase_common_sigma, rng)[0][np.newaxis, :]
            + _draw_phase_bias(n_mic, n_freq, mismatch.reflection_phase_per_mic_sigma, rng)
        ),
    }


def _apply_mismatch_fixed(wf, sched, mics, P, drawn_params):
    """Apply a FIXED set of mismatch parameters (drawn once per session) to the
    waveform.  ``drawn_params`` is the dict returned by ``_draw_mismatch_params``.
    """
    if not drawn_params:
        return wf.copy()

    sr = sched.sample_rate
    out = wf.copy()
    n_mic = mics.shape[0]

    tone_amps = drawn_params.get("tone_amps", np.ones(len(sched.segments)))
    ch_phase = drawn_params.get("channel_phase",
                                 np.zeros((n_mic, len(sched.segments))))
    delay_skew = drawn_params.get("delay_skew", np.zeros(n_mic))
    rfl_total = drawn_params.get("reflection_phase",
                                  np.zeros((n_mic, len(sched.segments))))
    ring = drawn_params.get("emitter_ring", 0.0)

    for i in range(n_mic):
        for k, seg in enumerate(sched.segments):
            f = seg.freq
            dly = (np.linalg.norm(P - ORIGIN) + np.linalg.norm(mics[i] - P)) / SPEED
            dly += delay_skew[i]

            # burst start = segment emission + total delay
            a = int((seg.start + dly) * sr)
            d = int(seg.duration * sr)
            b = a + d
            if b > out.shape[1]:
                b = out.shape[1]
                d = b - a
            if d == 0:
                continue

            phi_off = ch_phase[i, k] + rfl_total[i, k]
            amp = tone_amps[k]

            if ring > 0.0:
                burst = _emitter_burst(f, seg.duration, sr, seg.start,
                                        dly, damping=ring)
                out[i, a:b] = burst[:d] * amp
            else:
                tt_burst = np.arange(a, b) / sr
                carrier = np.cos(2.0 * np.pi * f * (tt_burst - seg.start - dly) + phi_off)
                win = np.hanning(d) if d > 1 else np.ones(d)
                out[i, a:b] = win * carrier * amp

    return out


def apply_mismatch(wf, sched, mics, P, mismatch, rng):
    """Return (wf_mismatched, true_params) where each call draws FRESH random
    mismatches (used for one-shot tests).  For batch use call
    ``_draw_mismatch_params`` once and ``_apply_mismatch_fixed`` per trial."""
    if mismatch is None:
        return wf.copy(), {}
    drawn = _draw_mismatch_params(mismatch, mics.shape[0], len(sched.segments), rng)
    return _apply_mismatch_fixed(wf, sched, mics, P, drawn), drawn


def _apply_multipath(wf, mp_config, sr, rng):
    """Add one secondary path: x_i(t) += a1/a0 * x_i(t - tau_extra).

    The entire waveform is shifted by ``delay_extra`` (+ per-mic noise) and
    added back at reduced amplitude.  The leading edge of the shifted copy
    is zero-padded so no wrap-around contamination occurs.
    """
    if mp_config is None or mp_config.amp_ratio <= 0.0:
        return wf.copy()

    n_mic = wf.shape[0]
    out = wf.copy()
    common = int(mp_config.delay_extra * sr)
    if common <= 0:
        return out

    for i in range(n_mic):
        per_mic = int(rng.normal(0.0, mp_config.delay_per_mic_sigma * sr))
        shift = max(1, common + per_mic)
        shifted = np.zeros_like(out[i])
        if shift < len(shifted):
            shifted[shift:] = out[i, :len(shifted) - shift]
        out[i] += mp_config.amp_ratio * shifted
    return out


def _calibrate(coarse, phi, freqs, true_params, cal, rng=None):
    """Apply calibration corrections to coarse ToF and per-(mic, freq) phi.

    Returns (coarse_cal, phi_cal). When calibration is not known,
    returns the original values unchanged.
    """
    if cal is None or not cal.known:
        return coarse, phi

    cal_phase = cal.known_phase
    cal_skew = cal.known_skew

    # Stale calibration: add drift to the known values
    if cal.stale_drift > 0.0 and rng is not None:
        if "channel_phase" in true_params:
            drift = true_params["channel_phase"] - cal_phase
            cal_phase += drift * (1.0 - cal.stale_drift)
        if "delay_skew" in true_params:
            drift = true_params["delay_skew"] - cal_skew
            cal_skew += drift * (1.0 - cal.stale_drift)

    # Correct phase: subtract known phase bias from phi
    if cal_phase is not None:
        phi = phi - cal_phase

    # Correct delay skew: subtract from coarse
    if cal_skew is not None:
        coarse = coarse - cal_skew

    return coarse, phi


def envelope_tof(waveforms, schedule, sr):
    """Matched-filter onset (envelope peak, integer-sample precise)."""
    arr = arrival_times(waveforms, schedule)
    return median_tof(arr, axis=1)


def phase_refined_tof(waveforms, schedule, sr, freqs):
    """Placeholder for carrier-phase timing refinement (NOT YET IMPLEMENTED).

    The envelope detector already lands on the burst onset to integer-sample
    precision (~4 us at 250 kHz). Sub-sample refinement via the residual
    carrier phase across the burst is a planned next step -- it is NOT wired
    into the estimator path yet, so callers should use estimator="envelope".
    """
    raise NotImplementedError("carrier-phase refinement not yet implemented")


@dataclass
class SensitivityConfig:
    ranges: list[float] = field(default_factory=lambda: [1.0, 2.0, 3.0])
    h_angles: list[float] = field(default_factory=lambda: np.deg2rad([-20, -10, 0, 10, 20]).tolist())
    v_angles: list[float] = field(default_factory=lambda: np.deg2rad([-10, 0, 10]).tolist())
    # Two independent timing-error classes (seconds, 1-sigma):
    common_sigma: float = 0.0    # common-mode delay on ALL mics -> range error
    diff_sigma: float = 10e-6   # per-mic independent jitter -> direction error
    sr: int = SAMPLE_RATE
    baseline_half: float = 0.06
    tilt: float = np.deg2rad(12.0)
    beam_axis: np.ndarray = field(default_factory=lambda: BEAM_AXIS.copy())
    n_trials: int = 30
    refine: bool = False         # fractional-delay refinement (envelope -> sub-sample)
    waveform_noise: float = 0.0  # if >0, inject broadband noise on the
                                    # raw waveform and let detect.py measure ToF
    seed: int = 0
    reference_mic: int = 0       # mic whose coarse ToF anchors the phase TDOA


def _estimate_tof(waveforms, sched, sr, cfg):
    if cfg.refine:
        # carrier-phase refinement is built/tested in isolation (see
        # tests/test_phase_model.py) before being wired into this path.
        raise NotImplementedError("carrier-phase refinement not yet wired in")
    return envelope_tof(waveforms, sched, SPEED)


def _true_tof(P, mics):
    return (np.linalg.norm(P) + np.linalg.norm(mics - P, axis=1)) / SPEED


def run_monte_carlo(cfg: SensitivityConfig) -> dict:
    """Return a dict of error arrays and summary statistics over the FOV grid.

    Two timing-error classes are injected independently:
      * common-mode  : one shared delay on all mics (hits RANGE)
      * differential : per-mic independent jitter (hits DIRECTION)
    If cfg.waveform_noise > 0 the raw waveform is perturbed with broadband
    noise and detect.py measures the ToF itself (the real chain), instead of
    injecting jitter directly on the ideal estimate.
    """
    rng = np.random.default_rng(cfg.seed)
    mics = diamond_mics(tilt=cfg.tilt, half=cfg.baseline_half)
    pts = _field_points(cfg.ranges, cfg.h_angles, cfg.v_angles)

    errors = np.zeros((len(pts), cfg.n_trials))
    range_err = np.zeros_like(errors)
    dir_err = np.zeros_like(errors)
    for pi, P in enumerate(pts):
        sched = build_schedule(FREQS, TONE, cfg.sr)
        true_tof = _true_tof(P, mics)
        for t in range(cfg.n_trials):
            wf = synthesize(sched, [Reflector(P)], mics, SPEED)
            if cfg.waveform_noise > 0.0:
                wf = wf + rng.normal(0.0, cfg.waveform_noise, wf.shape)
            tof = _estimate_tof(wf, sched, cfg.sr, cfg)
            # common-mode: same offset on every mic; differential: independent
            common = rng.normal(0.0, cfg.common_sigma)
            diff = rng.normal(0.0, cfg.diff_sigma, tof.shape)
            tof = tof + common + diff
            est = solve_point(tof, mics, SPEED, beam_axis=cfg.beam_axis)
            err_vec = est - P
            errors[pi, t] = np.linalg.norm(err_vec)
            # range error = component along the boresight (z); direction = lateral
            range_err[pi, t] = abs(err_vec[2])
            dir_err[pi, t] = np.linalg.norm(err_vec[[0, 1]])
    return {
        "points": np.array(pts),
        "errors": errors,
        "range_err": range_err,
        "dir_err": dir_err,
        "median": np.median(errors, axis=1),
        "p95": np.percentile(errors, 95, axis=1),
        "overall_median": float(np.median(errors)),
        "overall_p95": float(np.percentile(errors, 95)),
        "overall_range_p95": float(np.percentile(range_err, 95)),
        "overall_dir_p95": float(np.percentile(dir_err, 95)),
    }


def run_waveform_chain(cfg: "SensitivityConfig",
                        mismatch: MismatchConfig | None = None,
                        calibration: CalibrationConfig | None = None,
                        mp_config: MultipathConfig | None = None,
                        validator_thresholds: dict | None = None) -> dict:
    """Run the ACTUAL waveform detector inside the sensitivity sweep.

    For every FOV scene this synthesizes coherent four-mic waveforms, runs
    ``coherent_onset`` for the per-mic coarse ToF, runs ``phase_tdoa`` for the
    sub-sample differential delay, reconstructs the point three ways, and records
    the JOINT structure -- NOT a collapsed scalar jitter.

    Pipeline (per trial):
        reflector geometry
            -> synthesize(coherent=True) four-mic waveforms
            -> coherent_onset          (coarse per-mic ToF)
            -> phase_tdoa              (sub-sample TDOA vs reference mic)
            -> solve_point             (reconstruct point)

    Reconstructions compared:
      * envelope     : solve from coarse ToF only.
      * phase        : coarse[ref] + phase differential delays, solved.
      * fallback     : phase delays where valid, else coarse ToF (envelope).

    The differential-delay error is recorded as a VECTOR (one entry per mic),
    preserving the cross-microphone correlation -- it is never reduced to a
    single independent Gaussian sigma.

    Static-sensor / single-common-reflection limitation is explicit: no motion
    compensation, no multipath, no per-microphone reflection response, no
    material phase distortion. Those come later.

    Returns nested arrays indexed [point, trial] plus summaries, so the caller
    can cut by range / angle / SNR / baseline / ADC rate independently.
    """
    rng = np.random.default_rng(cfg.seed)
    mics = diamond_mics(tilt=cfg.tilt, half=cfg.baseline_half)
    pts = _field_points(cfg.ranges, cfg.h_angles, cfg.v_angles)
    ref = cfg.reference_mic
    n_mic = mics.shape[0]

    # [point, trial]
    err_env = np.zeros((len(pts), cfg.n_trials))
    err_phase = np.zeros_like(err_env)
    err_fallback = np.zeros_like(err_env)
    ref_range_err = np.zeros_like(err_env)          # |coarse[ref] - true[ref]|
    valid_rate = np.zeros((len(pts), cfg.n_trials), dtype=bool)
    false_valid = np.zeros_like(valid_rate)          # valid=True but error > 0.1m
    # decomposed error components (for the phase-refined reconstruction)
    # beam_axis defines the range direction; horizontal/vertical are the
    # perpendicular camera-like axes (x = horizontal, y = vertical).
    err_range = np.zeros_like(err_env)               # along beam_axis
    err_horiz = np.zeros_like(err_env)               # x component
    err_vert = np.zeros_like(err_env)                # y component
    # same decomposition for envelope and fallback
    err_env_range   = np.zeros_like(err_env)
    err_env_horiz   = np.zeros_like(err_env)
    err_fb_range    = np.zeros_like(err_env)
    err_fb_horiz    = np.zeros_like(err_env)
    # joint differential TDOA error vector: [point, trial, mic]
    tdoa_err_vec = np.zeros((len(pts), cfg.n_trials, n_mic))
    # draw FIXED mismatch parameters using a deterministic sub-seed so they
    # are identical across separate calls to run_waveform_chain (needed for
    # calibration to match the injected mismatch).
    _mm_rng = np.random.default_rng(cfg.seed + 2**16)
    drawn_mm = _draw_mismatch_params(
        mismatch, n_mic, len(build_schedule(FREQS, TONE, cfg.sr).segments), _mm_rng
    ) if mismatch is not None else {}

    for pi, P in enumerate(pts):
        sched = build_schedule(FREQS, TONE, cfg.sr)
        true_tof = _true_tof(P, mics)
        true_tdoa = true_tof - true_tof[ref]       # what phase_tdoa should yield
        for t in range(cfg.n_trials):
            wf = synthesize(sched, [Reflector(P)], mics, SPEED, coherent=True)
            if cfg.waveform_noise > 0.0:
                wf = wf + rng.normal(0.0, cfg.waveform_noise, wf.shape)

            # ---- model-mismatch injection (same fixed bias every trial) ----
            if mismatch is not None:
                wf = _apply_mismatch_fixed(wf, sched, mics, P, drawn_mm)

            # ---- multipath (secondary delayed copy) -------------------------
            if mp_config is not None:
                wf = _apply_multipath(wf, mp_config, cfg.sr, rng)

            # ---- detector chain --------------------------------------------
            coarse = median_tof(coherent_onset(wf, sched), axis=1)

            # ---- calibration corrections -----------------------------------
            # (applied BEFORE phase_tdoa so alignment uses corrected coarse)
            if calibration is not None and calibration.known:
                if calibration.known_skew is not None:
                    coarse = coarse - calibration.known_skew

            phase_correction = (
                calibration.known_phase if (calibration is not None
                                            and calibration.known
                                            and calibration.known_phase is not None)
                else None)
            if validator_thresholds is not None:
                from .detect import validate_phase_consistency
                delays, valid, _diag = validate_phase_consistency(
                    wf, sched.segments, cfg.sr, coarse, validator_thresholds,
                    reference_mic=ref, mics=mics, speed=SPEED,
                    beam_axis=cfg.beam_axis)
            else:
                delays, valid = phase_tdoa(
                    wf, sched.segments, cfg.sr, coarse, ref,
                    phase_correction=phase_correction)

            # ---- three per-mic ToF vectors ---------------------------------
            tof_env = coarse.copy()
            tof_phase = coarse[ref] + delays        # anchor range on coarse[ref]
            tof_fb = tof_phase if np.all(valid) else coarse.copy()

            est_env = solve_point(tof_env, mics, SPEED, beam_axis=cfg.beam_axis)
            est_phase = solve_point(tof_phase, mics, SPEED, beam_axis=cfg.beam_axis)
            est_fb = solve_point(tof_fb, mics, SPEED, beam_axis=cfg.beam_axis)

            d_env = est_env - P
            d_ph  = est_phase - P
            d_fb  = est_fb - P
            env_n   = np.linalg.norm(d_env)
            ph_n    = np.linalg.norm(d_ph)
            fb_n    = np.linalg.norm(d_fb)
            env_r   = abs(d_env[2]); env_h = abs(d_env[0])
            ph_r    = abs(d_ph[2]);  ph_h = abs(d_ph[0]);  ph_v = abs(d_ph[1])
            fb_r    = abs(d_fb[2]);  fb_h = abs(d_fb[0])

            err_env[pi, t]         = env_n
            err_phase[pi, t]       = ph_n
            err_fallback[pi, t]    = fb_n
            err_range[pi, t]       = ph_r
            err_horiz[pi, t]       = ph_h
            err_vert[pi, t]        = ph_v
            err_env_range[pi, t]   = env_r
            err_env_horiz[pi, t]   = env_h
            err_fb_range[pi, t]    = fb_r
            err_fb_horiz[pi, t]    = fb_h
            ref_range_err[pi, t]   = abs(coarse[ref] - true_tof[ref])
            valid_rate[pi, t]      = bool(np.all(valid))
            false_valid[pi, t]     = bool(np.all(valid)) and ph_n > 0.1
            tdoa_err_vec[pi, t]    = delays - true_tdoa

    def _sum(a):
        if a.size == 0:
            return {"median": float("nan"), "p95": float("nan"), "mean": float("nan")}
        return {
            "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)),
            "mean": float(np.mean(a)),
        }

    # the fixed mismatch params (same across all trials) for calibration
    _true_mean = {}
    if mismatch is not None and drawn_mm:
        _true_mean = {k: v for k, v in drawn_mm.items()
                       if isinstance(v, np.ndarray)}

    return {
        "points": np.array(pts),
        "ranges": list(cfg.ranges),
        "h_angles": list(cfg.h_angles),
        "v_angles": list(cfg.v_angles),
        "sr": cfg.sr,
        "baseline_half": cfg.baseline_half,
        "waveform_noise": cfg.waveform_noise,
        "reference_mic": ref,
        "n_mic": n_mic,
        # raw joint arrays [point, trial] / [point, trial, mic]
        "err_env": err_env,
        "err_phase": err_phase,
        "err_fallback": err_fallback,
        "err_range": err_range,
        "err_horiz": err_horiz,
        "err_vert": err_vert,
        "err_env_range": err_env_range,
        "err_env_horiz": err_env_horiz,
        "err_fb_range": err_fb_range,
        "err_fb_horiz": err_fb_horiz,
        "ref_range_err": ref_range_err,
        "valid_rate": valid_rate,
        "false_valid": false_valid,
        "tdoa_err_vec": tdoa_err_vec,
        "true_params_mean": _true_mean,
        # summaries
        "env": _sum(err_env),
        "env_horiz": _sum(err_env_horiz),
        "env_range": _sum(err_env_range),
        "phase": _sum(err_phase),
        "fallback": _sum(err_fallback),
        "fallback_horiz": _sum(err_fb_horiz),
        "fallback_range": _sum(err_fb_range),
        "valid_fraction": float(np.mean(valid_rate)),
        "false_valid_fraction": float(np.mean(false_valid)),
        "cond_false_valid_fraction": float(
            np.mean(false_valid) / max(np.mean(valid_rate), 0.01)
        ),
        "ref_range": _sum(ref_range_err),
        "phase_horiz": _sum(err_horiz),
        "phase_vert": _sum(err_vert),
        "phase_range": _sum(err_range),
        # accuracy WHEN valid: only trials where phase refinement was accepted
        "phase_when_valid": _sum(err_phase[valid_rate]),
        "valid_fraction_when_valid": 1.0,
    }


def detector_compare(cfg: "SensitivityConfig") -> dict:
    """Two pipelines on the SAME scenes, to prove phase refinement adds value
    beyond merely fixing the old Hann/rectangular correlation mismatch:

        coherent_onset + envelope solver      (coarse ToF solved directly)
        coherent_onset + phase_tdoa solver    (sub-sample differential added)

    Both use the SAME coherent_onset coarse anchor, so any gap is purely the
    phase-slope refinement, not a detector-detail difference.
    """
    res = run_waveform_chain(cfg)
    return {
        "envelope": res["env"],
        "phase": res["phase"],
        "fallback": res["fallback"],
        "valid_fraction": res["valid_fraction"],
        "phase_when_valid": res["phase_when_valid"],
        "ref_range": res["ref_range"],
    }


def _build_calibration(true_params, mode="none", stale_drift=0.0, rng=None,
                        config_sigma: float | None = None,
                        config_skew_sigma: float | None = None):
    """Return a CalibrationConfig for the given true injected mismatch.

    For stale calibration the known (previous) values are the current true
    biases minus an INDEPENDENTLY drawn drift, so the drift magnitude is
    ``stale_drift * config_sigma``, not derived from the realized biases.
    This avoids leaking the direction of the current bias into the stale
    estimate.
    """
    if mode == "none":
        return CalibrationConfig(known=False)

    known_phase = None
    known_skew = None
    if "channel_phase" in true_params:
        known_phase = true_params["channel_phase"].copy()
    if "delay_skew" in true_params:
        known_skew = true_params["delay_skew"].copy()

    if mode == "stale" and stale_drift > 0.0 and rng is not None:
        # independent drift, sigma = stale_drift * config_sigma for each variable
        if known_phase is not None and config_sigma is not None:
            sigma_drift = stale_drift * config_sigma
            drift = rng.normal(0.0, sigma_drift, known_phase.shape)
            known_phase = true_params["channel_phase"] - drift
        if known_skew is not None and config_skew_sigma is not None:
            sigma_drift = stale_drift * config_skew_sigma
            drift = rng.normal(0.0, sigma_drift, known_skew.shape)
            known_skew = true_params["delay_skew"] - drift

    return CalibrationConfig(known=True, known_phase=known_phase, known_skew=known_skew)


def run_mismatch_sweep(cfg: "SensitivityConfig",
                        mismatches: dict[str, MismatchConfig],
                        cal_modes: tuple[str, ...] = ("none", "perfect", "stale"),
                        stale_drift: float = 0.15) -> dict:
    """Run ``run_waveform_chain`` for each (label, mismatch, cal_mode) and
    return a dict keyed by ``"{mismatch_label}__{cal_mode}"`` containing the
    summary statistics (median/p95 position error, valid_fraction, phase error
    when valid).

    ``mismatches`` maps a label to a MismatchConfig.
    ``cal_modes`` selects which calibration regimes to test.
    ``stale_drift`` is the fraction of true mismatch left uncompensated when
    cal_mode == "stale".

    Calibration is built from the ``true_params_mean`` of a preliminary
    ``run_waveform_chain`` call (same cfg.seed, so mismatches match).
    """
    rng = np.random.default_rng(cfg.seed + 999)
    out = {}
    for label, mm in mismatches.items():
        # run once blindly to collect true_params_mean
        probe = run_waveform_chain(cfg, mismatch=mm)
        tp_mean = probe.get("true_params_mean", {})
        for cm in cal_modes:
            key = f"{label}__{cm}"
            if cm == "none":
                cal = None
            else:
                # pass the original config sigma values for stale drift
                cfg_s = mm.channel_phase_sigma if hasattr(mm, 'channel_phase_sigma') else None
                cfg_sk = mm.delay_skew_sigma if hasattr(mm, 'delay_skew_sigma') else None
                cal = _build_calibration(tp_mean, mode=cm,
                                          stale_drift=stale_drift, rng=rng,
                                          config_sigma=cfg_s,
                                          config_skew_sigma=cfg_sk)
            res = run_waveform_chain(cfg, mismatch=mm, calibration=cal)
            out[key] = {
                "env_p95": res["env"]["p95"],
                "phase_p95": res["phase"]["p95"],
                "fallback_p95": res["fallback"]["p95"],
                "valid_fraction": res["valid_fraction"],
                "phase_when_valid_p95": res["phase_when_valid"]["p95"],
            }
    return out


def run_hardware_map(cfg: "SensitivityConfig",
                      baselines_half=(0.03, 0.06, 0.12),
                      snrs=(0.0, 0.01, 0.03),
                      adc_rates=(250_000, 125_000, 62_500),
                      mismatch: MismatchConfig | None = None,
                      calibration: CalibrationConfig | None = None) -> dict:
    """Produce the hardware-choice table: baseline × range × FOV × SNR × ADC
    rate → envelope p95 / phase p95 / valid_fraction / phase p95 when valid.

    Each entry is a fixed FOV (the one in cfg) run at that geometry.  The
    returned dict is keyed ``{baseline_mm}__{snr}__{sr_khz}`` with sub-keys
    for each range.
    """
    out = {}
    for half in baselines_half:
        for noise in snrs:
            for sr in adc_rates:
                k = f"{int(half*1e3)}mm__{noise:.0e}__{int(sr/1e3)}k"
                sub = {}
                for r in cfg.ranges:
                    sub_cfg = SensitivityConfig(
                        ranges=[r], h_angles=cfg.h_angles, v_angles=cfg.v_angles,
                        sr=sr, baseline_half=half, tilt=cfg.tilt,
                        beam_axis=cfg.beam_axis, n_trials=cfg.n_trials,
                        waveform_noise=noise, seed=cfg.seed,
                        reference_mic=cfg.reference_mic,
                    )
                    res = run_waveform_chain(sub_cfg, mismatch=mismatch,
                                              calibration=calibration)
                    sub[r] = {
                        "env_p95": res["env"]["p95"],
                        "phase_p95": res["phase"]["p95"],
                        "fallback_p95": res["fallback"]["p95"],
                        "valid_fraction": res["valid_fraction"],
                        "phase_when_valid_p95": res["phase_when_valid"]["p95"],
                    }
                out[k] = sub
    return out


def run_hardware_assessment(
    cfg: "SensitivityConfig",
    device_seeds: tuple[int, ...] = (0, 10, 20),
    baselines_half: tuple[float, ...] = (0.06, 0.12),
    cal_modes: tuple[str, ...] = ("none", "perfect", "stale"),
    stale_drift: float = 0.15,
    mismatch_sigma: float = 0.3,
    skew_sigma: float = 10e-6,
) -> dict:
    """Run the full hardware-choice experiment across simulated devices.

    For each ``device_seed`` the fixed per-device hardware biases are drawn
    (channel phase + delay skew).  Each device is evaluated at every
    baseline × calibration-mode combination across the full FOV in *cfg*.

    The output dict is keyed ``{baseline_mm}__{cal_mode}`` and contains,
    averaged over device seeds:

    * phase_p95, phase_horiz_p95, phase_vert_p95, phase_range_p95
    * env_p95, env_horiz_p95
    * fallback_p95
    * valid_fraction, false_valid_fraction
    * phase_when_valid_p95

    This is the data for the 12‑cm‑vs‑6‑cm angular comparison the hardware
    decision needs: away from boresight, 12 cm must beat 6 cm in the
    horizontal/vertical error components.
    """
    out: dict = {}
    for half in baselines_half:
        for cm in cal_modes:
            key = f"{int(half*1e3)}mm__{cm}"
            metrics = {
                "phase_p95": [], "phase_horiz_p95": [], "phase_vert_p95": [],
                "phase_range_p95": [], "env_p95": [], "env_horiz_p95": [],
                "fallback_p95": [], "valid_fraction": [],
                "false_valid_fraction": [],
                "phase_when_valid_p95": [],
            }
            for ds in device_seeds:
                sub_cfg = SensitivityConfig(
                    ranges=cfg.ranges, h_angles=cfg.h_angles, v_angles=cfg.v_angles,
                    sr=cfg.sr, baseline_half=half, tilt=cfg.tilt,
                    beam_axis=cfg.beam_axis, n_trials=cfg.n_trials,
                    waveform_noise=cfg.waveform_noise, seed=ds,
                    reference_mic=cfg.reference_mic,
                )
                # fixed hardware biases for this device
                mm = MismatchConfig(
                    channel_phase_sigma=mismatch_sigma,
                    delay_skew_sigma=skew_sigma,
                    tone_amp_skew=0.2,
                )
                # probe for calibration
                probe = run_waveform_chain(sub_cfg, mismatch=mm)
                tp_mean = probe.get("true_params_mean", {})

                cal = None
                if cm != "none":
                    rng_drift = np.random.default_rng(ds + 2**18)
                    cal = _build_calibration(
                        tp_mean, mode=cm, stale_drift=stale_drift,
                        rng=rng_drift,
                        config_sigma=mismatch_sigma,
                        config_skew_sigma=skew_sigma)

                res = run_waveform_chain(sub_cfg, mismatch=mm, calibration=cal)
                metrics["phase_p95"].append(res["phase"]["p95"])
                metrics["phase_horiz_p95"].append(res["phase_horiz"]["p95"])
                metrics["phase_vert_p95"].append(res["phase_vert"]["p95"])
                metrics["phase_range_p95"].append(res["phase_range"]["p95"])
                metrics["env_p95"].append(res["env"]["p95"])
                metrics["env_horiz_p95"].append(res["env_horiz"]["p95"])
                metrics["fallback_p95"].append(res["fallback"]["p95"])
                metrics["valid_fraction"].append(res["valid_fraction"])
                metrics["false_valid_fraction"].append(res["false_valid_fraction"])
                metrics["phase_when_valid_p95"].append(res["phase_when_valid"]["p95"])

            # aggregate across device seeds: report median over device seeds
            out[key] = {k: float(np.median(v)) for k, v in metrics.items()}
    return out


def run_multipath_sweep(
    cfg: "SensitivityConfig",
    amp_ratios: tuple[float, ...] = (0.0, 0.1, 0.3, 0.5),
    delay_extras: tuple[float, ...] = (0.0005, 0.001, 0.003),
    per_mic_sigmas: tuple[float, ...] = (0.0, 50e-6),
    mp_config_extra: dict | None = None,
) -> dict:
    """Sweep the secondary-path parameters and report error/validity
    degradation before accuracy collapse.

    For each (amp_ratio, delay_extra, per_mic_sigma) the waveform chain runs
    with the secondary path injected.  The output key is
    ``{amp_label}__{delay_label}__{permic_label}`` and each entry contains:

    * phase_p95, env_p95, fallback_p95
    * valid_fraction
    * false_valid_fraction
    * the *ratio* phase_p95 / valid_fraction  (indicator of the gate:
      if this is large while valid_fraction is still high, the coherence
      check is failing too slowly).
    """
    out = {}
    for amp in amp_ratios:
        for delay in delay_extras:
            for per_mic in per_mic_sigmas:
                key = f"a{amp:.1f}__d{delay*1e6:.0f}us__pm{per_mic*1e6:.0f}us"
                mp = MultipathConfig(amp_ratio=amp, delay_extra=delay,
                                     delay_per_mic_sigma=per_mic)
                res = run_waveform_chain(cfg, mp_config=mp)
                ph = res["phase"]["p95"]
                env = res["env"]["p95"]
                fb = res["fallback"]["p95"]
                vf = res["valid_fraction"]
                fv = res["false_valid_fraction"]
                cfv = res["cond_false_valid_fraction"]
                ratio = ph / max(vf, 0.01)
                out[key] = {
                    "phase_p95": ph, "env_p95": env, "fallback_p95": fb,
                    "valid_fraction": vf, "false_valid_fraction": fv,
                    "cond_false_valid_fraction": cfv,
                    "phase_per_valid_ratio": ratio,
                }
    return out


def print_multipath_table(sweep_result: dict) -> None:
    """Print a human-readable table from the run_multipath_sweep output."""
    from operator import itemgetter
    keys = sorted(sweep_result)
    print(f"{'Params':<35} {'Ph_p95':>8} {'Env_p95':>8} {'Fb_p95':>8} "
          f"{'Vf':>6} {'Fv':>6} {'cFv':>6}")
    print("-" * 85)
    for k in keys:
        r = sweep_result[k]
        print(f"{k:<35} {r['phase_p95']:8.4f} {r['env_p95']:8.4f} "
              f"{r['fallback_p95']:8.4f} {r['valid_fraction']:6.3f} "
              f"{r['false_valid_fraction']:6.3f} {r['cond_false_valid_fraction']:6.3f}")


def required_tof_precision(
    targets_cm=(5, 10, 20),
    ranges=(1.0, 2.0, 3.0),
    fov_bins_deg=(0, 10, 20),   # |horizontal| angle bins (deg) to report
    baselines_half=(0.03, 0.06, 0.12, 0.24),
    sigma_grid=(1e-6, 2e-6, 5e-6, 10e-6, 20e-6, 50e-6),
    **kw,
):
    """The load-bearing artifact: a table of the timing precision ARGUS needs.

    For each (range, FOV-angle-bin, baseline, target-accuracy) it returns the
    differential-ToF sigma (1-sigma, seconds) that keeps the 95th-percentile
    error under the target, or None if even the smallest tested sigma fails.

    Differential sigma drives DIRECTION (the hard part); the common-mode sigma
    that keeps RANGE error under target is reported separately as `common_sigma`.
    Returns a nested dict keyed by baseline -> range -> fov_bin -> target_cm.
    """

    h = kw.pop("h_angles", np.deg2rad([-15, 0, 15]).tolist())
    v = kw.pop("v_angles", np.deg2rad([-8, 0, 8]).tolist())
    n_trials = kw.pop("n_trials", 40)
    seed = kw.pop("seed", 7)
    out = {}
    for half in baselines_half:
        out[half] = {}
        for r in ranges:
            out[half][r] = {}
            for fov in fov_bins_deg:
                ha = np.deg2rad([-fov, 0.0, fov]).tolist()
                out[half][r][fov] = {}
                # common-mode requirement (range error) at this geometry
                cm_need = None
                for sg in sigma_grid:
                    cfg = SensitivityConfig(
                        ranges=[r], h_angles=ha, v_angles=v,
                        common_sigma=sg, diff_sigma=0.0,
                        baseline_half=half, n_trials=n_trials, seed=seed, **kw)
                    res = run_monte_carlo(cfg)
                    if res["overall_range_p95"] <= targets_cm[0] / 100.0:
                        cm_need = sg
                        break
                out[half][r][fov]["common_sigma"] = cm_need
                for tgt in targets_cm:
                    need = None
                    for sg in sigma_grid:
                        cfg = SensitivityConfig(
                            ranges=[r], h_angles=ha, v_angles=v,
                            common_sigma=0.0, diff_sigma=sg,
                            baseline_half=half, n_trials=n_trials, seed=seed, **kw)
                        res = run_monte_carlo(cfg)
                        if res["overall_p95"] <= tgt / 100.0:
                            need = sg
                            break
                    out[half][r][fov][tgt] = need
    return out


def baseline_sweep(
    halves=(0.03, 0.06, 0.12, 0.24),
    ranges=(1.0, 2.0, 3.0),
    diff_sigma=10e-6,
    **kw,
):
    """Median 95th-percentile error as a function of microphone baseline."""
    h = kw.pop("h_angles", np.deg2rad([-15, 0, 15]).tolist())
    v = kw.pop("v_angles", np.deg2rad([-8, 0, 8]).tolist())
    n_trials = kw.pop("n_trials", 40)
    out = {}
    for half in halves:
        row = {}
        for r in ranges:
            cfg = SensitivityConfig(
                ranges=[r], h_angles=h, v_angles=v,
                diff_sigma=diff_sigma, baseline_half=half, n_trials=n_trials, seed=3, **kw)
            res = run_monte_carlo(cfg)
            row[r] = res["overall_p95"]
        out[half] = row
    return out


def estimator_compare(
    ranges=(1.0, 2.0, 3.0),
    diff_sigma=10e-6,
    **kw,
):
    """Median 95th-percentile error: envelope (integer-sample) vs fractional
    refinement, and the effect of a coarser ADC rate. Achievable delay
    precision may be SUB-SAMPLE, so the sample interval is a raw grid, not a
    hard floor."""
    h = kw.pop("h_angles", np.deg2rad([-15, 0, 15]).tolist())
    v = kw.pop("v_angles", np.deg2rad([-8, 0, 8]).tolist())
    n_trials = kw.pop("n_trials", 40)
    sr_fine = kw.pop("sr", SAMPLE_RATE)
    sr_coarse = max(40000, sr_fine // 4)
    out = {}
    for label, cfg in [
        ("envelope_fine_sr", SensitivityConfig(ranges=list(ranges), h_angles=h, v_angles=v,
             diff_sigma=diff_sigma, refine=False, sr=sr_fine, n_trials=n_trials, seed=5, **kw)),
        ("envelope_coarse_sr", SensitivityConfig(ranges=list(ranges), h_angles=h, v_angles=v,
             diff_sigma=diff_sigma, refine=False, sr=sr_coarse, n_trials=n_trials, seed=5, **kw)),
        ("refined_fine_sr", SensitivityConfig(ranges=list(ranges), h_angles=h, v_angles=v,
             diff_sigma=diff_sigma, refine=True, sr=sr_fine, n_trials=n_trials, seed=5, **kw)),
    ]:
        res = run_monte_carlo(cfg)
        out[label] = res["overall_p95"]
    return out

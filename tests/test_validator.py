"""Test that the second-stage validator closes the coherence check gap.

Derives thresholds from a clean/calibrated scene, then evaluates on the
hardest near-coherent destructive multipath case.  The validator must
bring conditional false-valid below 5 % (it may achieve that by rejecting
more trials — valid_fraction may fall, that is acceptable).
"""

import numpy as np
import argus.sensitivity as S
from argus.detect import calibrate_validator_thresholds
from argus.acoustic_scene import Reflector, diamond_mics
from argus.schedule import build_schedule
from argus.detect import coherent_onset, median_tof
from argus.synthesize import synthesize

SPEED = 343.0
SR = 250_000
FREQS = [40_000, 42_000, 44_000, 46_000, 48_000]
TONE = 0.004


def test_validator_thresholds_derived_from_clean():
    """calibrate_validator_thresholds runs on a clean coherent scene and
    returns positive finite thresholds."""
    mics = diamond_mics()
    P = np.array([0.3, -0.2, 2.0])
    sched = build_schedule(FREQS, TONE, SR)
    wf = synthesize(sched, [Reflector(P)], mics, SPEED, coherent=True)
    coarse = median_tof(coherent_onset(wf, sched), axis=1)
    th = calibrate_validator_thresholds(
        wf, sched.segments, SR, coarse, reference_mic=0,
        mics=mics, speed=SPEED, n_trials=10)
    for k in ("cross_band", "loto", "envelope", "geom_spread"):
        assert k in th, f"missing threshold {k}"
        assert 0 < th[k] < 1, f"{k}={th[k]} out of range"


def test_validator_closes_coherence_gap():
    """Derive thresholds from clean data; apply to the hardest near-coherent
    multipath case; cond_false_valid must drop below 5%."""
    mics = diamond_mics()
    P = np.array([0.3, -0.2, 2.0])
    sched = build_schedule(FREQS, TONE, SR)
    n_trials = 30

    wf_clean = synthesize(sched, [Reflector(P)], mics, SPEED, coherent=True)
    coarse_clean = median_tof(coherent_onset(wf_clean, sched), axis=1)
    th = calibrate_validator_thresholds(
        wf_clean, sched.segments, SR, coarse_clean, reference_mic=0,
        mics=mics, speed=SPEED, n_trials=30)
    print(f"  Thresholds: cross_band={th['cross_band']:.2e}  "
          f"loto={th['loto']:.2e}  envelope={th['envelope']:.2e}  "
          f"geom_spread={th['geom_spread']:.3f}")

    mp = S.MultipathConfig(amp_ratio=0.5, delay_extra=20e-6,
                            delay_per_mic_sigma=30e-6)
    cfg = S.SensitivityConfig(ranges=[2.0],
                               h_angles=np.deg2rad([-10, 0, 10]).tolist(),
                               v_angles=np.deg2rad([-8, 0, 8]).tolist(),
                               n_trials=n_trials, seed=0, waveform_noise=0.0)

    res_no_val = S.run_waveform_chain(cfg, mp_config=mp)
    print(f"  Without validator: vf={res_no_val['valid_fraction']:.3f}  "
          f"cfv={res_no_val['cond_false_valid_fraction']:.4f}  "
          f"ph_p95={res_no_val['phase']['p95']:.4f}")

    res_val = S.run_waveform_chain(cfg, mp_config=mp,
                                    validator_thresholds=th)
    print(f"  With validator:    vf={res_val['valid_fraction']:.3f}  "
          f"cfv={res_val['cond_false_valid_fraction']:.4f}  "
          f"ph_p95={res_val['phase']['p95']:.4f}  "
          f"fb_p95={res_val['fallback']['p95']:.4f}")

    assert res_val["cond_false_valid_fraction"] < 0.05, (
        f"cond_false_valid {res_val['cond_false_valid_fraction']:.4f} "
        f"still >= 0.05 after validator"
    )
    assert res_val["fallback"]["p95"] <= res_no_val["env"]["p95"] + 0.02

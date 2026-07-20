"""CLI: validate an echos-argus NPZ trajectory file against the v1 contract.

Usage:

    python -m argus.validate_echos_trajectory path/to/file.npz

Exit code 0 on pass, 1 on any contract violation.
"""

import sys
import json
import numpy as np
from .motion import SampledTrajectory, _REQUIRED_KEYS_STRICT


def validate(path: str) -> dict:
    data = dict(np.load(path))
    traj = SampledTrajectory(data, strict_contract=True)
    report = traj.contract_report()
    return report


def _pass_fail(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m argus.validate_echos_trajectory <file.npz>")
        sys.exit(1)

    path = sys.argv[1]
    try:
        report = validate(path)
    except Exception as e:
        print(f"VALIDATION FAILED: {e}")
        sys.exit(1)

    print(f"File:                     {path}")
    print(f"Schema:                   {report.get('schema', '?')}  v{report.get('version', '?')}")
    print(f"Samples:                  {report.get('n_samples', '?')}")
    print(f"Duration:                 {report.get('duration_s', '?'):.3f} s")
    print(f"dt median / max:          {report.get('dt_median_s', '?'):.6f} / {report.get('dt_max_s', '?'):.6f} s")
    print(f"Position bounds:          {report.get('pos_bounds_m', '?')}")
    print(f"Max speed:                {report.get('max_speed_mps', '?'):.3f} m/s")
    print(f"Max body omega:           {report.get('max_omega_body_rps', '?'):.3f} rad/s")
    print(f"Max body pitch:           {report.get('max_body_pitch_deg', '?'):.2f} deg")
    print(f"Max gimbal pitch:         {report.get('max_gimbal_pitch_deg', '?'):.2f} deg")
    print(f"Body quat norm range:     {report.get('q_body_norm_range', '?')}")
    print(f"Beam axis norm range:     {report.get('beam_axis_norm_range', '?')}")
    print()

    _TOLERANCES = {
        "xval_omega_transform": ("rad/s", 1e-8),
        "xval_gimbal_quat_composition_deg": ("deg", 1.0),
        "xval_beam_axis_deg": ("deg", 0.5),
        "xval_emitter_pos_mm": ("mm", 1.0),
        "xval_mic_pos_mm": ("mm", 1.0),
    }

    checks = [
        ("Transform: omega world from body", "xval_omega_transform"),
        ("Transform: gimbal quat composition", "xval_gimbal_quat_composition_deg"),
        ("Transform: beam axis", "xval_beam_axis_deg"),
        ("Transform: emitter position", "xval_emitter_pos_mm"),
        ("Transform: microphone positions", "xval_mic_pos_mm"),
    ]

    all_pass = True
    for name, key in checks:
        val = report.get(key, None)
        tol = _TOLERANCES.get(key, (None, 1e9))
        unit, limit = tol
        if val is not None:
            ok = val <= limit
            all_pass = all_pass and ok
            print(f"  {_pass_fail(ok):>4}  {name}: {val:.4f} {unit}  (limit {limit:.4f} {unit})")
        else:
            print(f"  SKIP  {name}: arrays not available for comparison")

    print()
    if all_pass:
        print("RESULT: PASS")
        sys.exit(0)
    else:
        print("RESULT: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()

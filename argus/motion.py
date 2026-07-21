"""Drone trajectory definitions for motion-acoustic simulation.

Each trajectory provides the time-varying emitter and microphone positions in
the world frame, plus the gimbal-stabilized beam axis.

The reflector is stationary; only the drone moves.
"""

import numpy as np


def _rotation_matrix(axis, angle):
    """Rodrigues' rotation formula."""
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if axis.shape != (3,) or not np.isfinite(norm) or norm == 0:
        raise ValueError("rotation axis must be a finite, non-zero 3-vector")
    if not np.isfinite(angle):
        raise ValueError("rotation angle must be finite")
    axis = axis / norm
    c, s = np.cos(angle), np.sin(angle)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + s * K + (1 - c) * (K @ K)


class Trajectory:
    """Base class: position and orientation as functions of absolute time.

    ``mic_local`` are the microphone positions in the drone body frame.
    The beam axis is the direction the gimbal keeps fixed in world space.
    Subclasses must implement ``emitter_pos`` and ``body_rot``.
    """

    def __init__(self, mic_local, beam_axis_world=None):
        self.mic_local = np.asarray(mic_local, dtype=float)
        if (self.mic_local.ndim != 2 or self.mic_local.shape[1] != 3 or
                len(self.mic_local) == 0 or np.any(~np.isfinite(self.mic_local))):
            raise ValueError("mic_local must have shape (n, 3) and be finite")
        axis = (np.array([0.0, 0.0, 1.0]) if beam_axis_world is None
                else np.asarray(beam_axis_world, dtype=float))
        norm = np.linalg.norm(axis)
        if axis.shape != (3,) or not np.isfinite(norm) or norm == 0:
            raise ValueError("beam_axis_world must be a finite, non-zero 3-vector")
        self.beam_axis_world = axis / norm

    def emitter_pos(self, t: float) -> np.ndarray:
        raise NotImplementedError

    def body_rot(self, t: float) -> np.ndarray:
        """Rotation matrix from body to world at time t."""
        raise NotImplementedError

    def mic_pos(self, t: float, mic_idx: int) -> np.ndarray:
        """World-frame position of microphone i at time t."""
        return self.emitter_pos(t) + self.body_rot(t) @ self.mic_local[mic_idx]

    def beam_axis(self, t: float) -> np.ndarray:
        """Gimbal-stabilized beam axis in world frame (constant by default)."""
        return self.beam_axis_world.copy()


class ConstantVelocity(Trajectory):
    """Drone moving at constant linear velocity, optionally rotating at constant
    angular rates.

    ``p0``: world position at t=0.
    ``v``: linear velocity vector (m/s).
    ``omega``: angular velocity vector (rad/s) in body frame.
    ``body_orientation0``: initial body-to-world rotation matrix.
    """
    def __init__(self, mic_local, p0, v, omega=np.zeros(3),
                 body_orientation0=None, beam_axis_world=None):
        super().__init__(mic_local, beam_axis_world)
        self.p0 = np.asarray(p0, dtype=float)
        self.v = np.asarray(v, dtype=float)
        self.omega = np.asarray(omega, dtype=float)
        if any(x.shape != (3,) or np.any(~np.isfinite(x))
               for x in (self.p0, self.v, self.omega)):
            raise ValueError("p0, v, and omega must be finite 3-vectors")
        self.body_orientation0 = (np.eye(3) if body_orientation0 is None
                                  else np.asarray(body_orientation0, dtype=float))
        if (self.body_orientation0.shape != (3, 3) or
                np.any(~np.isfinite(self.body_orientation0)) or
                not np.allclose(self.body_orientation0.T @ self.body_orientation0,
                                np.eye(3), atol=1e-7) or
                not np.isclose(np.linalg.det(self.body_orientation0), 1.0, atol=1e-7)):
            raise ValueError("body_orientation0 must be a proper rotation matrix")

    def emitter_pos(self, t: float) -> np.ndarray:
        return self.p0 + self.v * t

    def body_rot(self, t: float) -> np.ndarray:
        if np.linalg.norm(self.omega) < 1e-12:
            return self.body_orientation0
        angle = np.linalg.norm(self.omega) * t
        axis = self.omega / np.linalg.norm(self.omega)
        return self.body_orientation0 @ _rotation_matrix(axis, angle)


def _compute_delay(P_ref, t_emit, traj, mic_idx, c=343.0, max_iter=5):
    """Iteratively compute the round-trip delay for a moving platform.

    L = |P_emit(t_emit) - P_ref| + |P_ref - P_mic(t_emit + delay)|
    delay = L / c

    Fixed-point iteration starting from the static delay.
    """
    d = np.linalg.norm(traj.emitter_pos(t_emit) - P_ref) * 2.0 / c
    for _ in range(max_iter):
        t_ret = t_emit + d
        L = (np.linalg.norm(traj.emitter_pos(t_emit) - P_ref) +
             np.linalg.norm(P_ref - traj.mic_pos(t_ret, mic_idx)))
        d_new = L / c
        if abs(d_new - d) < 1e-12:
            break
        d = d_new
    return d


# ---------------------------------------------------------------------------
# ===========================================================================
# echos-argus-trajectory schema version 1  (strict contract)
#
# Canonical file: compressed NumPy NPZ.
#
# --- Required arrays -------------------------------------------------------
#
# t_s                       float64  [N]      simulation time, monotonic
# body_pos_world_m          float64  [N,3]    body-frame origin in world
# body_quat_world_wxyz      float64  [N,4]    body orientation (wxyz)
# body_vel_world_mps        float64  [N,3]    body velocity in world axes
# body_omega_body_rps       float64  [N,3]    angular velocity in body axes
# body_omega_world_rps      float64  [N,3]    angular velocity in world axes
# gimbal_pitch_rad          float64  [N]      gimbal pitch angle
# gimbal_quat_body_wxyz     float64  [N,4]    gimbal orientation rel. body
# gimbal_quat_world_wxyz    float64  [N,4]    gimbal orientation in world
# beam_axis_world           float64  [N,3]    beam direction in world
# emitter_pos_world_m       float64  [N,3]    emitter position in world
# mic_pos_world_m           float64  [N,4,3]  all four mic positions (world)
# metadata_json             scalar UTF-8 JSON string
#
# --- Body frame convention -------------------------------------------------
#   +X forward
#   +Y left
#   +Z up
#
# --- Quaternion convention -------------------------------------------------
#   order:    (w, x, y, z)
#   semantics: active body-to-world rotation
#
# --- Physical offsets (body frame, metres) ---------------------------------
#   emitter:          [0.0825, 0.0, 0.0]
#   front mic:        [0.0785, 0.0, 0.0215]
#   rear mic:         [-0.0785, 0.0, -0.0215]
#   left mic:         [0.0, 0.0785, 0.0]
#   right mic:        [0.0, -0.0785, 0.0]
#
# --- Required metadata keys (inside metadata_json) -------------------------
#   schema, version, physics_dt_s, control_dt_s, world_handedness,
#   world_up_axis, body_forward_axis, body_positive_y_semantics,
#   quaternion_order, quaternion_semantics, position_unit, time_unit,
#   angular_unit, physical_emitter_offset_body_m
#
# ===========================================================================


import json


ECHOS_EMITTER_OFFSET = np.array([0.0825, 0.0, 0.0])
ECHOS_MIC_OFFSETS = np.array([
    [0.0785, 0.0, 0.0215],
    [-0.0785, 0.0, -0.0215],
    [0.0, 0.0785, 0.0],
    [0.0, -0.0785, 0.0],
])


def _quat_multiply(q1, q2):
    """Hamilton product of two quaternions (w,x,y,z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _normalize_quaternion(q):
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if q.shape != (4,) or not np.isfinite(norm) or norm == 0:
        raise ValueError("quaternion must be a finite, non-zero 4-vector")
    return q / norm


def _quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_to_rotmat(q):
    """Convert (w,x,y,z) quaternion to 3×3 rotation matrix."""
    q = _normalize_quaternion(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z,   2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [2*x*y + 2*w*z,       1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y,       2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y],
    ])


def _slerp(q0, q1, t):
    """Spherical linear interpolation between two unit quaternions (w,x,y,z).

    Returns the quaternion at fraction *t* ∈ [0, 1] along the shortest path.
    """
    if not np.isfinite(t):
        raise ValueError("SLERP fraction must be finite")
    q0 = _normalize_quaternion(q0)
    q1 = _normalize_quaternion(q1)
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return _normalize_quaternion(q0 + t * (q1 - q0))
    theta = np.arccos(np.clip(dot, -1, 1))
    sin_th = np.sin(theta)
    a = np.sin((1 - t) * theta) / sin_th
    b = np.sin(t * theta) / sin_th
    return _normalize_quaternion(a * q0 + b * q1)


def _linear_interp(t, t_grid, y_grid):
    """Vectorised linear interpolation for 1D or 2D arrays along the first axis."""
    idx = np.searchsorted(t_grid, t) - 1
    idx = np.clip(idx, 0, len(t_grid) - 2)
    f = (t - t_grid[idx]) / max(t_grid[idx + 1] - t_grid[idx], 1e-15)
    return y_grid[idx] + f * (y_grid[idx + 1] - y_grid[idx])


_REQUIRED_KEYS_STRICT = [
    "t_s", "body_pos_world_m", "body_quat_world_wxyz",
    "body_vel_world_mps", "body_omega_body_rps", "body_omega_world_rps",
    "gimbal_pitch_rad", "gimbal_quat_body_wxyz", "gimbal_quat_world_wxyz",
    "beam_axis_world", "emitter_pos_world_m", "mic_pos_world_m",
    "metadata_json",
]

_REQUIRED_METADATA = {
    "schema": "echos-argus-trajectory",
    "version": 1,
    "world_handedness": "right",
    "world_up_axis": "+Z",
    "body_forward_axis": "+X",
    "body_positive_y_semantics": "left",
    "quaternion_order": "wxyz",
    "quaternion_semantics": "active_body_to_world",
    "position_unit": "m",
    "time_unit": "s",
    "angular_unit": "rad",
}

# Validation tolerances
_TOL_QUAT_NORM = 0.01
_TOL_OMEGA_TRANSFORM = 1e-8
_TOL_GIMBAL_QUAT_DEG = 1.0       # degrees
_TOL_BEAM_AXIS_DEG = 0.5
_TOL_POSITION_MM = 1.0
_TOL_AXIS_NORM = 0.01


def _parse_metadata(meta_str):
    """Parse metadata_json, validate required keys/values, return dict."""
    if isinstance(meta_str, bytes):
        meta_str = meta_str.decode("utf-8")
    if isinstance(meta_str, np.ndarray) and meta_str.ndim == 0:
        meta_str = str(meta_str)
    try:
        meta = json.loads(meta_str) if isinstance(meta_str, str) else dict(meta_str)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"metadata_json is not valid JSON/object data: {exc}") from exc
    for k, expected in _REQUIRED_METADATA.items():
        if k not in meta:
            raise ValueError(f"metadata missing required key {k!r}")
        if isinstance(expected, str) and meta[k] != expected:
            raise ValueError(
                f"metadata {k}={meta[k]!r}, expected {expected!r}")
        if isinstance(expected, int) and meta[k] != expected:
            raise ValueError(
                f"metadata {k}={meta[k]}, expected {expected}")
    if "physical_emitter_offset_body_m" not in meta:
        raise ValueError("metadata missing required key 'physical_emitter_offset_body_m'")
    offset = np.asarray(meta["physical_emitter_offset_body_m"], dtype=float)
    if offset.shape != (3,) or not np.allclose(offset, ECHOS_EMITTER_OFFSET, atol=1e-9):
        raise ValueError("metadata physical_emitter_offset_body_m does not match Echos")
    for key in ("physics_dt_s", "control_dt_s"):
        if (not np.isfinite(meta[key]) or meta[key] <= 0):
            raise ValueError(f"metadata {key} must be finite and positive")
    return meta


class SampledTrajectory(Trajectory):
    """Trajectory sampled at discrete timestamps, loaded from an echos-argus
    NPZ file or from a dict of arrays.

    Interpolates body position (linear), orientation (SLERP), and beam axis
    (linear) to arbitrary query times.  Validates the loaded schema against
    the version-1 contract (strict by default).

    Parameters
    ----------
    data : dict or str
        Dict of arrays (from np.load) or path to an NPZ file.
    mic_local : (4, 3) ndarray | None
        Body-frame microphone positions.  If None uses ECHOS_MIC_OFFSETS.
    beam_axis_gimbal : (3,) ndarray | None
        Beam axis in the gimbal frame (default [1,0,0] — +X forward).
    strict_contract : bool
        If True (default), reject missing required fields and metadata.
    """

    def __init__(self, data, mic_local=None, beam_axis_gimbal=None,
                 strict_contract=True):
        if isinstance(data, str):
            data = dict(np.load(data))
        if mic_local is None:
            mic_local = ECHOS_MIC_OFFSETS
        super().__init__(mic_local)
        self._strict = strict_contract

        # --- required field presence (strict only) --------------------------
        if strict_contract:
            for k in _REQUIRED_KEYS_STRICT:
                if k not in data:
                    raise ValueError(
                        f"Strict contract: missing required field {k!r}")

        # --- t_s  -----------------------------------------------------------
        t = np.asarray(data["t_s"], dtype=float)
        if t.ndim != 1:
            raise ValueError(f"t_s must be 1-D, got shape {t.shape}")
        if len(t) < 2:
            raise ValueError(f"Need at least 2 samples, got {len(t)}")
        if np.any(np.diff(t) <= 0):
            raise ValueError("t_s must be strictly increasing")
        if np.any(~np.isfinite(t)):
            raise ValueError("t_s contains non-finite values")
        self._t = t

        # --- shape + finite --------------------------------------------------
        field_shapes = {
            "body_pos_world_m": (len(t), 3),
            "body_quat_world_wxyz": (len(t), 4),
            "body_vel_world_mps": (len(t), 3),
            "body_omega_body_rps": (len(t), 3),
            "body_omega_world_rps": (len(t), 3),
            "gimbal_pitch_rad": (len(t),),
            "gimbal_quat_body_wxyz": (len(t), 4),
            "gimbal_quat_world_wxyz": (len(t), 4),
            "beam_axis_world": (len(t), 3),
            "emitter_pos_world_m": (len(t), 3),
            "mic_pos_world_m": (len(t), 4, 3),
        }
        for k, shape in field_shapes.items():
            if k not in data:
                continue
            arr = np.asarray(data[k], dtype=float)
            if arr.shape != shape:
                raise ValueError(f"{k} must have shape {shape}, got {arr.shape}")
            if np.any(~np.isfinite(arr)):
                raise ValueError(f"{k} contains non-finite values")

        # --- quaternion norm checks -----------------------------------------
        for qkey in ("body_quat_world_wxyz", "gimbal_quat_body_wxyz",
                     "gimbal_quat_world_wxyz"):
            if qkey in data:
                q = np.asarray(data[qkey], dtype=float)
                err = np.max(np.abs(np.linalg.norm(q, axis=1) - 1.0))
                if err > _TOL_QUAT_NORM:
                    raise ValueError(
                        f"{qkey} max norm error {err:.4f} > {_TOL_QUAT_NORM}")

        # --- metadata -------------------------------------------------------
        if "metadata_json" in data:
            self.metadata = _parse_metadata(data["metadata_json"])
        elif strict_contract:
            raise ValueError("metadata_json required in strict mode")
        else:
            self.metadata = {}

        # --- store main arrays (use .get with safe defaults for non-strict) --
        self._pos = np.asarray(data.get("body_pos_world_m", np.zeros((len(t), 3))), dtype=float)
        self._q_body = np.asarray(data.get("body_quat_world_wxyz", np.tile([1, 0, 0, 0], (len(t), 1))), dtype=float)
        self._vel = (np.asarray(data.get("body_vel_world_mps", np.zeros((len(t), 3))), dtype=float)
                     if "body_vel_world_mps" in data else None)
        self._omega_body = (np.asarray(data["body_omega_body_rps"], dtype=float)
                            if "body_omega_body_rps" in data else None)
        self._omega_world = (np.asarray(data["body_omega_world_rps"], dtype=float)
                             if "body_omega_world_rps" in data else None)
        self._gimbal_pitch = (np.asarray(data["gimbal_pitch_rad"], dtype=float)
                              if "gimbal_pitch_rad" in data else None)
        self._q_gimbal_body = np.asarray(data.get("gimbal_quat_body_wxyz", np.tile([1, 0, 0, 0], (len(t), 1))), dtype=float)
        self._q_gimbal_world = (np.asarray(data["gimbal_quat_world_wxyz"], dtype=float)
                                if "gimbal_quat_world_wxyz" in data else None)
        self._beam = (np.asarray(data["beam_axis_world"], dtype=float)
                      if "beam_axis_world" in data else None)
        self._emitter_pos_export = (np.asarray(data["emitter_pos_world_m"], dtype=float)
                                    if "emitter_pos_world_m" in data else None)
        self._mic_pos_export = (np.asarray(data["mic_pos_world_m"], dtype=float)
                                if "mic_pos_world_m" in data else None)

        self._beam_axis_gimbal = (np.array([1.0, 0.0, 0.0])
                                  if beam_axis_gimbal is None
                                  else np.asarray(beam_axis_gimbal, dtype=float))
        beam_gimbal_norm = np.linalg.norm(self._beam_axis_gimbal)
        if (self._beam_axis_gimbal.shape != (3,) or
                not np.isfinite(beam_gimbal_norm) or beam_gimbal_norm == 0):
            raise ValueError("beam_axis_gimbal must be a finite, non-zero 3-vector")
        self._beam_axis_gimbal /= beam_gimbal_norm

        self._q_body = self._q_body / np.linalg.norm(self._q_body, axis=1)[:, None]
        self._q_gimbal_body = self._q_gimbal_body / np.linalg.norm(
            self._q_gimbal_body, axis=1)[:, None]
        if self._q_gimbal_world is not None:
            self._q_gimbal_world /= np.linalg.norm(self._q_gimbal_world, axis=1)[:, None]
        if self._beam is not None:
            beam_norm = np.linalg.norm(self._beam, axis=1)
            if np.any(beam_norm == 0):
                raise ValueError("beam_axis_world contains a zero vector")
            if strict_contract and np.max(np.abs(beam_norm - 1.0)) > _TOL_AXIS_NORM:
                raise ValueError("beam_axis_world is not normalized")
            self._beam = self._beam / beam_norm[:, None]

    # -- Trajectory interface ------------------------------------------------

    def body_origin(self, t: float) -> np.ndarray:
        """World-frame position of the drone body origin."""
        return _linear_interp(t, self._t, self._pos)

    def emitter_pos(self, t: float) -> np.ndarray:
        """World-frame position of the physical emitter.

        Uses the exported ``emitter_pos_world_m`` array when available
        (Echos contract), otherwise derives from body origin + emitter
        offset rotated by body orientation.
        """
        if self._emitter_pos_export is not None:
            return _linear_interp(t, self._t, self._emitter_pos_export)
        R = self.body_rot(t)
        return self.body_origin(t) + R @ ECHOS_EMITTER_OFFSET

    def body_rot(self, t: float) -> np.ndarray:
        idx = np.clip(np.searchsorted(self._t, t) - 1, 0, len(self._t) - 2)
        t0, t1 = self._t[idx], self._t[idx + 1]
        frac = (t - t0) / (t1 - t0) if abs(t1 - t0) > 1e-15 else 0.0
        q = _slerp(self._q_body[idx], self._q_body[idx + 1], frac)
        return _quat_to_rotmat(q)

    def beam_axis(self, t: float) -> np.ndarray:
        if self._beam is None:
            axis = self.body_rot(t) @ self._beam_axis_gimbal
        else:
            axis = _linear_interp(t, self._t, self._beam)
        return axis / np.linalg.norm(axis)

    # -- Transform cross-validation ------------------------------------------

    def validate_transforms(self):
        """Run all independent transform checks.

        Returns dict of {name: max_error_or_None} for every derived quantity
        that can be compared against the exported Echos arrays.
        """
        results = {}
        n = len(self._t)

        # 1. omega world from body
        if self._omega_body is not None and self._omega_world is not None:
            max_om = 0.0
            for i in range(n):
                R = _quat_to_rotmat(self._q_body[i])
                derived = R @ self._omega_body[i]
                err = np.linalg.norm(derived - self._omega_world[i])
                max_om = max(max_om, err)
            results["omega_transform"] = max_om

        # 2. gimbal world quaternion from composition
        if self._q_gimbal_world is not None:
            max_gq = 0.0
            for i in range(n):
                q_world = _quat_multiply(self._q_body[i], self._q_gimbal_body[i])
                # handle q / -q equivalence
                dot = abs(np.dot(q_world, self._q_gimbal_world[i]))
                angle = np.rad2deg(2 * np.arccos(np.clip(dot, -1, 1)))
                max_gq = max(max_gq, angle)
            results["gimbal_quat_composition_deg"] = max_gq

        # 3. beam axis from quaternion
        if self._beam is not None:
            max_ba = 0.0
            for i in range(n):
                R = _quat_to_rotmat(self._q_body[i])
                Rg = _quat_to_rotmat(self._q_gimbal_body[i])
                derived_world = R @ Rg @ self._beam_axis_gimbal
                err = np.rad2deg(np.arccos(np.clip(
                    np.dot(derived_world, self._beam[i]), -1, 1)))
                max_ba = max(max_ba, err)
            results["beam_axis_deg"] = max_ba

        # 4. emitter position from body
        if self._emitter_pos_export is not None:
            max_ep = 0.0
            for i in range(n):
                R = _quat_to_rotmat(self._q_body[i])
                derived = self._pos[i] + R @ ECHOS_EMITTER_OFFSET
                err = np.linalg.norm(derived - self._emitter_pos_export[i]) * 1000
                max_ep = max(max_ep, err)
            results["emitter_pos_mm"] = max_ep

        # 5. mic positions from body
        if self._mic_pos_export is not None:
            max_mp = 0.0
            for i in range(n):
                R = _quat_to_rotmat(self._q_body[i])
                for j in range(4):
                    derived = self._pos[i] + R @ ECHOS_MIC_OFFSETS[j]
                    err = np.linalg.norm(derived - self._mic_pos_export[i, j]) * 1000
                    max_mp = max(max_mp, err)
            results["mic_pos_mm"] = max_mp

        return results

    def contract_report(self):
        """Return a human-readable dict with schema info and all metrics."""
        dur = self._t[-1] - self._t[0]
        dts = np.diff(self._t)
        pos_bounds = [float(self._pos.min()), float(self._pos.max())]
        speeds = np.linalg.norm(self._vel, axis=1) if self._vel is not None else None
        omega_norms = np.linalg.norm(self._omega_body, axis=1) if self._omega_body is not None else None
        q_norms = np.linalg.norm(self._q_body, axis=1)
        beam_norms = (np.linalg.norm(self._beam, axis=1)
                      if self._beam is not None else None)
        gimbal_pitch_max = float(np.max(np.abs(self._gimbal_pitch))) if self._gimbal_pitch is not None else None

        body_pitch = []
        for i in range(len(self._t)):
            R = _quat_to_rotmat(self._q_body[i])
            # pitch = asin(-R[2,0]) for Z-up body frame
            body_pitch.append(np.rad2deg(np.arcsin(np.clip(-R[2, 0], -1, 1))))
        max_body_pitch = max(abs(p) for p in body_pitch)

        transforms = self.validate_transforms()
        return {
            "schema": "echos-argus-trajectory",
            "version": 1,
            "n_samples": len(self._t),
            "duration_s": float(dur),
            "dt_median_s": float(np.median(dts)),
            "dt_max_s": float(np.max(dts)),
            "pos_bounds_m": pos_bounds,
            "max_speed_mps": float(np.max(speeds)) if speeds is not None else None,
            "max_omega_body_rps": float(np.max(omega_norms)) if omega_norms is not None else None,
            "max_body_pitch_deg": max_body_pitch,
            "max_gimbal_pitch_deg": gimbal_pitch_max,
            "q_body_norm_range": [float(np.min(q_norms)), float(np.max(q_norms))],
            "beam_axis_norm_range": ([float(np.min(beam_norms)), float(np.max(beam_norms))]
                                      if beam_norms is not None else None),
            **{f"xval_{k}": v for k, v in transforms.items()},
        }


def save_trajectory_npz(path, t_s, body_pos, body_quat, body_vel,
                         body_omega_body, body_omega_world,
                         gimbal_pitch, gimbal_quat_body, gimbal_quat_world,
                         beam_axis_world, emitter_pos_world, mic_pos_world,
                         metadata_json, strict_contract=False):
    """Write an echos-argus trajectory to a compressed NPZ file.

    See the module-level docstring for the full schema.
    When *strict_contract* is True, the required fields are validated before
    writing; synthetic test fixtures may disable this.
    """
    arrays = {
        "t_s": np.asarray(t_s, dtype=float),
        "body_pos_world_m": np.asarray(body_pos, dtype=float),
        "body_quat_world_wxyz": np.asarray(body_quat, dtype=float),
        "body_vel_world_mps": np.asarray(body_vel, dtype=float),
        "body_omega_body_rps": np.asarray(body_omega_body, dtype=float),
        "body_omega_world_rps": np.asarray(body_omega_world, dtype=float),
        "gimbal_pitch_rad": np.asarray(gimbal_pitch, dtype=float),
        "gimbal_quat_body_wxyz": np.asarray(gimbal_quat_body, dtype=float),
        "gimbal_quat_world_wxyz": np.asarray(gimbal_quat_world, dtype=float),
        "beam_axis_world": np.asarray(beam_axis_world, dtype=float),
        "emitter_pos_world_m": np.asarray(emitter_pos_world, dtype=float),
        "mic_pos_world_m": np.asarray(mic_pos_world, dtype=float),
        "metadata_json": json.dumps(metadata_json) if isinstance(metadata_json, dict) else metadata_json,
    }
    if strict_contract:
        for k in _REQUIRED_KEYS_STRICT:
            if k not in arrays:
                raise ValueError(f"save missing required field {k!r}")
    np.savez_compressed(path, **arrays)


def path_lengths(P_ref, t_emit, traj, c=343.0):
    """Return (n_mic,) array of round-trip path lengths for a moving drone.

    Uses fixed-point iteration to solve the implicit delay equation.
    """
    P_ref = np.asarray(P_ref, dtype=float)
    if P_ref.shape != (3,) or np.any(~np.isfinite(P_ref)):
        raise ValueError("P_ref must be a finite 3-vector")
    if not np.isfinite(t_emit) or not np.isfinite(c) or c <= 0:
        raise ValueError("t_emit must be finite and c must be positive")
    n = traj.mic_local.shape[0]
    Ls = np.zeros(n)
    for i in range(n):
        d = _compute_delay(P_ref, t_emit, traj, i, c)
        t_ret = t_emit + d
        Ls[i] = (np.linalg.norm(traj.emitter_pos(t_emit) - P_ref) +
                 np.linalg.norm(P_ref - traj.mic_pos(t_ret, i)))
    return Ls

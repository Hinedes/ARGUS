"""Synthesize four raw microphone waveforms from a known scene.

Pure signal chain: for each frequency segment, each mic receives a delayed,
attenuated, phase-shifted tone. No Genesis, no drone, no noise yet (noise is
added by callers / tests).
"""

import numpy as np

from .acoustic_scene import ORIGIN, diamond_mics, Reflector
from .motion import Trajectory, path_lengths as motion_path_lengths
from .schedule import Schedule


def _mic_waveform(
    schedule: Schedule,
    mic_pos: np.ndarray,
    reflectors: list[Reflector],
    speed: float = 343.0,
    amplitude: float = 1.0,
    extra: float = 0.0,
    coherent: bool = False,
    trajectory: Trajectory | None = None,
    mic_idx: int = 0,
) -> np.ndarray:
    # buffer must span the last emission plus the longest round-trip delay
    n = int(np.ceil((schedule.duration + extra) * schedule.sample_rate)) + 1
    t = np.arange(n) / schedule.sample_rate
    sig = np.zeros(n)
    for seg in schedule.segments:
        for refl in reflectors:
            P_ref = refl.pos
            if trajectory is not None:
                Ls = motion_path_lengths(P_ref, seg.start, trajectory, speed)
                delay = Ls[mic_idx] / speed
                path_len = Ls[mic_idx]
            else:
                lengths = refl.path_lengths(ORIGIN, np.atleast_2d(mic_pos))
                delay = lengths[0] / speed
                path_len = lengths[0]
            atten = amplitude / (1.0 + path_len)
            # the received burst is the emitted tone, delayed in TIME by `delay`.
            env = (t >= seg.start + delay) & (t < seg.start + delay + seg.duration)
            k = np.nonzero(env)[0]
            win = np.hanning(k.size) if k.size > 1 else np.ones(k.size)
            if coherent:
                # Physically-correct emitted tone: phase-coherent with emission,
                # so a delay shifts the carrier phase by 2*pi*f*delay. This is
                # what sub-sample phase refinement reads. The envelope detector
                # is unaffected by the carrier phase (it keys on the window), so
                # this mode is only used when phase refinement is also run.
                sig[env] += atten * win * np.cos(
                    2.0 * np.pi * seg.freq * (t[env] - seg.start - delay))
            else:
                sig[env] += atten * win * np.cos(2.0 * np.pi * seg.freq * t[env])
    return sig


def synthesize(
    schedule: Schedule,
    reflectors: list[Reflector],
    mics: np.ndarray | None = None,
    speed: float = 343.0,
    amplitude: float = 1.0,
    coherent: bool = False,
    trajectory: Trajectory | None = None,
) -> np.ndarray:
    """Return (4, N) array of raw waveforms, one row per mic.

    The buffer is extended by the worst-case round-trip delay among the
    reflectors so far-off bursts are not truncated.
    If *trajectory* is provided, per-tone path lengths are computed from
    the moving emitter/receiver positions (motion-acoustic coupling).
    """
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError("speed must be finite and positive")
    if not np.isfinite(amplitude):
        raise ValueError("amplitude must be finite")
    if mics is None:
        mics = diamond_mics()
    mics = np.asarray(mics, dtype=float)
    if mics.ndim != 2 or mics.shape[1] != 3 or np.any(~np.isfinite(mics)):
        raise ValueError("mics must have shape (n, 3) and contain finite values")
    if trajectory is not None and trajectory.mic_local.shape[0] != mics.shape[0]:
        raise ValueError("trajectory and microphone arrays must have the same length")
    if reflectors is None:
        raise ValueError("reflectors must be a sequence")
    for refl in reflectors:
        pos = np.asarray(refl.pos, dtype=float)
        if pos.shape != (3,) or np.any(~np.isfinite(pos)):
            raise ValueError("reflector positions must be finite 3-vectors")
    worst = 0.0
    for refl in reflectors:
        if trajectory is None:
            for m in mics:
                L = refl.path_lengths(ORIGIN, np.atleast_2d(m))[0]
                worst = max(worst, L / speed)
        else:
            for seg in schedule.segments:
                Ls = motion_path_lengths(refl.pos, seg.start, trajectory, speed)
                worst = max(worst, float(np.max(Ls)) / speed)
    extra = worst + 0.005
    return np.stack([
        _mic_waveform(schedule, mics[i], reflectors, speed, amplitude, extra,
                      coherent, trajectory, i)
        for i in range(mics.shape[0])
    ])

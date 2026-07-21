"""Method C transmitter schedule.

Emits a sequence of frequency-agile tone segments. The five frequencies,
tone duration and range window are experimental variables, not laws, so they
are passed in rather than baked in.

Method C pipelining: the emitter sweeps f1..f5 back to back with minimal gap,
so a single emission produces five independent arrival estimates that share one
emission timestamp.
"""

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass
class Segment:
    freq: float
    start: float
    duration: float


@dataclass
class Schedule:
    segments: list[Segment]
    sample_rate: float
    guard: float = 0.0

    def __post_init__(self):
        if not self.segments:
            raise ValueError("schedule must contain at least one segment")
        if not math.isfinite(self.sample_rate) or self.sample_rate <= 0:
            raise ValueError("sample_rate must be finite and positive")
        if not math.isfinite(self.guard) or self.guard < 0:
            raise ValueError("guard must be finite and non-negative")
        for seg in self.segments:
            if (not all(math.isfinite(float(v)) for v in
                        (seg.freq, seg.start, seg.duration)) or
                    seg.freq <= 0 or seg.start < 0 or seg.duration <= 0):
                raise ValueError("segments need finite positive freq/duration and non-negative start")
        ordered = sorted(self.segments, key=lambda s: s.start)
        if any(a.start + a.duration > b.start for a, b in zip(ordered, ordered[1:])):
            raise ValueError("schedule segments may not overlap")

    @property
    def duration(self) -> float:
        return max(s.start + s.duration for s in self.segments)

    def emission_times(self) -> list[float]:
        """Emission start time of each segment (Method C shares one clock)."""
        return [s.start for s in self.segments]


def build_schedule(
    freqs: Sequence[float],
    tone_duration: float,
    sample_rate: float,
    guard: float = 0.0,
) -> Schedule:
    if not freqs:
        raise ValueError("freqs must not be empty")
    if any(not math.isfinite(float(f)) or float(f) <= 0 for f in freqs):
        raise ValueError("freqs must be finite and positive")
    if not math.isfinite(tone_duration) or tone_duration <= 0:
        raise ValueError("tone_duration must be finite and positive")
    if not math.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("sample_rate must be finite and positive")
    if not math.isfinite(guard) or guard < 0:
        raise ValueError("guard must be finite and non-negative")
    segs = []
    t = 0.0
    for f in freqs:
        segs.append(Segment(freq=f, start=t, duration=tone_duration))
        t += tone_duration + guard
    return Schedule(segments=segs, sample_rate=sample_rate, guard=guard)

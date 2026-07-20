"""Method C transmitter schedule.

Emits a sequence of frequency-agile tone segments. The five frequencies,
tone duration and range window are experimental variables, not laws, so they
are passed in rather than baked in.

Method C pipelining: the emitter sweeps f1..f5 back to back with minimal gap,
so a single emission produces five independent arrival estimates that share one
emission timestamp.
"""

from dataclasses import dataclass
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

    @property
    def duration(self) -> float:
        return sum(s.duration for s in self.segments) + self.guard * (len(self.segments) - 1)

    def emission_times(self) -> list[float]:
        """Emission start time of each segment (Method C shares one clock)."""
        return [s.start for s in self.segments]


def build_schedule(
    freqs: Sequence[float],
    tone_duration: float,
    sample_rate: float,
    guard: float = 0.0,
) -> Schedule:
    segs = []
    t = 0.0
    for f in freqs:
        segs.append(Segment(freq=f, start=t, duration=tone_duration))
        t += tone_duration + guard
    return Schedule(segments=segs, sample_rate=sample_rate, guard=guard)

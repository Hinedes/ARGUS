"""Known acoustic scene: emitter, tilted four-mic diamond, reflectors, propagation.

The geometry is what we freeze. The frequencies, spacing and timing stay
experimental. The tilted diamond is the original four-microphone arrangement:
four mics on a square, with the square axis tilted relative to the emitter
boresight, so that signed vertical direction is recoverable from TDOA.
"""

from dataclasses import dataclass
import numpy as np

# Emitter / gimbal origin.
ORIGIN = np.zeros(3)

# Tilt of the diamond plane about the x-axis (radians). The boresight is +z.
# Tilting the mic square off the xy-plane is what gives the array signed
# vertical sensitivity instead of a left/right-only ambiguity.
DIAMOND_TILT = np.deg2rad(12.0)

# Half-spacing of the square (meters). Mic layout before tilt:
#   m0 (+x,+y)  m1 (-x,+y)  m2 (-x,-y)  m3 (+x,-y)
DIAMOND_HALF = 0.06


def diamond_mics(tilt: float = DIAMOND_TILT, half: float = DIAMOND_HALF) -> np.ndarray:
    """Return 4x3 mic positions for the canonical tilted diamond.

    This is the simulation geometry used for acoustic validation.  The Echos
    physical array uses separate offsets defined in ``motion.ECHOS_MIC_OFFSETS``.
    """
    flat = np.array([
        [ half,  half, 0.0],
        [-half,  half, 0.0],
        [-half, -half, 0.0],
        [ half, -half, 0.0],
    ])
    c, s = np.cos(tilt), np.sin(tilt)
    rot = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return flat @ rot.T


@dataclass
class Reflector:
    pos: np.ndarray  # 3D world point

    def path_lengths(self, origin: np.ndarray, mics: np.ndarray) -> np.ndarray:
        """Round-trip path length origin->reflector->each mic (one out, one back)."""
        to_r = np.linalg.norm(self.pos - origin)
        r_to_mic = np.linalg.norm(mics - self.pos, axis=1)
        return to_r + r_to_mic




"""Vibration sensors for bearings, gears and rotating assemblies.

Physics
-------
Acceleration is the second derivative of displacement:

    a(t) = d^2 x / dt^2

For sinusoidal shaft vibration:

    x(t)    = A sin(2*pi*f*t + phi)
    a(t)    = -(2*pi*f)^2 A sin(2*pi*f*t + phi)

Rotating frequency of a shaft at N RPM:

    f_r = N / 60

Gear-mesh frequency (pinion with Z_p teeth):

    f_GMF = N_p * Z_p / 60

Characteristic bearing defect frequencies:

    BPFO = (n/2) f_r [1 - (d/D) cos(theta)]
    BPFI = (n/2) f_r [1 + (d/D) cos(theta)]

Time-domain health indicators:

    RMS      = sqrt((1/N) sum a_i^2)
    Kurtosis = [(1/N) sum (a_i - a_bar)^4] / [(1/N) sum (a_i - a_bar)^2]^2
"""

from __future__ import annotations

import math

import numpy as np

from ..config import TankConfig


def characteristic_frequencies(cfg: TankConfig, rpm: float) -> dict[str, float]:
    """Return f_r, f_GMF, BPFO and BPFI in Hz for a given shaft RPM."""
    f_r = rpm / 60.0
    f_gmf = rpm * cfg.drive_pinion_teeth / 60.0
    cos_theta = math.cos(math.radians(cfg.contact_angle))
    bpfo = (cfg.n_bearings / 2.0) * f_r * (1.0 - (cfg.ball_d / cfg.bearing_pitch_d) * cos_theta)
    bpfi = (cfg.n_bearings / 2.0) * f_r * (1.0 + (cfg.ball_d / cfg.bearing_pitch_d) * cos_theta)
    return {"f_r": f_r, "f_gmf": f_gmf, "BPFO": bpfo, "BPFI": bpfi}


def rms(signal: np.ndarray) -> float:
    """Root-mean-square acceleration amplitude."""
    return float(np.sqrt(np.mean(signal**2)))


def kurtosis(signal: np.ndarray) -> float:
    """Normalised fourth-moment Pearson kurtosis of a signal (equals 3.0 for Gaussian).
 
     High kurtosis indicates impulsive events associated with bearing or
     gear damage.
     """
    mu = np.mean(signal)
    sigma2 = np.mean((signal - mu) ** 2)
    if sigma2 <= 0.0:
        return 0.0
    return float(np.mean((signal - mu) ** 4) / sigma2**2)


class VibrationSensor:
    """Simulates an accelerometer on the engine/drivetrain casing.

    The base signal is a combination of the fundamental shaft tone, a
    gear-mesh tone and (optionally) an impulsive bearing-defect train.
    Fault severity scales the defect amplitudes and impulse energy.
    """

    def __init__(self, cfg: TankConfig, rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(cfg.noise_seed)

    def time_series(self, rpm: float, fault_severity: float = 0.0,
                    defect_type: str = "none", n_samples: int | None = None) -> np.ndarray:
        """Generate a burst of acceleration samples (m/s^2)."""
        cfg = self.cfg
        n = n_samples or cfg.window_samples
        freq = characteristic_frequencies(cfg, rpm)
        t = np.arange(n) / cfg.sample_rate
        fs = cfg.sample_rate

        signal = cfg.shaft_amp_base * np.sin(2.0 * np.pi * freq["f_r"] * t)
        signal += 0.35 * cfg.shaft_amp_base * np.sin(2.0 * np.pi * 2.0 * freq["f_r"] * t)
        signal += 0.15 * cfg.shaft_amp_base * np.sin(2.0 * np.pi * freq["f_gmf"] * t)

        if defect_type in ("bearing_outer", "bearing_inner", "gear_wear"):
            amp = (1.0 + fault_severity) * cfg.shaft_amp_base
            if defect_type == "gear_wear":
                # Gear-mesh harmonics plus shaft-rate sidebands at f_GMF +/- f_r.
                # Sideband energy around the mesh frequency is the standard
                # discriminator for tooth pitting/spalling, so it has to be
                # present for the diagnostic to have anything to key on.
                for k in range(1, 5):
                    signal += amp * 0.9 / k * np.sin(
                        2.0 * np.pi * freq["f_gmf"] * k * t
                    )
                    for sb in (-1, 1):
                        f_sb = freq["f_gmf"] * k + sb * freq["f_r"]
                        if f_sb > 0:
                            signal += amp * 0.45 / k * np.sin(2.0 * np.pi * f_sb * t)
            else:
                f_def = freq["BPFO"] if defect_type == "bearing_outer" else freq["BPFI"]
                signal += amp * np.sin(2.0 * np.pi * f_def * t)
                period = int(round(fs / max(f_def, 1e-6)))
                impulses = np.zeros(n)
                for start in range(0, n, period):
                    idx = slice(start, min(start + max(2, period // 20), n))
                    decay = np.exp(-np.linspace(0, 3, idx.stop - idx.start))
                    impulses[idx] = amp * 6.0 * fault_severity * decay
                signal += impulses

        # The tone amplitudes above are already accelerations (cfg.shaft_amp_base
        # is m/s^2), so no d^2x/dt^2 conversion is applied here.  Two lines used
        # to sit at this point: a sign flip that evaluated to exactly -1.0 for
        # any realistic shaft speed, and a renormalisation pinning the standard
        # deviation to shaft_amp_base^2.  Together they erased fault severity
        # from the amplitude -- vib_rms moved only 0.365 -> 0.396 across the
        # entire severity range, making the configured warn/crit thresholds
        # (0.75 / 1.2 m/s^2) unreachable.  Amplitude must stay proportional to
        # the defect energy built above.
        signal += self.rng.normal(0.0, cfg.vibration_noise, n)
        return signal

    def features(self, rpm: float, fault_severity: float = 0.0,
                 defect_type: str = "none") -> dict[str, float]:
        """Compute RMS, kurtosis and dominant spectral peak."""
        sig = self.time_series(rpm, fault_severity, defect_type)
        spec = np.abs(np.fft.rfft(sig))
        freqs = np.fft.rfftfreq(len(sig), 1.0 / self.cfg.sample_rate)
        freqs = freqs[1:]
        spec = spec[1:]
        peak_idx = int(np.argmax(spec))
        return {
            "vib_rms": rms(sig),
            "vib_kurtosis": kurtosis(sig),
            "vib_dom_freq": float(freqs[peak_idx]),
            "vib_dom_amp": float(spec[peak_idx]),
            "vib_energy": float(np.sum(sig**2)),
        }
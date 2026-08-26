"""Acoustic and acoustic-emission sensors.

Physics
-------
Acoustic pressure wave:

    p(t) = P0 + A sin(2 pi f t)

Sound-pressure level:

    SPL = 20 log10(p_RMS / p_ref),   p_ref = 20 uPa (air)

Frequency-domain analysis (Fourier transform):

    X(f) = integral x(t) e^(-j 2 pi f t) dt

Frequency analysis reveals characteristic signatures of gears, bearings,
pumps, shafts, valves and structural resonances.

Acoustic emission (high-frequency elastic waves from cracks, friction,
impact, material deformation) obeys the wave equation:

    d^2 u / dt^2 = c^2 nabla^2 u

Useful AE features: amplitude, energy, frequency, duration, event rate.
"""

from __future__ import annotations

import math

import numpy as np

from ..config import TankConfig
from .oil import RATE_TAU_S


class AcousticSensor:
    """Microphone measuring airborne mechanical noise (SPL + spectrum)."""

    def __init__(self, cfg: TankConfig, rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(cfg.noise_seed)

    def pressure(self, rpm: float, n_samples: int | None = None,
                 mech_severity: float = 0.0) -> np.ndarray:
        """Airborne pressure trace.

        The fundamental tracks the engine firing frequency rather than a fixed
        120 Hz tone.  For a 4-stroke V12, f_fire = rpm/60 * cylinders/2, so the
        spectrum moves with shaft speed and ``acoustic_dom_freq`` carries
        information instead of reporting one constant value for every record.
        """
        cfg = self.cfg
        n = n_samples or cfg.window_samples
        t = np.arange(n) / cfg.sample_rate
        base_spl = cfg.acoustic_base_spl + 10 * np.log10(max(rpm / 1000.0, 0.1))
        base_spl += 6.0 * min(mech_severity, 3.0)   # wear radiates extra noise
        p_rms = cfg.P_REF_AIR * 10 ** (base_spl / 20.0)

        f_fire = max(rpm / 60.0 * cfg.cylinders / 2.0, 1.0)
        raw = (np.sin(2 * np.pi * f_fire * t)
               + 0.4 * np.sin(2 * np.pi * 2.0 * f_fire * t)
               + 0.2 * np.sin(2 * np.pi * 3.0 * f_fire * t))
        if mech_severity > 0.0:
            f_gmf = rpm * cfg.drive_pinion_teeth / 60.0
            if 0.0 < f_gmf < cfg.sample_rate / 2.0:
                raw = raw + 0.5 * min(mech_severity, 3.0) * np.sin(2 * np.pi * f_gmf * t)

        # Normalise so the trace actually realises the intended SPL.
        rms_raw = float(np.sqrt(np.mean(raw ** 2))) or 1.0
        p = raw * (p_rms / rms_raw)
        return p + self.rng.normal(0.0, 0.02 * p_rms, n)

    def spl(self, pressure: np.ndarray) -> float:
        p_rms = float(np.sqrt(np.mean(pressure**2)))
        return 20.0 * np.log10(max(p_rms / self.cfg.P_REF_AIR, 1e-12))

    def features(self, rpm: float, mech_severity: float = 0.0) -> dict[str, float]:
        p = self.pressure(rpm, mech_severity=mech_severity)
        spec = np.abs(np.fft.rfft(p))
        freqs = np.fft.rfftfreq(len(p), 1.0 / self.cfg.sample_rate)
        freqs = freqs[1:]
        spec = spec[1:]
        return {
            "spl_db": self.spl(p),
            "acoustic_dom_freq": float(freqs[int(np.argmax(spec))]),
            "acoustic_energy": float(np.sum(p**2)),
        }


class AcousticEmissionSensor:
    """High-frequency AE events associated with crack growth, friction,
    impact and material deformation.

    Event rate and energy rise as structural damage (fatigue cracks)
    progresses.
    """

    def __init__(self, cfg: TankConfig, rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(cfg.noise_seed)
        self._rate_est = float(cfg.ae_event_rate_base)

    def read(self, severity: float, dt: float) -> dict[str, float]:
        cfg = self.cfg
        # True (unobservable) arrival intensity of the Poisson event process.
        rate_true = cfg.ae_event_rate_base * (1.0 + 20.0 * severity)
        events = int(self.rng.poisson(max(rate_true * dt, 0.0)))

        if events > 0:
            # Energy is measured per burst, so it carries transducer noise.
            energy = events * (0.5 + 5.0 * severity)
            energy = max(energy * float(self.rng.lognormal(0.0, cfg.ae_energy_noise)), 0.0)
            # Amplitude is read off the measured burst energy (a hit-detection
            # threshold crossing), not off the severity.  `40 + 40*s` was an
            # exact readback of the injected parameter.
            amp = 20.0 * math.log10(max(energy, 1e-6) / 1e-3)
            amp = max(min(amp + float(self.rng.normal(0.0, cfg.ae_noise)), 100.0), 0.0)
        else:
            energy = 0.0
            amp = 0.0

        # Rate estimated from the hits the AE channel actually registered.
        # Publishing `rate_true` made this a noiseless bijection of severity
        # (s = (rate/2 - 1)/20).
        alpha = min(max(dt, 1e-9) / RATE_TAU_S, 1.0)
        self._rate_est += alpha * (events / max(dt, 1e-9) - self._rate_est)
        rate_measured = max(self._rate_est, 0.0)
        return {
            "ae_event_rate": float(rate_measured),
            "ae_events": float(events),
            "ae_energy": float(energy),
            "ae_amp_dB": float(amp),
            "ae_duration_s": float(events * 0.5e-3),
        }
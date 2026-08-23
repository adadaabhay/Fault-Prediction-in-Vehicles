"""Physics-derived health features and sensor-fusion RUL estimation.

Implements section 15/16 of the physics document:

    x(t) = [a, T, P, Q, tau, h, epsilon, p_acoustic, Np]
    x(t) -> Health State
    x(t) -> P(Fault_i)
    RUL = f(x_{1:t})

The fused health index maps normalised deviations of the physics-based
features onto a 0-100 score; RUL is estimated by fitting the health
index trajectory and extrapolating to a failure threshold.

Scoring rules
-------------
Three properties this module has to keep, each of which it previously broke:

* **Deviation is one-sided.**  Damage-indicative channels only count when they
  move in the damaging direction.  Using ``abs()`` penalised oil pressure above
  its reference exactly as hard as oil pressure below it.
* **No unbounded accumulators.**  ``debris_cumulative`` grows for the entire
  mission, so scoring it made a fault-free run decay steadily toward failure --
  health became a proxy for elapsed time.  Rates are scored; totals are not.
* **RUL is causal and bounded.**  The estimate uses only the trailing window,
  is capped at ``RUL_HORIZON_STEPS``, and reports ``inf`` for a stable
  trajectory rather than the number of steps left in the recording (which is a
  property of the file, not the vehicle).
"""

from __future__ import annotations

import math

import numpy as np

from .physics.vibration import kurtosis, rms

FAILURE_THRESHOLD = 20.0  # health index at which failure is declared

# Longest RUL the extrapolation will assert. Beyond this the trend is too flat
# to carry meaning and the estimate is reported as "no degradation detected".
RUL_HORIZON_STEPS = 20000.0

# (reference, span_to_critical, direction) per channel, in raw record units.
# `direction` is +1 when rising values indicate damage, -1 when falling do.
HEALTH_REFERENCES = {
    "vib_rms":           (0.46, 0.90, +1),
    "vib_kurtosis":      (2.0, 6.0, +1),
    "oil_pressure":      (5.2e5, 3.0e5, -1),
    "oil_temp":          (92.0, 43.0, +1),
    "coolant_temp":      (92.0, 28.0, +1),
    "exhaust_temp":      (560.0, 190.0, +1),
    "debris_rate":       (1.0, 14.0, +1),
    "ae_event_rate":     (2.0, 10.0, +1),
    "ae_energy":         (0.5, 14.5, +1),
    "susp_stress_MPa":   (12.0, 18.0, +1),
    "torsion_shear_MPa": (60.0, 90.0, +1),
    "hyd_pressure":      (2.1e7, 1.2e7, -1),
}

# The load-normalised lambda residual is symmetric about 1.0: both
# over-fuelling (low) and air restriction (high) are faults. Raw lambda is not
# scored -- on a quality-governed diesel it ranges 1.4-5.0 with duty alone, so
# a fixed reference marks idling as a combustion fault.
LAMBDA_REFERENCE = 1.0
LAMBDA_SPAN = 0.30
LAMBDA_CHANNEL = "lambda_residual"


class HealthFeatures:
    """Extracts health features from a window of raw sensor records."""

    def __init__(self, records: list[dict[str, float]] | None = None):
        self.records = records or []

    def add(self, record: dict[str, float]) -> None:
        self.records.append(record)

    def trends(self, column: str, window: int = 100) -> list[float]:
        """Slope of a sensor column over a trailing window (linear fit)."""
        n = len(self.records)
        out: list[float] = []
        for i in range(n):
            start = max(0, i - window + 1)
            seg = self.records[start:i + 1]
            if len(seg) < 3:
                out.append(0.0)
                continue
            x = np.arange(len(seg))
            y = np.array([r[column] for r in seg], dtype=float)
            out.append(float(np.polyfit(x, y, 1)[0]))
        return out

    def fused_health_index(self) -> list[float]:
        """Map normalised multi-sensor deviation onto a 0-100 health score.

        A channel exactly at its critical span scores a deviation of 1.0; the
        worst channel dominates, so one subsystem at critical is a failure
        rather than something averaged away by healthy siblings.
        """
        out: list[float] = []
        for r in self.records:
            devs = []
            for col, (ref, span, direction) in HEALTH_REFERENCES.items():
                if col not in r:
                    continue
                delta = (r[col] - ref) * direction
                devs.append(min(max(delta, 0.0) / span, 3.0))
            if LAMBDA_CHANNEL in r:
                devs.append(min(abs(r[LAMBDA_CHANNEL] - LAMBDA_REFERENCE)
                                / LAMBDA_SPAN, 3.0))
            if not devs:
                out.append(100.0)
                continue
            agg = 0.7 * max(devs) + 0.3 * (sum(devs) / len(devs))
            scale = (100.0 - FAILURE_THRESHOLD) / 100.0   # dev 1.0 -> threshold
            out.append(float(np.clip(100.0 * (1.0 - scale * agg), 0.0, 100.0)))
        return out

    def rul(self, window: int = 500) -> list[float]:
        """Remaining useful life, in steps, from the health-index trajectory.

        Causal: each estimate uses only the trailing ``window``.  Returns
        ``inf`` where the trend is flat or improving -- a stable machine has no
        finite time-to-failure, and substituting the number of samples left in
        the file leaks the recording length into the label.
        """
        health = self.fused_health_index()
        n = len(health)
        out: list[float] = [float("inf")] * n
        for i in range(n):
            start = max(0, i - window + 1)
            seg = health[start:i + 1]
            current = health[i]
            if current <= FAILURE_THRESHOLD:
                out[i] = 0.0
                continue
            if len(seg) < 10:
                continue
            slope = float(np.polyfit(np.arange(len(seg)), seg, 1)[0])
            if slope >= -1e-9:          # flat or recovering
                continue
            remaining = (current - FAILURE_THRESHOLD) / (-slope)
            out[i] = float(min(max(remaining, 0.0), RUL_HORIZON_STEPS))
        return out

    def anomaly_score(self, window: int = 500) -> list[float]:
        """Causal z-score of the health deviation.

        Statistics come from the trailing window only.  Using whole-run mean and
        standard deviation let information from the future leak into every
        earlier sample.
        """
        health = np.array(self.fused_health_index(), dtype=float)
        out: list[float] = []
        for i in range(len(health)):
            start = max(0, i - window + 1)
            seg = health[start:i + 1]
            if len(seg) < 2:
                out.append(0.0)
                continue
            mean = float(np.mean(seg))
            std = float(np.std(seg)) + 1e-9
            out.append(float((mean - health[i]) / std))
        return out


__all__ = ["HealthFeatures", "rms", "kurtosis", "FAILURE_THRESHOLD",
           "RUL_HORIZON_STEPS", "HEALTH_REFERENCES"]

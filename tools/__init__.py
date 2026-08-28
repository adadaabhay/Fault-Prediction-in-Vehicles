"""Developer and ops entry points for the ``phm-vehicle`` package.

Modules:
    * :mod:`tools.generate_sensor_data` -- the ``phm-generate`` CLI for
      producing per-subsystem sensor data CSVs + manifest sidecars.
    * :mod:`tools.check_artifacts` -- the ``phm-check-artifacts`` CI gate
      that verifies the dashboard's published model.json / config.json /
      live_stream.json are mutually consistent.
"""

from __future__ import annotations

__all__ = ["generate_sensor_data", "check_artifacts"]

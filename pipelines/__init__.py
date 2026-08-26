"""Multi-Subsystem Data Ingestion & Preprocessing Pipelines.
Provides leak-free feature extraction for:
- Engine & Turbocharging: Deutz TCD 12.0 V6 (Zenodo 5766940)
- Turret & Actuation Hydraulics: ZeMA Condition Monitoring (UCI 447)
- Auxiliary Power & Air Treatment: MetroPT-3 APU (UCI 791)
- Heavy Vehicle Fleet Anomaly Detection: Scania APS (UCI 421)
"""

from .engine_deutz import load_deutz_nrtc_data
from .hydraulics_zema import load_zema_hydraulic_data
from .apu_metropt import load_metropt_apu_data
from .heavy_scania import load_scania_aps_data

__all__ = [
    "load_deutz_nrtc_data",
    "load_zema_hydraulic_data",
    "load_metropt_apu_data",
    "load_scania_aps_data",
]

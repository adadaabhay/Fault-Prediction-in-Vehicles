"""Unit tests for the 4 real subsystem data ingestion pipelines."""

import unittest
import numpy as np
import pandas as pd

from pipelines.engine_deutz import load_deutz_nrtc_data
from pipelines.hydraulics_zema import load_zema_hydraulic_data
from pipelines.apu_metropt import load_metropt_apu_data
from pipelines.heavy_scania import load_scania_aps_data


class TestSubsystemPipelines(unittest.TestCase):
    def test_deutz_pipeline(self):
        try:
            df = load_deutz_nrtc_data()
        except FileNotFoundError as exc:
            self.skipTest(f"Deutz testbench dataset absent: {exc}")
        self.assertGreater(len(df), 100)
        self.assertIn("combustion_anomaly", df.columns)
        self.assertIn("boost_pressure_ratio", df.columns)
        self.assertFalse(df["boost_pressure_ratio"].isna().any())

    def test_zema_pipeline(self):
        try:
            df = load_zema_hydraulic_data()
        except FileNotFoundError as exc:
            self.skipTest(f"ZeMA dataset absent: {exc}")
        self.assertEqual(len(df), 2205)
        self.assertIn("cooler_fault", df.columns)
        self.assertIn("valve_fault", df.columns)
        self.assertIn("PS1_rms", df.columns)
        self.assertFalse(df["PS1_rms"].isna().any())

    def test_metropt_pipeline(self):
        try:
            df = load_metropt_apu_data(max_rows=1000)
        except FileNotFoundError as exc:
            self.skipTest(f"MetroPT3 dataset absent: {exc}")
        self.assertEqual(len(df), 1000)
        self.assertIn("apu_system_fault", df.columns)
        self.assertIn("pressure_differential_bar", df.columns)

    def test_scania_pipeline(self):
        try:
            df = load_scania_aps_data(max_rows=500)
        except FileNotFoundError as exc:
            self.skipTest(f"Scania APS dataset absent: {exc}")
        self.assertEqual(len(df), 500)
        self.assertIn("aps_failure", df.columns)
        self.assertFalse(df["aps_failure"].isna().any())


if __name__ == "__main__":
    unittest.main()

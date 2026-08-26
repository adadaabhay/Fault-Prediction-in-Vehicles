"""Shared training hyperparameter constants for the ML subsystem.

Centralising these avoids silent divergence between lstm.py (which defines
the model and the loss function) and train.py (which calls them via CLI
flags).  Both files import DEFAULT_W_CLS; the CLI flag default and the
function-signature default are derived from the same source of truth.
"""

from __future__ import annotations

# Default classification-loss weight used by lstm.train() and train.py --w-cls.
# Previously lstm.py defaulted to 0.4 and train.py defaulted to 1.5,
# so running `python -m ml.train` without --w-cls produced gradients that
# did not match the loss reported on screen.
DEFAULT_W_CLS: float = 1.0

# Default regression-loss weight.
DEFAULT_W_REG: float = 2.0

"""Project-wide constants.

These values centralize numeric tolerances, submission formatting rules, and
competition constraints used throughout the codebase.
"""

from __future__ import annotations

# Numerical tolerance used across collision/formatting/validation.
EPS: float = 1e-9

# Submission formatting.
SUBMISSION_DECIMALS: int = 17
SUBMISSION_PREFIX: str = "s"

# Competition constraints.
XY_LIMIT: float = 100.0

"""Safe export helpers for investigation results."""

from __future__ import annotations

import pandas as pd

FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _neutralise_formula(value: object) -> object:
    if isinstance(value, str) and value.startswith(FORMULA_PREFIXES):
        return f"'{value}"
    return value


def dataframe_to_safe_csv(dataframe: pd.DataFrame) -> bytes:
    """Encode a CSV while preventing cells from becoming spreadsheet formulas."""

    safe_dataframe = dataframe.map(_neutralise_formula)
    return safe_dataframe.to_csv(index=False).encode("utf-8")

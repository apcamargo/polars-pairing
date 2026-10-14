"""Shared constants and helpers for the pairing test suite."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from polars_pairing import Method as PairingMethod

if TYPE_CHECKING:
    from polars._typing import PolarsDataType

# Expected pair value for each method on the shared inputs
# (n1, n2): (2, 6), (2, 7), (3, 8), (4, 9), (1_501, 10_740), (None, 11).
# Null inputs pair to null.
PAIRED: dict[PairingMethod, list[int | None]] = {
    "hagen": [40, 54, 70, 90, 115_350_602, None],
    "szudzik": [38, 51, 67, 85, 115_349_101, None],
    "cantor": [42, 52, 74, 100, 74_937_901, None],
    "peter": [41, 54, 71, 90, 115_350_603, None],
    "rosenberg_strong": [38, 51, 67, 85, 115_349_101, None],
    "thotakura": [34, 42, 54, 66, 57_698_282, None],
}

ALL_METHODS: list[PairingMethod] = list(PAIRED)

# (method, left, right, pair) near the top of the u64 range. Szudzik and
# Rosenberg-Strong coincide whenever left < right, so left > right is what pins
# them apart. Thotakura maps its pair to exactly u64::MAX, the full range.
LARGE_PAIRS: list[tuple[PairingMethod, int, int, int]] = [
    ("hagen", 2**32 - 1, 0, 18_446_744_065_119_617_025),
    ("szudzik", 2**32 - 1, 0, 18_446_744_069_414_584_320),
    ("cantor", 2**32 - 1, 0, 9_223_372_034_707_292_160),
    ("peter", 2**32 - 1, 0, 18_446_744_065_119_617_025),
    ("rosenberg_strong", 2**32 - 1, 0, 18_446_744_073_709_551_615),
    ("thotakura", 3_037_000_500, 145_474_192, 18_446_744_073_709_551_615),
]

# Integer dtypes the plugin accepts: every dtype the is_integer() guard in
# src/expressions.rs accepts must import cleanly over FFI.
INTEGER_DTYPES: list[PolarsDataType] = [
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.Int128,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.UInt128,
]

# Signed subset for negative-input tests; unsigned dtypes reject negatives at
# Series construction.
SIGNED_INTEGER_DTYPES: list[PolarsDataType] = [
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.Int128,
]


def u64_frame(left: int, right: int) -> pl.DataFrame:
    """Single-row frame with two UInt64 columns, for boundary-value checks."""
    return pl.DataFrame(
        [
            pl.Series("a", [left], dtype=pl.UInt64),
            pl.Series("b", [right], dtype=pl.UInt64),
        ]
    )

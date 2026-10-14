from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl
import pytest
from helpers import (
    ALL_METHODS,
    INTEGER_DTYPES,
    LARGE_PAIRS,
    PAIRED,
    SIGNED_INTEGER_DTYPES,
    PairingMethod,
    u64_frame,
)
from polars.testing import assert_frame_equal

import polars_pairing as plp

if TYPE_CHECKING:
    from polars._typing import PolarsDataType


def _expected_unpair_frame() -> pl.DataFrame:
    # Unpair of the shared PAIRED inputs from helpers: PAIRED[m][i] -> (Left[i], Right[i]),
    # with a None row for the null pair input.
    return pl.DataFrame(
        {
            "Left": [2, 2, 3, 4, 1_501, None],
            "Right": [6, 7, 8, 9, 10_740, None],
        },
        schema={"Left": pl.UInt64, "Right": pl.UInt64},
    )


def _unpair(df: pl.DataFrame, method: str | None = None) -> pl.DataFrame:
    kwargs: dict[str, Any] = {} if method is None else {"method": method}
    return df.select(plp.col("pair").pairing.unpair(**kwargs).alias("unpair")).unnest(
        "unpair"
    )


@pytest.mark.parametrize(
    ("method", "paired"),
    [*PAIRED.items(), (None, PAIRED["hagen"])],
    ids=[*ALL_METHODS, "default"],
)
def test_unpair_exact_values(method: str | None, paired: list[int | None]) -> None:
    result = _unpair(pl.DataFrame({"pair": paired}), method)
    assert_frame_equal(result, _expected_unpair_frame())


def test_unpair_method_case_insensitive() -> None:
    result = _unpair(pl.DataFrame({"pair": PAIRED["cantor"]}), "CanTor")
    assert_frame_equal(result, _expected_unpair_frame())


@pytest.mark.parametrize("frame", ["populated", "empty"])
def test_unpair_invalid_method(frame: str) -> None:
    # The method is resolved up front, so it is rejected even with no rows.
    df = pl.DataFrame({"pair": [42]})
    target = df if frame == "populated" else df.clear()
    with pytest.raises(pl.exceptions.PolarsError, match="Invalid unpairing method"):
        target.select(
            plp.col("pair").pairing.unpair(method="invalid")  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("dtype", SIGNED_INTEGER_DTYPES)
def test_unpair_negative_input_raises(dtype: PolarsDataType) -> None:
    df = pl.DataFrame({"pair": [-4]}, schema={"pair": dtype})
    with pytest.raises(
        pl.exceptions.ComputeError,
        match=r"unpairing input out of range for method 'hagen' with input \(-4\)",
    ):
        _unpair(df)


@pytest.mark.parametrize("dtype", [pl.Int128, pl.UInt128])
def test_unpair_input_beyond_uint64_raises(dtype: PolarsDataType) -> None:
    df = pl.DataFrame({"pair": [2**64]}, schema={"pair": dtype})
    with pytest.raises(
        pl.exceptions.ComputeError, match="unpairing input out of range"
    ):
        _unpair(df)


@pytest.mark.parametrize("dtype", [pl.Int128, pl.UInt128])
def test_unpair_uint64_max_input_is_accepted(dtype: PolarsDataType) -> None:
    # The largest value that still fits the UInt64 encoding must not raise.
    df = pl.DataFrame({"pair": [2**64 - 1]}, schema={"pair": dtype})
    result = _unpair(df, "thotakura")
    assert result.item(0, "Left") == 3_037_000_500
    assert result.item(0, "Right") == 145_474_192


def test_unpair_out_of_range_reports_first_bad_row_not_null_row() -> None:
    df = pl.DataFrame({"pair": [None, 70, -9, -10]})
    with pytest.raises(pl.exceptions.ComputeError, match=r"input \(-9\)"):
        _unpair(df)


@pytest.mark.parametrize("dtype", INTEGER_DTYPES)
def test_unpair_accepts_integer_dtypes(dtype: PolarsDataType) -> None:
    # hagen(3, 8) == 70 (pinned by test_pair_exact_values) fits every dtype.
    df = pl.DataFrame({"pair": [PAIRED["hagen"][2]]}, schema={"pair": dtype})
    result = _unpair(df)
    assert result.item(0, "Left") == 3
    assert result.item(0, "Right") == 8


def test_unpair_null_dtype_gives_all_null() -> None:
    df = pl.DataFrame({"pair": [None, None]}, schema={"pair": pl.Null})
    expected = pl.DataFrame(
        {"Left": [None, None], "Right": [None, None]},
        schema={"Left": pl.UInt64, "Right": pl.UInt64},
    )
    assert_frame_equal(_unpair(df), expected)


@pytest.mark.parametrize(
    "paired",
    [
        pl.Series("pair", [1.5]),
        pl.Series("pair", [42], dtype=pl.Decimal(10, 0)),
    ],
    ids=["float", "decimal"],
)
def test_unpair_rejects_non_integer(paired: pl.Series) -> None:
    df = pl.DataFrame([paired])
    with pytest.raises(pl.exceptions.PolarsError, match="only be used on integers"):
        df.select(plp.col("pair").pairing.unpair())


@pytest.mark.parametrize(("method", "left", "right", "paired"), LARGE_PAIRS)
def test_unpair_large_values_exact(
    method: PairingMethod, left: int, right: int, paired: int
) -> None:
    df = pl.DataFrame({"pair": [paired]}, schema={"pair": pl.UInt64})
    result = _unpair(df, method)
    assert result.item(0, "Left") == left
    assert result.item(0, "Right") == right


def test_unpair_empty_frame_keeps_schema() -> None:
    df = pl.DataFrame({"pair": []}, schema={"pair": pl.UInt64})
    result = _unpair(df)
    assert result.schema == pl.Schema({"Left": pl.UInt64, "Right": pl.UInt64})
    assert result.height == 0


@pytest.mark.parametrize(
    ("n1", "n2"),
    [(0, 0), (1, 1), (42, 7), (1_000, 2_000), (2**31, 1), (10**9, 10**9)],
)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_pair_unpair_round_trip(method: PairingMethod, n1: int, n2: int) -> None:
    paired = u64_frame(n1, n2).select(
        plp.col("a").pairing.pair(plp.col("b"), method=method).alias("pair")
    )
    result = _unpair(paired, method)
    assert result.item(0, "Left") == n1
    assert result.item(0, "Right") == n2

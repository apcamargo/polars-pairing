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
from polars_pairing import PairingFunctions

if TYPE_CHECKING:
    from polars._typing import PolarsDataType


@pytest.fixture
def df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "n1": [2, 2, 3, 4, 1_501, None],
            "n2": [6, 7, 8, 9, 10_740, 11],
        }
    )


def _expected_pair_frame(expected: list[int | None]) -> pl.DataFrame:
    return pl.DataFrame({"pair": expected}, schema={"pair": pl.UInt64})


@pytest.mark.parametrize(
    ("method", "expected"),
    [*PAIRED.items(), (None, PAIRED["hagen"])],
    ids=[*ALL_METHODS, "default"],
)
def test_pair_exact_values(
    df: pl.DataFrame, method: str | None, expected: list[int | None]
) -> None:
    kwargs: dict[str, Any] = {} if method is None else {"method": method}
    result = df.select(
        plp.col("n1").pairing.pair(plp.col("n2"), **kwargs).alias("pair")
    )
    assert_frame_equal(result, _expected_pair_frame(expected))


def test_pair_method_case_insensitive(df: pl.DataFrame) -> None:
    result = df.select(
        plp.col("n1").pairing.pair(plp.col("n2"), method="CanTor").alias("pair")  # type: ignore[arg-type]
    )
    assert_frame_equal(result, _expected_pair_frame(PAIRED["cantor"]))


@pytest.mark.parametrize("frame", ["populated", "empty"])
def test_pair_invalid_method(df: pl.DataFrame, frame: str) -> None:
    # The method is resolved up front, so it is rejected even with no rows.
    target = df if frame == "populated" else df.clear()
    with pytest.raises(pl.exceptions.PolarsError, match="Invalid pairing method"):
        target.select(
            plp.col("n1").pairing.pair(plp.col("n2"), method="invalid")  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(("n1", "n2"), [(1, -6), (-6, 1)])
@pytest.mark.parametrize("dtype", SIGNED_INTEGER_DTYPES)
def test_pair_negative_input_raises(dtype: PolarsDataType, n1: int, n2: int) -> None:
    df = pl.DataFrame({"a": [n1], "b": [n2]}, schema={"a": dtype, "b": dtype})
    with pytest.raises(pl.exceptions.ComputeError, match="pairing input out of range"):
        df.select(plp.col("a").pairing.pair(plp.col("b")))


@pytest.mark.parametrize("dtype", [pl.Int128, pl.UInt128])
def test_pair_input_beyond_uint64_raises(dtype: PolarsDataType) -> None:
    # 2**64 fits the 128-bit input dtypes but not the UInt64 encoding.
    df = pl.DataFrame({"a": [2**64], "b": [1]}, schema={"a": dtype, "b": dtype})
    with pytest.raises(pl.exceptions.ComputeError, match="pairing input out of range"):
        df.select(plp.col("a").pairing.pair(plp.col("b")))


def test_pair_out_of_range_message_names_method_and_inputs() -> None:
    df = pl.DataFrame({"a": [2, 3], "b": [6, -7]})
    with pytest.raises(
        pl.exceptions.ComputeError,
        match=r"method 'cantor' with inputs \(3, -7\)",
    ):
        df.select(plp.col("a").pairing.pair(plp.col("b"), method="cantor"))


def test_pair_out_of_range_reports_first_bad_row_not_null_row() -> None:
    df = pl.DataFrame({"a": [None, 1, 2], "b": [5, -1, -2]})
    with pytest.raises(pl.exceptions.ComputeError, match=r"inputs \(1, -1\)"):
        df.select(plp.col("a").pairing.pair(plp.col("b")))


@pytest.mark.parametrize(
    ("build", "inputs"),
    [
        (lambda: plp.col("a").pairing.pair(-1), r"\(1, -1\)"),
        (lambda: PairingFunctions(pl.lit(-3)).pair(pl.col("a")), r"\(-3, 1\)"),
    ],
    ids=["right", "left"],
)
def test_pair_out_of_range_scalar_raises(build: Any, inputs: str) -> None:
    df = pl.DataFrame({"a": _BROADCAST_A})
    with pytest.raises(pl.exceptions.ComputeError, match=f"inputs {inputs}"):
        df.select(build())


@pytest.mark.parametrize("dtype", INTEGER_DTYPES)
def test_pair_accepts_integer_dtypes(dtype: PolarsDataType) -> None:
    df = pl.DataFrame({"a": [3], "b": [8]}, schema={"a": dtype, "b": dtype})
    result = df.select(plp.col("a").pairing.pair(plp.col("b")).alias("pair"))
    assert result["pair"].dtype == pl.UInt64
    # hagen(3, 8) == 70 (pinned by test_pair_exact_values) fits every dtype.
    assert result.item(0, "pair") == PAIRED["hagen"][2]


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (pl.Series("n1", [1.5]), pl.Series("n2", [1])),
        (pl.Series("n1", [1]), pl.Series("n2", [1.5])),
        (
            pl.Series("n1", [1], dtype=pl.Decimal(10, 0)),
            pl.Series("n2", [2], dtype=pl.Decimal(10, 0)),
        ),
    ],
    ids=["float-left", "float-right", "decimal"],
)
def test_pair_rejects_non_integer(left: pl.Series, right: pl.Series) -> None:
    df = pl.DataFrame([left, right])
    with pytest.raises(pl.exceptions.PolarsError, match="only be used on integers"):
        df.select(plp.col("n1").pairing.pair(plp.col("n2")))


@pytest.mark.parametrize(("method", "left", "right", "expected"), LARGE_PAIRS)
def test_pair_large_values_exact(
    method: PairingMethod, left: int, right: int, expected: int
) -> None:
    result = u64_frame(left, right).select(
        plp.col("a").pairing.pair(plp.col("b"), method=method).alias("pair")
    )
    assert result["pair"].dtype == pl.UInt64
    assert result.item(0, "pair") == expected


@pytest.mark.parametrize("method", ALL_METHODS)
def test_pair_overflow_raises(method: PairingMethod) -> None:
    df = u64_frame(2**64 - 1, 2**64 - 1)
    with pytest.raises(pl.exceptions.PolarsError, match="pairing overflow"):
        df.select(plp.col("a").pairing.pair(plp.col("b"), method=method))


# Broadcasting: a length-1 operand is expanded against the other side.
# A Python int and pl.lit build the same plan, so each side is tested once.
# Hagen values: (1, 5), (2, 5), (3, 5) and (5, 1), (5, 2), (5, 3).
_BROADCAST_A = [1, 2, 3]


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
@pytest.mark.parametrize(
    ("build", "expected"),
    [
        (lambda: plp.col("a").pairing.pair(5), [28, 30, 32]),
        (lambda: PairingFunctions(pl.lit(5)).pair(pl.col("a")), [27, 29, 31]),
    ],
    ids=["right", "left"],
)
def test_pair_broadcasts_scalar(
    engine: Any, build: Any, expected: list[int | None]
) -> None:
    result = (
        pl.DataFrame({"a": _BROADCAST_A})
        .lazy()
        .select(build().alias("pair"))
        .collect(engine=engine)
    )
    assert_frame_equal(result, _expected_pair_frame(expected))


def test_pair_null_literal_gives_all_null() -> None:
    result = pl.DataFrame({"a": _BROADCAST_A}).select(
        plp.col("a").pairing.pair(pl.lit(None, dtype=pl.Int64)).alias("pair")
    )
    assert_frame_equal(result, _expected_pair_frame([None, None, None]))


@pytest.mark.parametrize(
    "build",
    [
        lambda: plp.col("a").pairing.pair(None),
        lambda: plp.col("a").pairing.pair(pl.col("n")),
        lambda: PairingFunctions(pl.lit(None)).pair(pl.col("a")),
        lambda: plp.col("n").pairing.pair(pl.col("a")),
    ],
    ids=["right-lit", "right-column", "left-lit", "left-column"],
)
def test_pair_null_dtype_gives_all_null(build: Any) -> None:
    # A Null-dtype operand behaves like an all-null integer, as in Polars
    # arithmetic (pl.col("a") + None).
    df = pl.DataFrame(
        {"a": _BROADCAST_A, "n": [None] * len(_BROADCAST_A)},
        schema={"a": pl.Int64, "n": pl.Null},
    )
    result = df.select(build().alias("pair"))
    assert_frame_equal(result, _expected_pair_frame([None, None, None]))


def test_pair_empty_frame_with_scalar() -> None:
    df = pl.DataFrame({"a": []}, schema={"a": pl.Int64})
    result = df.select(plp.col("a").pairing.pair(5).alias("pair"))
    assert_frame_equal(result, _expected_pair_frame([]))


@pytest.mark.parametrize(
    "build",
    [
        lambda: plp.col("a").pairing.pair(pl.lit(-1)),
        lambda: pl.lit(-3).pairing.pair(plp.col("a")),  # type: ignore[attr-defined]
    ],
    ids=["right", "left"],
)
def test_pair_empty_frame_with_out_of_range_scalar(build: Any) -> None:
    # No output rows, so the out-of-range scalar is never paired.
    df = pl.DataFrame({"a": []}, schema={"a": pl.Int64})
    result = df.select(build().alias("pair"))
    assert_frame_equal(result, _expected_pair_frame([]))


# The out-of-range case pins that the shape error takes precedence.
@pytest.mark.parametrize("other", [[1, 2], [-1, -2]], ids=["valid", "out-of-range"])
def test_pair_length_mismatch_raises(other: list[int]) -> None:
    df = pl.DataFrame({"a": _BROADCAST_A})
    with pytest.raises(pl.exceptions.PolarsError, match="different lengths"):
        df.select(plp.col("a").pairing.pair(pl.Series(other)))

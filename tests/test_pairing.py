import polars as pl
from polars.testing import assert_frame_equal

import polars_pairing as plp


def test_pairing_default():
    df = pl.DataFrame(
        {
            "n1": [2, 2, 3, 4, 1_501, None],
            "n2": [-6, 7, 8, 9, 10_740, 11],
        }
    )
    result = df.select(plp.col("n1").pairing.pair(plp.col("n2")).alias("pair"))
    expected_df = pl.DataFrame(
        {
            "pair": [None, 54, 70, 90, 115_350_602, None],
        },
        schema=pl.Schema({"pair": pl.UInt64}),
    )
    assert_frame_equal(result, expected_df)


def test_pairing_hagen():
    df = pl.DataFrame(
        {
            "n1": [2, 2, 3, 4, 1_501, None],
            "n2": [-6, 7, 8, 9, 10_740, 11],
        }
    )
    result = df.select(
        plp.col("n1").pairing.pair(plp.col("n2"), method="hagen").alias("pair")
    )
    expected_df = pl.DataFrame(
        {
            "pair": [None, 54, 70, 90, 115_350_602, None],
        },
        schema=pl.Schema({"pair": pl.UInt64}),
    )
    assert_frame_equal(result, expected_df)


def test_pairing_szudzik():
    df = pl.DataFrame(
        {
            "n1": [2, 2, 3, 4, 1_501, None],
            "n2": [-6, 7, 8, 9, 10_740, 11],
        }
    )
    result = df.select(
        plp.col("n1").pairing.pair(plp.col("n2"), method="szudzik").alias("pair")
    )
    expected_df = pl.DataFrame(
        {
            "pair": [None, 51, 67, 85, 115_349_101, None],
        },
        schema=pl.Schema({"pair": pl.UInt64}),
    )
    assert_frame_equal(result, expected_df)


def test_pairing_cantor():
    df = pl.DataFrame(
        {
            "n1": [2, 2, 3, 4, 1_501, None],
            "n2": [-6, 7, 8, 9, 10_740, 11],
        }
    )
    result = df.select(
        plp.col("n1").pairing.pair(plp.col("n2"), method="cantor").alias("pair")
    )
    expected_df = pl.DataFrame(
        {
            "pair": [None, 52, 74, 100, 74_937_901, None],
        },
        schema=pl.Schema({"pair": pl.UInt64}),
    )
    assert_frame_equal(result, expected_df)


def test_pairing_method_uppercase():
    df = pl.DataFrame(
        {
            "n1": [2, 2, 3, 4, 1_501, None],
            "n2": [-6, 7, 8, 9, 10_740, 11],
        }
    )
    result = df.select(
        plp.col("n1").pairing.pair(plp.col("n2"), method="Szudzik").alias("pair")
    )
    expected_df = pl.DataFrame(
        {
            "pair": [None, 51, 67, 85, 115_349_101, None],
        },
        schema=pl.Schema({"pair": pl.UInt64}),
    )
    assert_frame_equal(result, expected_df)

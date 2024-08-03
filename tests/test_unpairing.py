import polars as pl
from polars.testing import assert_frame_equal

import polars_pairing as plp


def test_unpairing_default():
    df = pl.DataFrame(
        {
            "pair": [None, 51, 67, 85, 115_349_101, None],
        }
    )
    result = df.select(plp.col("pair").pairing.unpair().alias("unpair")).unnest(
        "unpair"
    )
    expected_df = pl.DataFrame(
        {
            "Left": [None, 7, 8, 9, 10_740, None],
            "Right": [None, 1, 1, 2, 750, None],
        },
        schema=pl.Schema({"Left": pl.UInt64, "Right": pl.UInt64}),
    )
    assert_frame_equal(result, expected_df)


def test_unpairing_hagen():
    df = pl.DataFrame(
        {
            "pair": [None, 51, 67, 85, 115_349_101, None],
        }
    )
    result = df.select(plp.col("pair").pairing.unpair("hagen").alias("unpair")).unnest(
        "unpair"
    )
    expected_df = pl.DataFrame(
        {
            "Left": [None, 7, 8, 9, 10_740, None],
            "Right": [None, 1, 1, 2, 750, None],
        },
        schema=pl.Schema({"Left": pl.UInt64, "Right": pl.UInt64}),
    )
    assert_frame_equal(result, expected_df)


def test_unpairing_szudzik():
    df = pl.DataFrame(
        {
            "pair": [None, 51, 67, 85, 115_349_101, None],
        }
    )
    result = df.select(
        plp.col("pair").pairing.unpair(method="szudzik").alias("unpair")
    ).unnest("unpair")
    expected_df = pl.DataFrame(
        {
            "Left": [None, 2, 3, 4, 1_501, None],
            "Right": [None, 7, 8, 9, 10_740, None],
        },
        schema=pl.Schema({"Left": pl.UInt64, "Right": pl.UInt64}),
    )
    assert_frame_equal(result, expected_df)


def test_unpairing_cantor():
    df = pl.DataFrame(
        {
            "pair": [None, 52, 74, 100, 74_937_901, None],
        }
    )
    result = df.select(
        plp.col("pair").pairing.unpair(method="cantor").alias("unpair")
    ).unnest("unpair")
    expected_df = pl.DataFrame(
        {
            "Left": [None, 2, 3, 4, 1_501, None],
            "Right": [None, 7, 8, 9, 10_740, None],
        },
        schema=pl.Schema({"Left": pl.UInt64, "Right": pl.UInt64}),
    )
    assert_frame_equal(result, expected_df)


def test_unpairing_uppercase():
    df = pl.DataFrame(
        {
            "pair": [None, 51, 67, 85, 115_349_101, None],
        }
    )
    result = df.select(
        plp.col("pair").pairing.unpair(method="Szudzik").alias("unpair")
    ).unnest("unpair")
    expected_df = pl.DataFrame(
        {
            "Left": [None, 2, 3, 4, 1_501, None],
            "Right": [None, 7, 8, 9, 10_740, None],
        },
        schema=pl.Schema({"Left": pl.UInt64, "Right": pl.UInt64}),
    )
    assert_frame_equal(result, expected_df)

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, cast

import polars as pl

from polars_pairing._internal import __version__ as __version__

if TYPE_CHECKING:
    from polars._typing import IntoExpr, PolarsDataType

lib = Path(__file__).parent

Method = Literal["hagen", "szudzik", "cantor", "peter", "rosenberg_strong", "thotakura"]


@pl.api.register_expr_namespace("pairing")
class PairingFunctions:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def pair(
        self,
        other: IntoExpr,
        method: Method = "hagen",
    ) -> pl.Expr:
        """
        Applies a pairing function to uniquely encode two nonnegative integers
        into a single nonnegative integer. Null inputs give null outputs.

        Parameters
        ----------
        method
            The method to use for pairing. Either "hagen", "szudzik", "cantor",
            "peter", "rosenberg_strong" or "thotakura". Defaults to "hagen".

        Raises
        ------
        pl.exceptions.ComputeError
            If `method` is not a supported name, either input is neither an
            integer nor all-null, either input is negative or larger than the
            maximum UInt64 value, the encoded pair does not fit in a UInt64, or
            the two inputs have different lengths and neither has length 1.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "n1": [2, 2, 3, 1501, None],
        ...         "n2": [6, 7, 8, 10740, 11],
        ...     }
        ... )
        >>> df.select(pl.col("n1").pairing.pair(pl.col("n2")).alias("pair"))
        shape: (5, 1)
        ┌───────────┐
        │ pair      │
        │ ---       │
        │ u64       │
        ╞═══════════╡
        │ 40        │
        │ 54        │
        │ 70        │
        │ 115350602 │
        │ null      │
        └───────────┘
        >>> df.select(
        ...     pl.col("n1").pairing.pair(pl.col("n2"), method="hagen").alias("hagen"),
        ...     pl.col("n1").pairing.pair(pl.col("n2"), method="szudzik").alias("szudzik"),
        ...     pl.col("n1").pairing.pair(pl.col("n2"), method="cantor").alias("cantor"),
        ...     pl.col("n1").pairing.pair(pl.col("n2"), method="peter").alias("peter"),
        ...     pl.col("n1").pairing.pair(pl.col("n2"), method="rosenberg_strong").alias("rosenberg_strong"),
        ...     pl.col("n1").pairing.pair(pl.col("n2"), method="thotakura").alias("thotakura"),
        ... )
        shape: (5, 6)
        ┌───────────┬───────────┬──────────┬───────────┬──────────────────┬───────────┐
        │ hagen     ┆ szudzik   ┆ cantor   ┆ peter     ┆ rosenberg_strong ┆ thotakura │
        │ ---       ┆ ---       ┆ ---      ┆ ---       ┆ ---              ┆ ---       │
        │ u64       ┆ u64       ┆ u64      ┆ u64       ┆ u64              ┆ u64       │
        ╞═══════════╪═══════════╪══════════╪═══════════╪══════════════════╪═══════════╡
        │ 40        ┆ 38        ┆ 42       ┆ 41        ┆ 38               ┆ 34        │
        │ 54        ┆ 51        ┆ 52       ┆ 54        ┆ 51               ┆ 42        │
        │ 70        ┆ 67        ┆ 74       ┆ 71        ┆ 67               ┆ 54        │
        │ 115350602 ┆ 115349101 ┆ 74937901 ┆ 115350603 ┆ 115349101        ┆ 57698282  │
        │ null      ┆ null      ┆ null     ┆ null      ┆ null             ┆ null      │
        └───────────┴───────────┴──────────┴───────────┴──────────────────┴───────────┘
        """
        return pl.plugins.register_plugin_function(
            args=[self._expr, other],
            kwargs={"method": method},
            function_name="pair",
            is_elementwise=True,
            plugin_path=lib,
        )

    def unpair(
        self,
        method: Method = "hagen",
    ) -> pl.Expr:
        """
        Applies an unpairing function to uniquely decode a nonnegative integer
        into two nonnegative integers. Null inputs give null outputs.

        Parameters
        ----------
        method
            The method to use for unpairing. Either "hagen", "szudzik", "cantor",
            "peter", "rosenberg_strong" or "thotakura". Defaults to "hagen".

        Raises
        ------
        pl.exceptions.ComputeError
            If `method` is not a supported name, the input is neither an
            integer nor all-null, or the input is negative or larger than the
            maximum UInt64 value.

        Examples
        --------
        >>> df = pl.DataFrame({"pair": [None, 607, 4, 18871362]})
        >>> df.select(pl.col("pair").pairing.unpair().alias("unpair"))
        shape: (4, 1)
        ┌─────────────┐
        │ unpair      │
        │ ---         │
        │ struct[2]   │
        ╞═════════════╡
        │ {null,null} │
        │ {24,15}     │
        │ {0,2}       │
        │ {513,4344}  │
        └─────────────┘
        >>> df.select(pl.col("pair").pairing.unpair(method="szudzik").alias("unpair"))
        shape: (4, 1)
        ┌─────────────┐
        │ unpair      │
        │ ---         │
        │ struct[2]   │
        ╞═════════════╡
        │ {null,null} │
        │ {24,7}      │
        │ {0,2}       │
        │ {1026,4344} │
        └─────────────┘
        """
        return pl.plugins.register_plugin_function(
            args=[self._expr],
            kwargs={"method": method},
            function_name="unpair",
            is_elementwise=True,
            plugin_path=lib,
        )


class DExpr(pl.Expr):
    @property
    def pairing(self) -> PairingFunctions:
        return PairingFunctions(self)


class PairingCol(Protocol):
    def __call__(
        self,
        name: str | PolarsDataType | Iterable[str] | Iterable[PolarsDataType],
        *more_names: str | PolarsDataType,
    ) -> DExpr: ...

    def __getattr__(self, name: str) -> pl.Expr: ...

    @property
    def pairing(self) -> PairingFunctions: ...


col = cast(PairingCol, pl.col)

__all__ = ["col"]

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, cast

import polars as pl

from polars_pairing._internal import __version__ as __version__

if TYPE_CHECKING:
    from polars._typing import IntoExpr, PolarsDataType

lib = Path(__file__).parent


@pl.api.register_expr_namespace("pairing")
class PairingFunctions:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def pair(
        self, other: IntoExpr, method: Literal["hagen", "szudzik", "cantor"] = "hagen"
    ) -> pl.Expr:
        """
        Applies a pairing function to uniquely encode two nonnegative integers
        into a single nonnegative integer.

        Parameters
        ----------
        method
            The method to use for pairing. Either "hagen", "szudzik" or "cantor",
            defaults to "hagen".

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "n1": [2, 2, 3, 1501],
        ...         "n2": [-6, 7, 8, 10740],
        ...     }
        ... )
        >>> df.select(pl.col("n1").pairing.pair(pl.col("n2")).alias("pair"))
        shape: (4, 1)
        ┌───────────┐
        │ pair      │
        │ ---       │
        │ u64       │
        ╞═══════════╡
        │ null      │
        │ 54        │
        │ 70        │
        │ 115350602 │
        └───────────┘
        >>> df.select(pl.col("n1").pairing.pair(pl.col("n2"), method="szudzik").alias("pair"))
        shape: (4, 1)
        ┌───────────┐
        │ pair      │
        │ ---       │
        │ u64       │
        ╞═══════════╡
        │ null      │
        │ 51        │
        │ 67        │
        │ 115349101 │
        └───────────┘
        """
        return pl.plugins.register_plugin_function(
            args=[self._expr, other],
            kwargs={"method": method},
            function_name="pair",
            is_elementwise=True,
            plugin_path=lib,
        )

    def unpair(
        self, method: Literal["hagen", "szudzik", "cantor"] = "hagen"
    ) -> pl.Expr:
        """
        Applies an unpairing function to uniquely decode a nonnegative integer
        into two nonnegative integers.

        Parameters
        ----------
        method
            The method to use for unpairing. Either "hagen", "szudzik" or "cantor",
            defaults to "hagen".

        Examples
        --------
        >>> df = pl.DataFrame({"pair": [None, 607, -4, 18871362]})
        >>> df.select(pl.col("pair").pairing.unpair().alias("unpair"))
        shape: (4, 1)
        ┌─────────────┐
        │ unpair      │
        │ ---         │
        │ struct[2]   │
        ╞═════════════╡
        │ {null,null} │
        │ {24,15}     │
        │ {null,null} │
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
        │ {null,null} │
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

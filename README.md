# polars-pairing

This plugin provides pairing functions to Polars. These functions encode a pair of natural numbers into a single natural number.

## Installation

```bash
pip install polars-pairing
```

## Functionality and usage

This plugin implements three pairing functions (Hagen[^1], Szudzik[^2], and Cantor[^3]), along with their corresponding unpairing functions, as Polars expressions. The pairing functions take two integer columns as input and produce a single column containing the encoded pair. Conversely, the unpairing functions accept a single column as input and generate a struct column containing the values of the original pair.

```python
>>> import polars as pl
>>> import polars_pairing as plp

>>> df = pl.DataFrame({
...     "a": [1, 2, 38, 4],
...     "b": [5, 6, 77, 80]
... })
>>> df.select(plp.col("a").pairing.pair(plp.col("b")).alias("pair"))
shape: (4, 1)
┌──────┐
│ pair │
│ ---  │
│ u64  │
╞══════╡
│ 28   │
│ 40   │
│ 6006 │
│ 6408 │
└──────┘

# By default, the Hagen pairing function is used, but you can specify alternative functions:
>>> df.select(
...     plp.col("a").pairing.pair(plp.col("b"), method="hagen").alias("hagen_pair"),
...     plp.col("a").pairing.pair(plp.col("b"), method="szudzik").alias("szudzik_pair"),
...     plp.col("a").pairing.pair(plp.col("b"), method="cantor").alias("cantor_pair")
... )
shape: (4, 3)
┌────────────┬──────────────┬─────────────┐
│ hagen_pair ┆ szudzik_pair ┆ cantor_pair │
│ ---        ┆ ---          ┆ ---         │
│ u64        ┆ u64          ┆ u64         │
╞════════════╪══════════════╪═════════════╡
│ 28         ┆ 26           ┆ 26          │
│ 40         ┆ 38           ┆ 42          │
│ 6006       ┆ 5967         ┆ 6747        │
│ 6408       ┆ 6404         ┆ 3650        │
└────────────┴──────────────┴─────────────┘

# The corresponding unpairing functions are also available:
>>> df = pl.DataFrame({"p": [28, 40, 6006, 6408]})
>>> df.select(plp.col("p").pairing.unpair().alias("unpair"))
shape: (4, 1)
┌───────────┐
│ unpair    │
│ ---       │
│ struct[2] │
╞═══════════╡
│ {1,5}     │
│ {2,6}     │
│ {38,77}   │
│ {4,80}    │
└───────────┘

>>> df = pl.DataFrame({"p": [26, 38, 5967, 6404]})
>>> df.select(
...     plp.col("p").pairing.unpair(method="szudzik").alias("unpair")
... ).unnest("unpair")
shape: (4, 2)
┌──────┬───────┐
│ Left ┆ Right │
│ ---  ┆ ---   │
│ u64  ┆ u64   │
╞══════╪═══════╡
│ 1    ┆ 5     │
│ 2    ┆ 6     │
│ 38   ┆ 77    │
│ 4    ┆ 80    │
└──────┴───────┘
```

[^1]: Hagen, D. R. Superior Pairing Function. <https://drhagen.com/blog/superior-pairing-function/> (2018).
[^2]: Szudzik, Matthew. "An elegant pairing function." *Wolfram Research (ed.) Special NKS 2006 Wolfram Science Conference*. 2006.
[^3]: Cantor, G. Ein Beitrag zur Mannigfaltigkeitslehre. *Journal für die reine und angewandte Mathematik* **1878**, 242–258 (1878).
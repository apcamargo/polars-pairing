# Introduction

## About `polars-pairing`

`polars-pairing` is a plugin that provides pairing functions to [Polars](https://github.com/pola-rs/polars). These functions encode a pair of natural numbers into a single natural number.

<figure markdown="span">
  ![Image title](images/hagen_pairing_light.svg#only-light)
  ![Image title](images/hagen_pairing_dark.svg#only-dark)
  <figcaption>A pairing function performs a one-to-one mapping between <code>(x, y)</code> and <code>i</code> (represented by the circles in the Cartesian coordinate system). Figure by <a href="https://drhagen.com/blog/superior-pairing-function/">David Hagen</a>.</figcaption>
</figure>

This plugin implements six pairing functions (Hagen[^1], Szudzik[^2], Cantor[^3], Peter[^4], Rosenberg-Strong[^5], and Thotakura[^6]), along with their corresponding unpairing functions, as Polars expressions. The pairing functions take two nonnegative integer columns as input and produce a single column containing the encoded pair. Conversely, the unpairing functions accept a single column as input and generate a struct column containing the values of the original pair.

## Installation

`polars-pairing` can be installed from PyPI with `pip` or [`uv`](https://docs.astral.sh/uv/), or from [conda-forge](https://conda-forge.org/) with [`pixi`](https://pixi.sh/), [`conda`](https://docs.conda.io/projects/conda/), or [`mamba`](https://mamba.readthedocs.io/en/latest/).

=== "pip"

    ```sh
    pip install polars-pairing
    ```

=== "uv"

    ```sh
    uv init my_project && cd my_project # (1)!
    uv add polars-pairing # (2)!
    ```

    1. Create a new Python project with `uv` and navigate into the project directory.
    2. Add `polars-pairing` as a dependency to your project.

=== "Pixi"

    ```sh
    pixi init my_workspace && cd my_workspace # (1)!
    pixi add polars-pairing # (2)!
    ```

    1. Create a new Pixi workspace and navigate into the workspace directory.
    2. Add `polars-pairing` to your Pixi workspace.

=== "Conda"

    ```sh
    conda create -n my_environment -c conda-forge polars-pairing # (1)!
    conda activate my_environment # (2)!
    ```

    1. Create a new Conda environment with `polars-pairing`.
    2. Activate the environment.

=== "Mamba"

    ```sh
    mamba create -n my_environment -c conda-forge polars-pairing # (1)!
    mamba activate my_environment # (2)!
    ```

    1. Create a new Mamba environment with `polars-pairing`.
    2. Activate the environment.

## Example

```pycon
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

>>> df.select(
...     plp.col("a")
...     .pairing.pair(plp.col("b"), method="hagen") # (1)!
...     .alias("hagen_pair"),
...     plp.col("a")
...     .pairing.pair(plp.col("b"), method="szudzik")
...     .alias("szudzik_pair"),
...     plp.col("a")
...     .pairing.pair(plp.col("b"), method="cantor")
...     .alias("cantor_pair"),
...     plp.col("a")
...     .pairing.pair(plp.col("b"), method="peter")
...     .alias("peter_pair"),
...     plp.col("a")
...     .pairing.pair(plp.col("b"), method="rosenberg_strong")
...     .alias("rosenberg_strong_pair"),
...     plp.col("a")
...     .pairing.pair(plp.col("b"), method="thotakura")
...     .alias("thotakura_pair"),
... )
shape: (4, 6)
┌────────────┬──────────────┬─────────────┬────────────┬───────────────────────┬────────────────┐
│ hagen_pair ┆ szudzik_pair ┆ cantor_pair ┆ peter_pair ┆ rosenberg_strong_pair ┆ thotakura_pair │
│ ---        ┆ ---          ┆ ---         ┆ ---        ┆ ---                   ┆ ---            │
│ u64        ┆ u64          ┆ u64         ┆ u64        ┆ u64                   ┆ u64            │
╞════════════╪══════════════╪═════════════╪════════════╪═══════════════════════╪════════════════╡
│ 28         ┆ 26           ┆ 26          ┆ 28         ┆ 26                    ┆ 24             │
│ 40         ┆ 38           ┆ 42          ┆ 41         ┆ 38                    ┆ 34             │
│ 6006       ┆ 5967         ┆ 6747        ┆ 6006       ┆ 5967                  ┆ 3194           │
│ 6408       ┆ 6404         ┆ 3650        ┆ 6409       ┆ 6404                  ┆ 3368           │
└────────────┴──────────────┴─────────────┴────────────┴───────────────────────┴────────────────┘

>>> df = pl.DataFrame({"p": [28, 40, 6006, 6408]})
>>> df.select(plp.col("p").pairing.unpair().alias("unpair")) # (2)!
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

>>> df = pl.DataFrame({"p": [28, 41, 6006, 6409]})
>>> df.select(
...     plp.col("p").pairing.unpair(method="peter").alias("unpair")
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

>>> df = pl.DataFrame({"p": [26, 38, 5967, 6404]})
>>> df.select(
...     plp.col("p").pairing.unpair(method="rosenberg_strong").alias("unpair")
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

>>> df = pl.DataFrame({"p": [24, 34, 3194, 3368]})
>>> df.select(
...     plp.col("p").pairing.unpair(method="thotakura").alias("unpair")
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

1. By default, the Hagen pairing function is used, but you can specify alternative functions. The available methods are: `hagen`, `szudzik`, `cantor`, `peter`, `rosenberg_strong`, and `thotakura`.
2. The unpairing functions return a struct column containing the values of the original pair.

[^1]: Hagen, D. R. Superior Pairing Function. <https://drhagen.com/blog/superior-pairing-function/> (2018).
[^2]: Szudzik, Matthew. "An elegant pairing function." *Wolfram Research (ed.) Special NKS 2006 Wolfram Science Conference*. 2006.
[^3]: Cantor, G. Ein Beitrag zur Mannigfaltigkeitslehre. *Journal für die reine und angewandte Mathematik* **1878**, 242–258 (1878).
[^4]: Péter, R. *Rekursive Funktionen*. Akadémiai Kiadó, Budapest (1951).
[^5]: Rosenberg, A. L., and Strong, H. R. "Addressing arrays by shells." *IBM Technical Disclosure Bulletin* **14** (10), 3026–3028 (1972).
[^6]: Thotakura, Manideep. "A New Bijective Pairing Alternative for Encoding Natural Numbers." *Cryptology ePrint Archive* (2025). <https://eprint.iacr.org/2025/1244>.

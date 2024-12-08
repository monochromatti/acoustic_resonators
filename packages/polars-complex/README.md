# polars-complex


This is a [namespace
extension](https://docs.pola.rs/py-polars/html/reference/api/polars.api.register_expr_namespace.html)
for the [polars](https://github.com/pola-rs/polars) library, which
provides a complex number representation for the DataFrame type,
together with an implementation of complex number arithmetic.

To import the namespace extension, `pip install` the project and import
as `polars_complex`.

``` python
import polars as pl
import polars_complex
```

## Creating the complex number representation

We use the following DataFrame for illustration:

``` python
import numpy as np

df = pl.DataFrame({
    "x": np.random.random(6),
    "y": np.random.random(6),
})
```

    ┌───────┬───────┐
    │ x     ┆ y     │
    │ ---   ┆ ---   │
    │ f64   ┆ f64   │
    ╞═══════╪═══════╡
    │ 0.808 ┆ 0.602 │
    │ 0.443 ┆ 0.155 │
    │ 0.858 ┆ 0.653 │
    │ 0.527 ┆ 0.982 │
    │ 0.004 ┆ 0.963 │
    │ 0.566 ┆ 0.749 │
    └───────┴───────┘

The complex number struct is created in one of two ways. For create it
from two columns with arbitrary names, use

``` python
df = df.with_columns(
    pl.col("x", "y").complex.struct("z") # or .struct().alias("z[c]")
)
```

    ┌───────┬───────┬───────────────┐
    │ x     ┆ y     ┆ z[c]          │
    │ ---   ┆ ---   ┆ ---           │
    │ f64   ┆ f64   ┆ struct[2]     │
    ╞═══════╪═══════╪═══════════════╡
    │ 0.808 ┆ 0.602 ┆ {0.808,0.602} │
    │ 0.443 ┆ 0.155 ┆ {0.443,0.155} │
    │ 0.858 ┆ 0.653 ┆ {0.858,0.653} │
    │ 0.527 ┆ 0.982 ┆ {0.527,0.982} │
    │ 0.004 ┆ 0.963 ┆ {0.004,0.963} │
    │ 0.566 ┆ 0.749 ┆ {0.566,0.749} │
    └───────┴───────┴───────────────┘

Note that the `[c]` suffix is the adopted convention to signal a complex
number representation. If the columns are named with the suffixes
“.real” and “.imag”, with a common prefix,

``` python
df = pl.DataFrame({
    "z.real": np.random.random(6),
    "z.imag": np.random.random(6),
})
```

    ┌────────┬────────┐
    │ z.real ┆ z.imag │
    │ ---    ┆ ---    │
    │ f64    ┆ f64    │
    ╞════════╪════════╡
    │ 0.708  ┆ 0.038  │
    │ 0.779  ┆ 0.948  │
    │ 0.837  ┆ 0.369  │
    │ 0.262  ┆ 0.292  │
    │ 0.157  ┆ 0.004  │
    │ 0.859  ┆ 0.880  │
    └────────┴────────┘

then the construction is a little more succinct:

``` python
df = df.with_columns(
    pl.col("z").complex.struct()
)
```

    ┌────────┬────────┬───────────────┐
    │ z.real ┆ z.imag ┆ z[c]          │
    │ ---    ┆ ---    ┆ ---           │
    │ f64    ┆ f64    ┆ struct[2]     │
    ╞════════╪════════╪═══════════════╡
    │ 0.708  ┆ 0.038  ┆ {0.708,0.038} │
    │ 0.779  ┆ 0.948  ┆ {0.779,0.948} │
    │ 0.837  ┆ 0.369  ┆ {0.837,0.369} │
    │ 0.262  ┆ 0.292  ┆ {0.262,0.292} │
    │ 0.157  ┆ 0.004  ┆ {0.157,0.004} │
    │ 0.859  ┆ 0.880  ┆ {0.859,0.880} │
    └────────┴────────┴───────────────┘

## Arithmetic

Common operations, such as extracting phase, modulus, conjugation,
division, multiplication, and addition, are supported. They result in a
new complex number representation, or a numerical representation,
depending on the operation.

Using the dataframe above, here are some examples:

``` python
df = df.select(
    pl.col("z[c]"),
    (pl.col("z[c]").complex.conj()).alias("conj(z)[c]"),
    (pl.col("z[c]").complex.modulus()).alias("|z|"),
    (pl.col("z[c]").complex.phase()).alias("arg(z)"),
    (pl.col("z[c]") / 2.024).alias("w[c]"),
).with_columns(
    pl.col("z[c]").complex.divide(pl.col("w[c]")).alias("z/w[c]"),
    pl.col("z[c]").complex.multiply(pl.col("w[c]")).alias("z*w[c]"),
)
```

    ┌───────────────┬────────────────┬───────┬────────┬───────────────┬────────────────┬───────────────┐
    │ z[c]          ┆ conj(z)[c]     ┆ |z|   ┆ arg(z) ┆ w[c]          ┆ z/w[c]         ┆ z*w[c]        │
    │ ---           ┆ ---            ┆ ---   ┆ ---    ┆ ---           ┆ ---            ┆ ---           │
    │ struct[2]     ┆ struct[2]      ┆ f64   ┆ f64    ┆ struct[2]     ┆ struct[2]      ┆ struct[2]     │
    ╞═══════════════╪════════════════╪═══════╪════════╪═══════════════╪════════════════╪═══════════════╡
    │ {0.708,0.038} ┆ {0.708,-0.038} ┆ 0.709 ┆ 1.518  ┆ {0.350,0.019} ┆ {2.024,0.000}  ┆ {0.247,0.026} │
    │ {0.779,0.948} ┆ {0.779,-0.948} ┆ 1.227 ┆ 0.688  ┆ {0.385,0.468} ┆ {2.024,0.000}  ┆ {-0.144,0.730 │
    │               ┆                ┆       ┆        ┆               ┆                ┆ }             │
    │ {0.837,0.369} ┆ {0.837,-0.369} ┆ 0.915 ┆ 1.155  ┆ {0.414,0.183} ┆ {2.024,-0.000} ┆ {0.279,0.306} │
    │ {0.262,0.292} ┆ {0.262,-0.292} ┆ 0.392 ┆ 0.731  ┆ {0.129,0.144} ┆ {2.024,-0.000} ┆ {-0.008,0.076 │
    │               ┆                ┆       ┆        ┆               ┆                ┆ }             │
    │ {0.157,0.004} ┆ {0.157,-0.004} ┆ 0.157 ┆ 1.548  ┆ {0.078,0.002} ┆ {2.024,0.000}  ┆ {0.012,0.001} │
    │ {0.859,0.880} ┆ {0.859,-0.880} ┆ 1.230 ┆ 0.773  ┆ {0.424,0.435} ┆ {2.024,0.000}  ┆ {-0.018,0.747 │
    │               ┆                ┆       ┆        ┆               ┆                ┆ }             │
    └───────────────┴────────────────┴───────┴────────┴───────────────┴────────────────┴───────────────┘

## Converting back to numeric representation

Converting back (unnesting the structs) can be done either in the
dataframe namespace,

``` python
df.complex.unnest()
```

    ┌───────┬────────┬────────┬────────┬───┬──────────┬──────────┬──────────┬──────────┐
    │ |z|   ┆ arg(z) ┆ z.real ┆ z.imag ┆ … ┆ z/w.real ┆ z/w.imag ┆ z*w.real ┆ z*w.imag │
    │ ---   ┆ ---    ┆ ---    ┆ ---    ┆   ┆ ---      ┆ ---      ┆ ---      ┆ ---      │
    │ f64   ┆ f64    ┆ f64    ┆ f64    ┆   ┆ f64      ┆ f64      ┆ f64      ┆ f64      │
    ╞═══════╪════════╪════════╪════════╪═══╪══════════╪══════════╪══════════╪══════════╡
    │ 0.709 ┆ 1.518  ┆ 0.708  ┆ 0.038  ┆ … ┆ 2.024    ┆ 0.000    ┆ 0.247    ┆ 0.026    │
    │ 1.227 ┆ 0.688  ┆ 0.779  ┆ 0.948  ┆ … ┆ 2.024    ┆ 0.000    ┆ -0.144   ┆ 0.730    │
    │ 0.915 ┆ 1.155  ┆ 0.837  ┆ 0.369  ┆ … ┆ 2.024    ┆ -0.000   ┆ 0.279    ┆ 0.306    │
    │ 0.392 ┆ 0.731  ┆ 0.262  ┆ 0.292  ┆ … ┆ 2.024    ┆ -0.000   ┆ -0.008   ┆ 0.076    │
    │ 0.157 ┆ 1.548  ┆ 0.157  ┆ 0.004  ┆ … ┆ 2.024    ┆ 0.000    ┆ 0.012    ┆ 0.001    │
    │ 1.230 ┆ 0.773  ┆ 0.859  ┆ 0.880  ┆ … ┆ 2.024    ┆ 0.000    ┆ -0.018   ┆ 0.747    │
    └───────┴────────┴────────┴────────┴───┴──────────┴──────────┴──────────┴──────────┘

or on individual columns,

``` python
df.select(
    pl.col("w[c]").complex.unnest()
)
```

    ┌────────┬────────┐
    │ w.real ┆ w.imag │
    │ ---    ┆ ---    │
    │ f64    ┆ f64    │
    ╞════════╪════════╡
    │ 0.350  ┆ 0.019  │
    │ 0.385  ┆ 0.468  │
    │ 0.414  ┆ 0.183  │
    │ 0.129  ┆ 0.144  │
    │ 0.078  ┆ 0.002  │
    │ 0.424  ┆ 0.435  │
    └────────┴────────┘

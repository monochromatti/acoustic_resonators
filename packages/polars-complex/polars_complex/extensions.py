import json
import re
import warnings

import polars as pl
from polars.expr.expr import Expr

from . import arithmetic as ar


@pl.api.register_expr_namespace("complex")
class ComplexMethods:
    def __init__(self, expr: Expr):
        self.expr = expr

    @property
    def names(self):
        meta_dict = json.loads(self.expr.meta.serialize(format="json"))
        if self.expr.meta.has_multiple_outputs():
            names = ComplexMethods.find_item(meta_dict, "Columns")
            return names
        else:
            name = self.expr.meta.output_name()
            name = re.sub(r"\.real|\.imag", "", name)
            return [name]

    def format_complex(self, name: str):
        return name if name.endswith("[c]") else name + "[c]"

    def struct(self, name: str = None):
        if self.expr.meta.has_multiple_outputs():
            name_real, name_imag = self.names
            stem = name or name_real.replace(".real", "")
            return pl.struct(
                pl.col(name_real).alias("real"), pl.col(name_imag).alias("imag")
            ).alias(self.format_complex(stem))
        else:
            alias = self.format_complex(name or self.names[0])
            print(alias)
            return pl.struct(self.expr.alias("real"), pl.lit(0).alias("imag")).alias(
                alias
            )

    def nest(self, *args, **kwargs):
        return self.struct(*args, **kwargs)

    def to_list(self):
        stem = self.names[0].replace("[c]", "")
        self.expr = self.expr.struct.rename_fields([f"{stem}.real", f"{stem}.imag"])
        return [
            self.expr.struct.field("real"),
            self.expr.struct.field("imag"),
        ]

    def real(self):
        return self.expr.struct.field("real").alias(self.names[0])

    def imag(self):
        return self.expr.struct.field("imag").alias(self.names[0])

    @staticmethod
    def find_item(d, key="Columns"):
        if isinstance(d, dict):
            if key in d:
                return d[key]
            else:
                for value in d.values():
                    result = ComplexMethods.find_item(value, key)
                    if result is not None:
                        return result
        elif isinstance(d, list):
            for item in d:
                result = ComplexMethods.find_item(item, key)
                if result is not None:
                    return result
        return None

    def add(self, other):
        return ar.add(self.expr, other)

    def subtract(self, other):
        return ar.subtract(self.expr, other)

    def multiply(self, other):
        return ar.multiply(self.expr, other)

    def divide(self, other):
        return ar.divide(self.expr, other)

    def inverse(self):
        return ar.inverse(self.expr)

    def exp(self):
        return ar.exp(self.expr)

    def sin(self):
        return ar.sin(self.expr)

    def cos(self):
        return ar.cos(self.expr)

    def pow(self):
        return ar.pow(self.expr)

    def squared_modulus(self):
        return ar.squared_modulus(self.expr)

    def modulus(self):
        return ar.modulus(self.expr)

    def phase(self):
        return ar.phase(self.expr)

    def unwrap_phase(self):
        return ar.unwrap_phase(self.expr)

    def conj(self):
        return ar.conj(self.expr)


@pl.api.register_dataframe_namespace("complex")
class ComplexFrame:
    def __init__(self, df: pl.DataFrame):
        self._df = df

    def unnest(self, names: str | list[str] = None):
        df = self._df
        if names is None:
            names = [name for name in df.columns if name.endswith("[c]")]
        elif not isinstance(names, list):
            names = [names]
        if len(names) < 1:
            warnings.warn("No complex columns found.")
            return df
        else:
            stems = [name.replace("[c]", "") for name in names]
            return df.select(
                pl.all().exclude(names),
                *[
                    pl.col(name).struct.rename_fields([f"{stem}.real", f"{stem}.imag"])
                    for (name, stem) in zip(names, stems)
                ],
            ).unnest(*names)

    def nest(self, varnames: str | list[str] = None):
        if varnames is None:
            varnames = [
                name.replace(".real", "")
                for name in self._df.columns
                if name.endswith(".real")
                and name.replace(".real", ".imag") in self._df.columns
            ]
            varnames = list(set(varnames))
        elif isinstance(varnames, str):
            varnames = [varnames]

        complex_columns = self._df.select(
            pl.col(f"{varname}.{comp}")
            for varname in varnames
            for comp in ("real", "imag")
        ).columns

        for varname in varnames:
            if f"{varname}.real" not in complex_columns:
                raise ValueError(f"Column {varname}.real missing.")
            if f"{varname}.imag" not in complex_columns:
                raise ValueError(f"Column {varname}.imag missing.")

        return self._df.select(
            pl.all().exclude(complex_columns),
            *[
                pl.struct(
                    pl.col(f"{var}.real").alias("real"),
                    pl.col(f"{var}.imag").alias("imag"),
                ).alias(f"{var}[c]")
                for var in varnames
            ],
        )

    def struct(self, *args, **kwargs):
        return self.nest(*args, **kwargs)


class ComplexExpr(Expr):
    def __init__(self, expr: Expr):
        self.expr = expr
        self._pyexpr = expr._pyexpr

    def __getattr__(self, name: str) -> Expr:
        if hasattr(self.expr.complex, name):
            return getattr(self.expr.complex, name)
        return getattr(self.expr, name)

    def coerce_other(self, other):
        if isinstance(other, (int, float)):
            return pl.struct(pl.lit(other).alias("real"), pl.lit(0.0).alias("imag"))
        elif isinstance(other, type(self)):
            return other.expr
        elif isinstance(other, Expr):
            if other.meta.is_column_selection():
                if other.meta.has_multiple_outputs():
                    msg = (
                        "Invalid object for complex arithmetic."
                        "\nSupported types are ComplexExpr, struct[2], float, int."
                    )
                    raise ValueError(msg)
                else:
                    return pl.struct(other.alias("real"), pl.lit(0.0).alias("imag"))
            return other.struct.rename_fields(["real", "imag"])
        else:
            raise ValueError(f"Cannot coerce {type(other)} to Expr.")

    def __add__(self, other):
        other = self.coerce_other(other)
        return ComplexExpr(self.expr.complex.add(other))

    def __radd__(self, other):
        other = self.coerce_other(other)
        return ComplexExpr(self.__add__(other))

    def __sub__(self, other):
        other = self.coerce_other(other)
        return ComplexExpr(self.expr.complex.subtract(other))

    def __rsub__(self, other):
        other = self.coerce_other(other)
        return ComplexExpr((other - self.expr) * -1)

    def __mul__(self, other):
        other = self.coerce_other(other)
        return ComplexExpr(self.expr.complex.multiply(other))

    def __rmul__(self, other):
        other = self.coerce_other(other)
        return ComplexExpr(self.__mul__(other))

    def __truediv__(self, other):
        other = self.coerce_other(other)
        return ComplexExpr(self.expr.complex.divide(other))

    def __rtruediv__(self, other):
        other = self.coerce_other(other)
        return ComplexExpr(self.expr.complex.inverse() * other)

    def exp(self):
        return ComplexExpr(self.expr.complex.exp())

    def sin(self):
        return ComplexExpr(self.expr.complex.sin())

    def cos(self):
        return ComplexExpr(self.expr.complex.cos())

    def pow(self, n):
        return ComplexExpr(ar.pow(self.expr, n))

    def squared_modulus(self):
        return ComplexExpr(self.expr.complex.squared_modulus())

    def modulus(self):
        return ComplexExpr(self.expr.complex.modulus())

    def phase(self):
        return ComplexExpr(self.expr.complex.phase())

    def unwrap_phase(self):
        return ComplexExpr(self.expr.complex.unwrap_phase())

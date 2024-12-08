from typing import Iterable, cast

from polars.expr.expr import Expr
from polars.functions.col import Col
from polars import col
from polars._typing import PolarsDataType

from . import extensions  # noqa: F401
from .extensions import ComplexExpr


class ComplexExprFactory:
    def __new__(
        cls, name: str | PolarsDataType | Iterable[str] | Iterable[PolarsDataType]
    ) -> ComplexExpr:
        return ComplexExpr(col(name))

    def __call__(
        self, name: str | PolarsDataType | Iterable[str] | Iterable[PolarsDataType]
    ) -> ComplexExpr:
        return ComplexExpr(col(name))

    def __getattr__(self, name: str) -> Expr:
        return getattr(type(self), name)


ccol = cast(Col, ComplexExprFactory)

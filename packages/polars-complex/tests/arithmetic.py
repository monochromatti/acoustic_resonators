import unittest

import numpy as np
import polars as pl
from polars import col

from polars_complex import ccol

test_data = pl.DataFrame(
    {
        "a": np.random.rand(10),
        "b": np.random.rand(10),
        "c": np.random.rand(10),
        "d": np.random.rand(10),
    }
).select(
    col("a", "b").complex.nest().alias("z1[c]"),
    col("c", "d").complex.nest().alias("z2[c]"),
    col("a"),
)


class TestAddition(unittest.TestCase):
    def test_ccol_ccol(self):
        test_data.with_columns((ccol("z1[c]") + ccol("z2[c]")).alias("z1+z2"))

    def test_ccol_col(self):
        test_data.with_columns((ccol("z1[c]") + col("a")).alias("z1+a"))

    def test_col_ccol(self):
        test_data.with_columns((col("a") + ccol("z1[c]")).alias("a+z1"))

    def test_ccol_float(self):
        test_data.with_columns((ccol("z1[c]") + 1.0).alias("z1+1"))

    def test_float_ccol(self):
        test_data.with_columns((1.0 + ccol("z1[c]")).alias("1+z1"))


class TestSubtraction(unittest.TestCase):
    def test_ccol_ccol(self):
        test_data.with_columns((ccol("z1[c]") - ccol("z2[c]")).alias("z1-z2"))

    def test_ccol_col(self):
        test_data.with_columns((ccol("z1[c]") - col("a")).alias("z1-a"))

    def test_col_ccol(self):
        test_data.with_columns((col("a") - ccol("z1[c]")).alias("a-z1"))

    def test_ccol_float(self):
        test_data.with_columns((ccol("z1[c]") - 1.0).alias("z1-1"))

    def test_float_ccol(self):
        test_data.with_columns((1.0 - ccol("z1[c]")).alias("1-z1"))


class TestMultiplication(unittest.TestCase):
    def test_ccol_ccol(self):
        test_data.with_columns((ccol("z1[c]") * ccol("z2[c]")).alias("z1/z2"))

    def test_ccol_col(self):
        test_data.with_columns((ccol("z1[c]") * col("a")).alias("z1/a"))

    def test_ccol_float(self):
        test_data.with_columns((ccol("z1[c]") * 1.0).alias("z1/a"))

    def test_float_ccol(self):
        test_data.with_columns((1.0 * ccol("z1[c]")).alias("1/z1"))


class TestDivision(unittest.TestCase):
    def test_ccol_ccol(self):
        test_data.with_columns((ccol("z1[c]") / ccol("z2[c]")).alias("z1/z2"))

    def test_ccol_col(self):
        test_data.with_columns((ccol("z1[c]") / col("a")).alias("z1/a"))

    def test_col_ccol(self):
        test_data.with_columns((col("a") / ccol("z1[c]")).alias("a/z1"))

    def test_ccol_float(self):
        test_data.with_columns((ccol("z1[c]") / 1.0).alias("z1/a"))

    def test_float_ccol(self):
        test_data.with_columns((1.0 / ccol("z1[c]")).alias("1/z1"))


class TestTranscendental(unittest.TestCase):
    def test_exp(self):
        test_data.with_columns((ccol("z1[c]").exp()).alias("exp(z1)"))

    def test_sin(self):
        test_data.with_columns((ccol("z1[c]").sin()).alias("sin(z1)"))

    def test_cos(self):
        test_data.with_columns((ccol("z1[c]").cos()).alias("cos(z1)"))

    def test_pow(self):
        test_data.with_columns((ccol("z1[c]").pow(2)).alias("z1^2"))


if __name__ == "__main__":
    unittest.main()

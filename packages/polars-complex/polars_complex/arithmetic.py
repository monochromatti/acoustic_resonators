import polars as pl


def split_complex(expr):
    return (
        expr.struct.field("real"),
        expr.struct.field("imag"),
    )


def phase(expr):
    real, imag = split_complex(expr)
    return pl.arctan2(real, imag)


def unwrap_phase(expr):
    diff = expr - expr.shift()
    if_lower = pl.when(diff < -3.14159).then(diff + 2 * 3.14159)
    if_higher = pl.when(diff > 3.14159).then(diff - 2 * 3.14159)
    corrected_diff = (
        if_higher.otherwise(if_lower.otherwise(diff)).fill_null(0).cum_sum()
    )
    return expr.head(1).cast(pl.Float64) + corrected_diff


def modulus(expr):
    return squared_modulus(expr).sqrt()


def squared_modulus(expr):
    real, imag = split_complex(expr)
    return real.pow(2) + imag.pow(2)


def conj(expr):
    real, imag = split_complex(expr)
    return pl.struct(
        real,
        (imag * -1).alias("imag"),
    )


def divide(expr, other):
    a, b = split_complex(expr)
    c, d = split_complex(other)
    return pl.struct(
        ((a * c + b * d) / (c.pow(2) + d.pow(2))).alias("real"),
        ((b * c - a * d) / (c.pow(2) + d.pow(2))).alias("imag"),
    )


# def quotient(expr, other):
#     return divide(expr, other)


# def ratio(expr, other):
#     return divide(expr, other)


def inverse(expr):
    real, imag = split_complex(expr)
    sqmod = real.pow(2) + imag.pow(2)
    real /= sqmod
    imag /= sqmod * -1
    return pl.struct(
        real.alias("real"),
        imag.alias("imag"),
    )


def multiply(expr, other):
    a, b = split_complex(expr)
    c, d = split_complex(other)
    return pl.struct(
        (a * c - b * d).alias("real"),
        (a * d + b * c).alias("imag"),
    )


def add(expr, other):
    a, b = split_complex(expr)
    c, d = split_complex(other)
    return pl.struct(
        (a + c).alias("real"),
        (b + d).alias("imag"),
    )


def subtract(expr, other):
    a, b = split_complex(expr)
    c, d = split_complex(other)
    return pl.struct(
        (a - c).alias("real"),
        (b - d).alias("imag"),
    )


# def difference(expr, other):
#     return subtract(expr, other)


def exp(expr):
    real, imag = split_complex(expr)
    exp_real = real.exp()
    return pl.struct(
        (exp_real * imag.cos()).alias("real"),
        (exp_real * imag.sin()).alias("imag"),
    )


def sin(expr):
    real, imag = split_complex(expr)
    return pl.struct(
        (real.sin() * imag.cosh()).alias("real"),
        (real.cos() * imag.sinh()).alias("imag"),
    )


def cos(expr):
    real, imag = split_complex(expr)
    return pl.struct(
        (real.cos() * imag.cosh()).alias("real"),
        (real.sin() * imag.sinh() * -1).alias("imag"),
    )


def pow(expr, n: int):
    angle = n * phase(expr)
    amplitude = modulus(expr).pow(n)
    return pl.struct(
        (amplitude * angle.cos()).alias("real"),
        (amplitude * angle.sin()).alias("imag"),
    )

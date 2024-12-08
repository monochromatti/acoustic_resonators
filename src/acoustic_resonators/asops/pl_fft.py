import json
import logging

import hvplot
import lmfit as lm
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import polars.selectors as cs

from acoustic_resonators.asops.config import paths as proj_paths

def pl_fft(df, xname, id_vars=None):
    def varname_iter(fft_dict, value_vars):
        operators = {"real": np.real, "imag": np.imag}
        for name in value_vars:
            for component in ("real", "imag"):
                yield pl.Series(
                    f"{name}.{component}",
                    operators[component](fft_dict[name]),
                )

    def fftfreq(df):
        return np.fft.rfftfreq(
            len(df[xname]),
            abs(df[xname][1] - df[xname][0]),
        )

    value_vars = [var for var in df.columns if var not in id_vars and var != xname]

    frames = []
    if not id_vars:
        fft_dict = {
            name: np.fft.rfft(df[name].to_numpy())
            for name in value_vars
        }
        frames.append(
            pl.DataFrame(
                (
                    pl.Series("freq", fftfreq(df)),
                    *varname_iter(fft_dict, value_vars),
                )
            )
        )
    else:
        for id_vals, group in df.group_by(*id_vars):
            if isinstance(id_vals, (float, int, str)):
                id_vals = [id_vals]
            fft_dict = {
                name: np.fft.rfft(group[name].to_numpy())
                for name in value_vars
            }
            frames.append(
                pl.DataFrame(
                    (
                        pl.Series("freq", fftfreq(group)),
                        *varname_iter(fft_dict, value_vars),
                    )
                )
                .with_columns(
                    pl.lit(value).alias(name)
                    for name, value in zip(id_vars, list(id_vals))
                )
                .select(
                    *(pl.col(name) for name in id_vars),
                    "freq",
                    pl.all().exclude("freq", *id_vars),
                )
            )
    return pl.concat(frames)

import logging

import lmfit as lm
import numpy as np
import polars as pl

from acoustic_resonators import create_dataset
from acoustic_resonators.stroboscopy.config import paths as proj_paths


def pl_fft(df, xname, id_vars=None) -> pl.DataFrame:
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
        fft_dict = {name: np.fft.rfft(df[name].to_numpy()) for name in value_vars}
        frames.append(
            pl.DataFrame(
                (
                    pl.Series("freq", fftfreq(df)),
                    *varname_iter(fft_dict, value_vars),
                )
            )
        )
    else:
        for id_vals, group in df.group_by(id_vars):
            if isinstance(id_vals, (float, int, str)):
                id_vals = [id_vals]
            fft_dict = {
                name: np.fft.rfft(group[name].to_numpy()) for name in value_vars
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


def sample_comparison() -> None:
    paths = pl.read_csv(
        proj_paths.file_lists / "sample_comparison.csv", comment_prefix="#"
    ).with_columns(
        pl.col("path").map_elements(lambda path: str(proj_paths.raw_data / path))
    )
    data = create_dataset(
        paths,
        column_names=["delay", "X", "Y"],
        id_schema={"sample": str},
        lockin_schema={"X": ("X", "Y")},
        x0=224.9,
    )

    offsets = {"SRO/CF": -0.3, "SRO/STO": 0.1, "STO": 0.0, "CF": 3.5}

    data = (
        data.with_columns(
            (pl.col("time") - pl.col("sample").replace(offsets).cast(pl.Float64)).alias(
                "time"
            ),
            (-1 * (pl.col("X") - pl.col("X").slice(3, 3).mean()))
            .over("sample")
            .alias("X"),
        )
        .group_by(["sample"])
        .agg(pl.all().slice(2, pl.col("time").count() - 2))
        .explode(pl.all().exclude("sample"))
    )

    data.write_csv(proj_paths.processed_data / "sample_comparison.csv")


def wavelength_comparison() -> None:
    paths = pl.read_csv(proj_paths.file_lists / "MIRvVIS.csv").with_columns(
        (str(proj_paths.raw_data) + "/" + pl.col("path")).alias("path")
    )

    data = create_dataset(
        paths,
        column_names=["delay", "X1", "Y1", "X2", "Y2", "R"],
        lockin_schema={
            "rf_signal": ("X1", "Y1"),
            "monitor": ("X2", "Y2"),
            "R": "R",
        },
        id_schema={"wavelength": pl.Float64, "power": pl.Float64},
    )

    zeroes = {
        15.0: 234.6,
        0.64: 232.72,
    }

    data = data.with_columns(
        (6.67 * (pl.col("wavelength").replace(zeroes) + pl.col("time") / 6.67)).alias(
            "time"
        ),
        pl.col("monitor").median().over("wavelength").alias("median_monitor"),
        pl.col("rf_signal").slice(0, 5).mean().over("wavelength").alias("rf_offset"),
        pl.col("R").slice(0, 5).mean().over("wavelength").alias("R_offset"),
    ).with_columns(
        (
            (pl.col("rf_signal") - pl.col("rf_offset"))
            / (pl.col("median_monitor") * pl.col("power"))
        )
        .over("power")
        .alias("rf_signal"),
        (
            (pl.col("R") - pl.col("R_offset"))
            / (pl.col("median_monitor") * pl.col("power"))
        )
        .over("power")
        .alias("R"),
    )

    data = data.select(
        pl.col("wavelength"),
        pl.col("time"),
        pl.col("rf_signal").alias("rotation"),
        pl.col("R").alias("transmission"),
    )

    data.write_csv(proj_paths.processed_data / "wavelength_comparison.csv")


def temperature_bfield() -> None:
    files = (
        proj_paths.raw_data / path
        for path in (proj_paths.root / "file_lists/xTxM.csv").read_text().splitlines()
    )
    frames = []
    for path in files:
        df = pl.read_csv(
            str(path),
            separator="\t",
            has_header=False,
            comment_prefix="#",
            new_columns=["delay", "X1", "Y1", "X2", "Y2", "X3", "Y3"],
        ).select(
            pl.lit(path.name).alias("filename"),
            pl.col("delay"),
            pl.col("^X.*$"),
        )
        frames.append(df)

    data = pl.concat(frames)
    data = (
        data.with_columns((6.67 * (232.8 - pl.col("delay"))).alias("time"))
        .filter(pl.col("time") > pl.col("time").slice(1, 1))
        .with_columns(pl.col("X2").median().over("filename").alias("X_total"))
        .select(
            pl.col("filename"),
            pl.col("time"),
            (
                (pl.col("X1") - pl.col("X1").slice(0, 5).mean()) / pl.col("X_total")
            ).alias("rotation"),
            (
                (pl.col("X3") - pl.col("X3").slice(0, 5).mean()) / pl.col("X_total")
            ).alias("transmission"),
        )
    )

    field_pattern = r"Mag=(-?\d+\.\d+).txt"
    tempr_pattern = r"Tem=(-?\d+\.\d+)_Mag"
    data = data.with_columns(
        bfield=pl.col("filename").str.extract(field_pattern).cast(pl.Float32),
        temperature=pl.col("filename").str.extract(tempr_pattern).cast(pl.Float32),
    ).select(
        pl.col(
            "filename",
            "temperature",
            "bfield",
            "time",
            "rotation",
            "transmission",
        )
    )

    data.write_csv(proj_paths.processed_data / "temperature_bfield.csv")

    data = data.with_columns(
        pl.col("bfield").abs().alias("|bfield|"),
    )

    data = (
        data.filter(pl.col("bfield") > 0)
        .select(
            pl.col("temperature", "time", "bfield", "|bfield|"),
            pl.col("filename", "rotation", "transmission").name.suffix("_pos"),
        )
        .join(
            data.filter(pl.col("bfield") < 0).select(
                pl.col("temperature", "time", "bfield", "|bfield|"),
                pl.col("filename", "rotation", "transmission").name.suffix("_neg"),
            ),
            on=["temperature", "time", "|bfield|"],
        )
        .select(
            pl.col("filename_neg", "filename_pos", "temperature", "time", "|bfield|"),
            (pl.col("rotation_pos") - pl.col("rotation_neg")).alias("rotation.odd"),
            (pl.col("rotation_pos") + pl.col("rotation_neg")).alias("rotation.even"),
            (pl.col("transmission_pos") - pl.col("transmission_neg")).alias(
                "transmission.odd"
            ),
            (pl.col("transmission_pos") + pl.col("transmission_neg")).alias(
                "transmission.even"
            ),
        )
    )
    data.write_csv(proj_paths.processed_data / "temperature_|bfield|.csv")

    return data


def damped_cosine(t, t0, amp, freq, tau, phi) -> np.ndarray:
    return (
        amp
        * np.cos(2 * 3.1415 * freq * (t - t0) + 3.1415 * phi)
        * np.exp(-(t - t0) / tau)
    )


def tanh_step(t, t0, sigma) -> np.ndarray:
    return np.tanh((t - t0) / sigma) / 2 + 0.5


def exponential_decay(t, t0, amp, tau) -> np.ndarray:
    return amp * np.exp(-(t - t0) / tau)


def step_decay_osc(
    t,
    t0,
    sigma,
    amp_osc,
    freq_osc,
    tau_osc,
    phi_osc,
    amp_thermal,
    tau_thermal,
) -> np.ndarray:
    return tanh_step(t, t0, sigma) * (
        exponential_decay(t, t0, amp_thermal, tau_thermal)
        + damped_cosine(t, t0, amp_osc, freq_osc, tau_osc, phi_osc)
    )


def xP_VIS():
    paths = pl.read_csv(
        proj_paths.file_lists / "xP_VIS.csv",
    ).with_columns(
        pl.col("path").map_elements(lambda path: str(proj_paths.raw_data / path))
    )
    data = create_dataset(
        paths,
        column_names=["delay", "X1", "Y1", "X2", "Y2", "X3", "Y3"],
        lockin_schema={"X1": ("X1", "Y1"), "X2": ("X2", "Y2"), "X3": ("X3", "Y3")},
        id_schema={"power": pl.Float64},
        x0=224.9,
    ).join(paths.rename({"path": "filename"}), on="power")
    data.id_vars += ["filename"]
    data = data.sort_columns()

    step_index = data.select(
        (pl.col("X1").rolling_mean(3) - pl.col("X1").shift(-3).rolling_mean(3))
        .abs()
        .arg_max()
        .over("filename")
        .alias("step_index"),
        pl.col("filename"),
        pl.col("time"),
    )
    data = data.join(step_index, on=["filename", "time"])

    t0 = pl.col("time").get(pl.col("step_index").first())
    data = data.with_columns((pl.col("time") - t0).over("filename"))

    step = ((pl.col("time") / 0.1).tanh() + 1) / 2
    data = data.with_columns(
        (pl.col("X1") / pl.col("X2").median()).over("filename").alias("dR/R")
    ).with_columns((-1 * pl.col("dR/R") * step).alias("dR/R"))

    data = data.select(
        pl.col("filename", "power", "time", "dR/R"),
    )

    model = lm.Model(step_decay_osc)
    params = lm.Parameters()
    params.add_many(
        # name, value, vary, min, max, expr
        ("t0", 0, False, -1e-2, 1e-2),
        ("sigma", 0.1, True, 0.01, 1),
        ("amp_osc", -1e-2, True, -1, 1),
        ("freq_osc", 0.13, True, 0.10, 0.14),
        ("tau_osc", 20, True, 1, 100),
        ("phi_osc", 0, True, -1, 1),
        ("amp_thermal", -0.01, True, -0.1, 0),
        ("tau_thermal", 300, True, 100, 1000),
    )

    def fit_model(df: pl.DataFrame) -> tuple[pl.DataFrame, lm.model.ModelResult]:
        fitresult = model.fit(
            df["dR/R"].to_numpy(),
            t=df["time"].to_numpy(),
            params=params,
        )
        return df.with_columns(
            residual=fitresult.residual,
            best_fit=fitresult.best_fit,
        ), fitresult

    results = []

    def store_results(df, fitfunc):
        filename = df["filename"][0]
        power = df["power"][0]
        df, res = fitfunc(df)

        results.append(
            pl.DataFrame(
                {
                    key: value
                    for pname, p in res.params.items()
                    for key, value in {
                        f"{pname}.value": p.value,
                        f"{pname}.stderr": p.stderr,
                    }.items()
                    if res.params[pname].vary
                }
            ).with_columns(
                pl.lit(filename).alias("filename"),
                pl.lit(power).alias("power"),
            )
        )
        return df

    data = (
        data.group_by(["filename", "power"])
        .map_groups(lambda group: store_results(group, fit_model))
        .sort("filename", "time", "power")
    )
    data.write_csv(proj_paths.processed_data / "xP_VIS.csv")

    results = pl.concat(results).sort("power")
    results.write_csv(proj_paths.processed_data / "xP_VIS_fit.csv")

    data_FT = pl_fft(data, "time", id_vars=["filename"]).sort("filename", "freq")
    data_FT.write_csv(proj_paths.processed_data / "xP_VIS_FT.csv")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    logging.info("Running `sample_comparison`")
    sample_comparison()

    logging.info("Running `wavelength_comparison`")
    wavelength_comparison()

    logging.info("Running `temperature_bfield`")
    temperature_bfield()

    logging.info("Running `xP_VIS`")
    data = xP_VIS()

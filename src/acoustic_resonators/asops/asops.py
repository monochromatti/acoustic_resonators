import json
import logging
import re
from pathlib import Path

import h5py
import lmfit as lm
import numpy as np
import polars as pl


import matplotlib.pyplot as plt
import mplstylize
import seaborn as sns
import hvplot

hvplot.extension("plotly")

from acoustic_resonators.asops.config import paths as proj_paths


def remove_slope(df):
    slope_model = lm.models.LinearModel()
    slope_params = slope_model.make_params(intercept=0, slope=0)

    slope_result = slope_model.fit(
        df.select(pl.col("signal").slice(0, 100)).get_columns(),
        x=df.select(pl.col("time").slice(0, 100)).get_columns(),
        params=slope_params,
    )
    s = pl.lit(slope_result.eval(x=df.get_column("time")))
    return df.with_columns((pl.col("signal") - s).alias("signal"))


def generate_frames(paths):
    for path in paths:
        with h5py.File(str(path), "r") as hdf5_file:
            time = 1e9 * np.array(hdf5_file["treal"]).squeeze()
            measurements = {
                str(i): np.asarray(
                    hdf5_file[hdf5_file["data"][i, 0]]["dev2434"]["outputpwas"]["wave"][
                        "x"
                    ]
                ).squeeze()
                for i in range(hdf5_file["data"].shape[0])
            }

            data = (
                pl.LazyFrame(measurements)
                .with_columns(
                    pl.lit(Path(path).stem).alias("filename"),
                    pl.lit(time).alias("time"),
                )
                .melt(
                    id_vars=["time", "filename"],
                    value_name="signal",
                    variable_name="scan_num",
                )
                .sort("time", "scan_num")
                .with_columns(
                    pl.col("scan_num").cast(pl.Int32),
                    (pl.col("signal") - pl.col("signal").slice(0, 10).mean()).over(
                        "scan_num"
                    ),
                )
            )

            data = data.group_by("scan_num", "filename").map_groups(
                remove_slope, schema=data.schema
            )  # Removal of linear background (detection artefact)

            data = (
                data.group_by("time", "filename")
                .agg(
                    pl.col("signal").mean().alias("signal"),
                    pl.col("signal").std().alias("std"),
                )
                .drop("scan_num")
                .sort("time")
            )  # Average over scans

            step_index = (
                data.select(
                    (
                        pl.col("signal").rolling_mean(2)
                        - pl.col("signal").shift(2).rolling_mean(2)
                    )
                    .arg_min()
                    .alias("step_index")
                )
                .collect()
                .item()
            )

            data = data.with_columns(
                (pl.col("time") - pl.col("time").get(step_index)).alias("time"),
                pl.col("signal"),
            )  # Shift time by step detection

            yield data.select(
                "filename",
                "time",
                "signal",
                "std",
            )


def find_temperature(set_label):
    regex_pattern = r"STAMP_(\d+|NoHeat)"
    result = re.search(regex_pattern, set_label).group(1)
    if result == "NoHeat":
        return 60
    return int(result)



def compute_scale():
    measurements = {}
    measurements["flake1"] = {
        "waveplate_angle": (170, 165, 160, 155, 150, 145, 140, 135, 130, 125, 120),
        "voltage_rf": (-133, -60, -16, -6, -30, -89, -175, -272, -387, -484, -562),
    }
    measurements["flake2"] = {
        "waveplate_angle": (100, 110, 120, 130, 140, 150, 170),
        "voltage_rf": (-512, -612, -569, -391, -179, -31, -135),
    }

    def balance_model(angle, amp, offset, yoffset):
        return amp * np.sin(2 * angle * np.pi / 180 - offset) ** 2 + yoffset

    model = lm.Model(balance_model)
    params = model.make_params(amp=3000, offset=124, yoffset=0)

    amplitudes = {}

    gain_correction = 6 / 2  # Calibration at 2X, dynamics measurements at 6X

    for key, value in measurements.items():
        voltage_rf = np.array(value["voltage_rf"]) * gain_correction
        waveplate_angle = value["waveplate_angle"]
        res = model.fit(voltage_rf, params, angle=waveplate_angle)
        amplitudes[key] = res.params["amp"].value / 2

    return np.mean(list(amplitudes.values()))  # [mV]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="%H:%M:%S")

    def verify(filename):
        correct_filetype = filename.suffix == ".mat"
        correct_data = "SRO" in filename.name
        return correct_filetype and correct_data

    folders = [f"{d}.07.2023" for d in ("03", "04", "05", "06", "07", "10")]
    paths = [
        file
        for date in folders
        for file in (Path("./raw_data") / date).iterdir()
        if verify(file)
    ]

    logging.info("Reading from HDF5 files")
    data = pl.concat(generate_frames(paths)).sort("filename", "time")

    # Parse filenames
    data = data.select(
        pl.col("filename"),
        pl.col("filename").str.split("_").list.get(-5).alias("label"),
        pl.col("filename").map_elements(find_temperature).alias("temperature"),
        pl.col("time"),
        pl.col("signal"),
        pl.col("std"),
    )

    # Compute equilibrium voltage (<=> R)
    total_voltage = compute_scale()  # [mV]
    data = data.with_columns(
        (pl.col("signal") / (1e-3 * total_voltage)).alias("signal")
    )

    # Materialize data
    logging.info("LazyFrame -> DataFrame")
    data = data.collect()

    logging.info("Writing data to CSV")
    data.write_csv(proj_paths.processed_data / "data.csv")

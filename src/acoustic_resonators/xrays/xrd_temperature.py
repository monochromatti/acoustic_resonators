import re

import lmfit as lm
import numpy as np
import polars as pl
import xrayutilities as xu
from acoustic_resonators.asops.config import paths as asops_paths
from acoustic_resonators.xrays.config import paths as xray_paths

from scipy.interpolate import PchipInterpolator
from xrayutilities.materials.elements import O, Ru, Sr


def to_distance(x):
    lda = 0.15406  # [nm], Cu K-alpha
    return lda / (2 * np.sin(np.deg2rad(x) / 2)) * 2


def pseudovoigt(x, amp, cen, sigma, fraction):
    def gaussian(x, amp, cen, sigma):
        return amp * np.exp(-((x - cen) ** 2) / (2.0 * sigma**2))

    def lorentzian(x, amp, cen, sigma):
        return amp / (1 + ((x - cen) / sigma) ** 2)

    return fraction * gaussian(x, amp, cen, sigma) + (1 - fraction) * lorentzian(
        x, amp, cen, sigma
    )


def logpseudovoigt(x, amp, cen, sigma, fraction):
    return np.log10(pseudovoigt(x, amp, cen, sigma, fraction))


def log2pseudovoigt(x, amp1, cen1, sigma1, fraction1, amp2, cen2, sigma2, fraction2):
    return np.log10(
        pseudovoigt(x, amp1, cen1, sigma1, fraction1)
        + pseudovoigt(x, amp2, cen2, sigma2, fraction2)
    )


def load_xrd_data() -> pl.DataFrame:
    folder = xray_paths.raw_data / "heating/XRDs/"
    filenames = list(folder.glob("*"))

    xrd_data = pl.concat(
        pl.DataFrame(
            xu.io.getxrdml_scan(filename),
            schema={"2theta": pl.Float64, "intensity": pl.Float64},
        ).with_columns(
            pl.lit(float(re.search(r"(\d+)(?=øC)", filename.name).group())).alias(
                "temperature"
            )
        )
        for filename in filenames
    ).sort("temperature", "2theta")

    xrd_data = (
        xrd_data.with_columns(
            (pl.col("intensity") / pl.col("intensity").max())
            .over("temperature")
            .alias("norm_intensity"),
        )
        .with_columns(
            pl.col("norm_intensity").log10().alias("log_intensity"),
        )
        .with_columns(
            (pl.col("log_intensity") - pl.col("log_intensity").slice(0, 50).mean())
        )
        .with_columns(
            pl.col("2theta").map_batches(to_distance).alias("lattice_spacing"),
        )
    )

    temperature_order = {27: 1} | {100 + 50 * i: i + 2 for i in range(14)} | {25: 16}

    xrd_data = xrd_data.with_columns(
        pl.col("temperature").replace(temperature_order).cast(pl.Int32).alias("number")
    ).sort("number", "2theta")

    return xrd_data.select(
        "temperature", "number", "2theta", "lattice_spacing", "^.*intensity.*$"
    )


def fit_xrd(xrd_data: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    best_params = []

    def fit_group(data: pl.DataFrame):
        pseudovoigt_model = lm.Model(log2pseudovoigt)
        pseudovoigt_params = lm.Parameters()

        x = data["2theta"].to_numpy()
        y = data["log_intensity"].to_numpy()
        temperature = data["temperature"][0]
        number = data["number"][0]

        fit_range = slice(250, 600)
        center_sto = 46.5 - (46.5 - 46.1) / 725 * temperature

        # (name, value, vary, min, max)
        pseudovoigt_params.add_many(
            ("amp1", 3e6, True, 1e6, 4e6),
            ("cen1", center_sto, True, 0, None),
            ("sigma1", 0.005, True, 0, 1e-2),
            ("fraction1", 0.9, True, 0, 1),
            ("amp2", 60, True, 10, 100),
            ("cen2", 45.8, True, 45.2, 46.3),
            ("sigma2", 0.15, True, 0.05, 0.35),
            ("fraction2", 1, True, 0, 1),
        )

        pseudovoigt_fitresult = pseudovoigt_model.fit(
            y[fit_range], pseudovoigt_params, x=x[fit_range], method="nelder-mead"
        )
        best_params.append(
            {"temperature": temperature, "number": number}
            | pseudovoigt_fitresult.params.valuesdict()
        )

        data = data.with_columns(
            pl.lit(pseudovoigt_fitresult.eval(x=x)).alias("best_fit")
        )

        return data

    xrd_data = xrd_data.group_by("temperature").map_groups(fit_group)
    best_params = (
        pl.DataFrame(best_params)
        .with_columns(
            pl.col("cen1").map_batches(to_distance).alias("c1"),
            pl.col("cen2").map_batches(to_distance).alias("c2"),
        )
        .sort("number")
    )
    return xrd_data, best_params


def load_xrr_data() -> pl.DataFrame:
    folder = xray_paths.raw_data / "heating/XRR1/"
    filenames = folder.glob("*.xrdml")
    temperature_order = {27: 1} | {100 + 50 * i: i + 2 for i in range(14)} | {25: 16}
    xrr_data = (
        pl.concat(
            pl.DataFrame(
                xu.io.getxrdml_scan(filename),
                schema={"2theta": pl.Float64, "intensity": pl.Float64},
            ).with_columns(
                pl.lit(float(re.search(r"(\d+)(?=øC)", filename.name).group())).alias(
                    "temperature"
                )
            )
            for filename in filenames
        )
        .with_columns(
            pl.col("temperature")
            .replace(temperature_order)
            .cast(pl.Int32)
            .alias("number")
        )
        .sort("number", "2theta")
        .with_columns(
            (pl.col("intensity") / pl.col("intensity").max()).over("temperature")
        )
    )
    return xrr_data


def fit_xrr(xrr_data: pl.DataFrame) -> list[pl.DataFrame, pl.DataFrame]:
    STO = xu.materials.SrTiO3
    SRO = xu.materials.Crystal(
        "SrRuO3",
        xu.materials.SGLattice(
            123, 3.905, 3.95, atoms=[Sr, Ru, O, O], pos=["1a", "1d", "1c", "2e"]
        ),
    )

    sub = xu.simpack.Layer(STO, float("inf"), roughness=1, density=5130)
    lay1 = xu.simpack.Layer(SRO, 350, roughness=4, density=5700)
    m = xu.simpack.SpecularReflectivityModel(
        sub + lay1,
        energy="CuKa1",
        sample_width=5,
        beam_width=0.4,
        resolution_width=0.002,
        background=2,
        I0=1,
    )
    fitm = xu.simpack.FitModel(m)

    fitm.set_param_hint("SrTiO3_density", vary=False)
    fitm.set_param_hint("SrTiO3_roughness", vary=True, value=1, min=0)
    fitm.set_param_hint("SrRuO3_density", vary=True, value=5600)
    fitm.set_param_hint("SrRuO3_thickness", vary=True, value=350, min=300, max=400)
    fitm.set_param_hint("SrRuO3_roughness", vary=True, value=6, min=0, max=10)
    fitm.set_param_hint("I0", vary=True, value=20)
    fitm.set_param_hint("background", vary=True, value=0)
    p = fitm.make_params()

    best_params = []

    def fit_group(data: pl.DataFrame):
        x = data["2theta"].to_numpy()
        y = data["intensity"].to_numpy()
        temperature = data["temperature"][0]

        p = fitm.make_params()

        fit_range = slice(20, 250)
        result = fitm.fit(y[fit_range], p, x[fit_range] / 2)

        data = data.with_columns(pl.lit(result.eval(x=x / 2)).alias("best_fit"))
        best_params.append({"temperature": temperature} | result.params.valuesdict())

        return data

    xrr_data = xrr_data.group_by("temperature").map_groups(fit_group)
    best_params = pl.DataFrame(best_params).sort("temperature")

    return xrr_data, best_params


if __name__ == "__main__":
    xrd_data = load_xrd_data()
    xrd_data, xrd_params = fit_xrd(xrd_data)
    xrd_data.write_csv(xray_paths.processed_data / "xrd_temperature.csv")
    xrd_params.write_csv(xray_paths.processed_data / "xrd_temperature_params.csv")

    xrr_data = load_xrr_data()
    xrr_data, xrr_params = fit_xrr(xrr_data)
    xrr_data.write_csv(xray_paths.processed_data / "xrr_temperature.csv")
    xrr_params.write_csv(xray_paths.processed_data / "xrr_temperature_params.csv")

    # import seaborn as sns

    # g = sns.relplot(
    #     xrr_params.select(
    #         "temperature", "background", "SrTiO3_roughness", "^SrRuO3.*$"
    #     ).melt(id_vars="temperature"),
    #     x="temperature",
    #     y="value",
    #     col="variable",
    #     col_wrap=3,
    #     facet_kws=dict(sharey=False, despine=False),
    #     height=1.7,
    # )
    # g.set_titles("{col_name}")
    # g.set_xlabels(r"$T$ (°C)")
    # g.figure.set_size_inches((6.8, 6.8))

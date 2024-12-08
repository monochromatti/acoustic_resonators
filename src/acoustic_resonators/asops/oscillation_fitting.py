import json
import logging

import hvplot
import lmfit as lm
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import polars.selectors as cs
from polars_dataset import Dataset, Datafile

hvplot.extension("plotly")

from acoustic_resonators.asops.config import paths as proj_paths


DATAFILES = {
    "modelled_data": Datafile(
        name="modelled_data",
        directory=proj_paths.processed_data,
        index="time",
        id_vars=["filename", "label", "temperature"],
    ),
    "average_models": Datafile(
        name="average_models",
        directory=proj_paths.processed_data,
        index="time",
        id_vars=["temperature"],
    ),
    "optimal_parameters": Datafile(
        name="optimal_parameters",
        directory=proj_paths.processed_data,
        id_vars=["label", "temperature"],
    ),
}


def initialize_params(params_json=None):
    params_json = (
        params_json
        or proj_paths.root / "oscillation_fitting/initial_guesses/default.json"
    )
    with open(params_json) as f:
        return lm.create_params(**json.load(f))


def initialize_model():
    def damped_cosine(x, x0, amplitude, frequency, decay, phase):
        return (
            amplitude
            * np.exp(-(x - x0) / decay)
            * np.cos(2 * np.pi * frequency * (x - x0) + phase)
        )

    def exp_decay(x, x0, amplitude, decay):
        return amplitude * np.exp(-(x - x0) / (decay + 1e-16))

    def tanh_step(x, x0, sigma):
        return (np.tanh((x - x0) / (sigma + 1e-16)) + 1) / 2

    osc1 = lm.Model(damped_cosine, prefix="osc1_")
    osc2 = lm.Model(damped_cosine, prefix="osc2_")
    osc3 = lm.Model(damped_cosine, prefix="osc3_")
    exp_fast = lm.Model(exp_decay, prefix="exp_fast_")
    exp_slow = lm.Model(exp_decay, prefix="exp_slow_")
    step = lm.Model(tanh_step, prefix="step_")

    model = (osc1 + osc2 + osc3 + exp_fast + exp_slow) * step
    model.background = (exp_fast + exp_slow) * step
    model.brillouin = osc1 * step
    model.first_harmonic = osc2 * step
    model.third_harmonic = osc3 * step

    return model


def oscillation_minimize(data):
    df_list, results_list = [], []
    for (temperature,), df in data.group_by(["temperature"]):
        model = initialize_model()
        params = initialize_params(
            proj_paths.root
            / f"oscillation_fitting/initial_guesses/{temperature}degC.json"
        )
        results = {}

        def fit_group(group):
            x, y = group.select("time", "signal").get_columns()

            label = group["label"][0]
            logging.info(f"Fitting {label} at {temperature}°C")
            try:
                bgr_params = model.background.make_params()

                for name in bgr_params:
                    if (p := params.get(name)) is not None:
                        bgr_params[name].set(
                            value=p.value,
                            vary=p.vary,
                            min=p.min,
                            max=p.max,
                            expr=p.expr,
                        )
                bgr_fit = model.background.fit(
                    y, bgr_params, x=x, method="least_squares"
                )

                x0 = bgr_fit.params["step_x0"]
                sig = bgr_fit.params["step_sigma"]

                params.update(bgr_fit.params)
                params["step_x0"].set(vary=False)
                params["step_sigma"].set(vary=False)

                weights = 1 - np.exp(
                    -((x.to_numpy() - x0.value) ** 2) / (4 * sig.value**2)
                )

                fit_result = model.fit(
                    y, params, x=x, method="least_squares", weights=weights
                )
                results[label] = fit_result
                return group.with_columns(
                    pl.lit(fit_result.residual).alias("residual"),
                    pl.lit(model.background.eval(fit_result.params, x=x)).alias(
                        "background"
                    ),
                    pl.lit(model.brillouin.eval(fit_result.params, x=x)).alias(
                        "brillouin"
                    ),
                    pl.lit(model.first_harmonic.eval(fit_result.params, x=x)).alias(
                        "first_harmonic"
                    ),
                    pl.lit(model.third_harmonic.eval(fit_result.params, x=x)).alias(
                        "third_harmonic"
                    ),
                ).with_columns((pl.col("time") - x0.value).alias("time"))
            except Exception:
                logging.warning(f"Failed fitting {label} at {temperature}°C")
                plt.plot(x, y, label=label)
                plt.plot(x, model.eval(params, x=x), label="initial guess")
                plt.xlim(-0.01, 0.1)
                plt.legend()
                plt.show()

        df_list.append(df.group_by("label").map_groups(fit_group))

        results_list.append(
            pl.concat(
                pl.DataFrame(
                    [
                        [name, r.params[name].value, r.params[name].stderr]
                        for name in r.params
                    ],
                    schema={
                        "variable": pl.Utf8,
                        "value": pl.Float64,
                        "stderr": pl.Float64,
                    },
                ).with_columns(pl.lit(label).alias("label"))
                for label, r in results.items()
            ).with_columns(
                pl.lit(temperature).alias("temperature"),
            )
        )

    data = pl.concat(df_list)
    results = pl.concat(results_list)

    return data, results


def average_parameters(results, zero=True):
    mean_params = {}
    for params_dict in (
        results.group_by("temperature", "variable")
        .agg(
            pl.col("value").mean(),
            pl.col("stderr").pow(2).mean().sqrt(),
        )
        .pivot(columns=["variable"], values="value", index="temperature")
        .to_dicts()
    ):
        params = initialize_params()

        for name, value in params_dict.items():
            if name == "temperature":
                temperature = value
            elif (p := params.get(name)) is not None:
                p.set(value=value)
            else:
                raise ValueError(f"Parameter {name} not found in model.")

        if zero:
            params["step_x0"].set(value=0.0)

        mean_params[temperature] = params

    return mean_params


def apply_spline(group, xi, id_vars):
    id_vals = group.select(pl.col(*id_vars).first())
    group = group.select(
        pl.struct(xi.name, col)
        .splines.spline(xi=list(xi), method="catmullrom")
        .alias(col)
        for col in filter(lambda col: col not in id_vars + [xi.name], group.columns)
    ).with_columns(xi, *id_vals)
    return group


def eval_average_model(params, time):
    model = initialize_model()
    return pl.DataFrame(
        {
            "time": time,
            "background": model.background.eval(params, x=time),
            "brillouin": model.brillouin.eval(params, x=time),
            "first_harmonic": model.first_harmonic.eval(params, x=time),
            "third_harmonic": model.third_harmonic.eval(params, x=time),
        }
    )


if __name__ == "__main__":
    data = pl.read_csv(proj_paths.processed_data / "data.csv").filter(
        ~pl.col("label").str.contains("double|spotty|translucent")
    )
    id_vars = ["filename", "label", "temperature"]

    data, results = oscillation_minimize(data)

    import seaborn as sns

    g = sns.catplot(
        results,
        x="temperature",
        y="value",
        col="variable",
        kind="violin",
        height=1.7,
        col_wrap=3,
        facet_kws=dict(despine=False),
        sharey=False,
    )
    g.set_titles("{col_name}")

    # DATAFILES["optimal_parameters"].write(results)

    # t_min, t_max = (
    #     data.group_by(id_vars)
    #     .agg(
    #         pl.col("time").min().alias("min_time"),
    #         pl.col("time").max().alias("max_time"),
    #     )
    #     .select(pl.col("min_time").max(), pl.col("max_time").min())
    # ).row(0)
    # dt = 4e-4
    # time = pl.Series("time", np.arange(t_min, t_max, dt))

    # data = (
    #     data.group_by(id_vars, maintain_order=True)
    #     .map_groups(lambda group: apply_spline(group, time, id_vars))
    #     .drop_nulls()
    # )

    # data = data.select(cs.by_name(*id_vars, "time"), ~cs.by_name(*id_vars, "time"))
    # data = Dataset(data, index="time", id_vars=id_vars)

    # DATAFILES["modelled_data"].write(data)

    # time_extended = pl.Series("time", np.arange(-0.1, 2, dt))
    # average_models = Dataset(
    #     [
    #         eval_average_model(params, time_extended).with_columns(
    #             pl.lit(temperature).alias("temperature")
    #         )
    #         for temperature, params in average_parameters(results).items()
    #     ],
    #     index="time",
    #     id_vars=["temperature"],
    # )
    # DATAFILES["average_models"].write(average_models)

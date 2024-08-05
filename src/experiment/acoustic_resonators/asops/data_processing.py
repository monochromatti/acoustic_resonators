import os
import re

import h5py
import lmfit as lm
import numpy as np
import pandas as pd
import polars as pl


def process_datafiles(paths):
    # Initialize a list to hold the results
    results = []

    # Loop over all files in 'paths'
    for k, path in enumerate(paths):
        # Open the file
        with h5py.File(path, "r") as hdf5_file:
            # Get 'treal' and 'data'
            treal = np.array(hdf5_file["treal"]).squeeze()

            data = hdf5_file["data"]

            # Initialize lists to hold offset-corrected data, mean curves, and standard deviations
            corrected_data = []
            mean_curve = np.zeros_like(treal)
            std_dev = np.zeros_like(treal)

            # Loop through each measurement in 'data'
            for i in range(data.shape[0]):
                # Get the reference
                ref = data[i, 0]

                # Dereference the object and get 'x' data
                measurement = hdf5_file[ref]["dev2434"]["outputpwas"]["wave"]["x"]
                measurement_array = np.asarray(measurement).squeeze()

                # Define the condition for offset correction
                treal -= treal[find_zero(measurement_array)]

                # Correct the offset
                corrected_measurement = measurement_array - measurement_array[treal < 0].mean()

                # Add the corrected data to the list
                corrected_data.append(corrected_measurement)

                # Add the corrected data to the mean curve
                mean_curve += corrected_measurement

            # Calculate the mean curve and standard deviation
            mean_curve /= data.shape[0]
            for corrected_measurement in corrected_data:
                std_dev += (corrected_measurement - mean_curve) ** 2
            std_dev = np.sqrt(std_dev / data.shape[0])

            filename = os.path.basename(path)
            parts = filename.split("_")
            set_label = parts[3]
            label = re.match(r"([a-z]+)([0-9]+)", parts[4], re.I)
            scan_label = label.group(1)
            scan_index = int(label.group(2))
            temperature = extract_temperature(set_label)

            # Append the results for this file to the results list
            results.append(
                {
                    "temperature": float(temperature),
                    "scan_label": str(scan_label),
                    "scan_number": int(scan_index),
                    "file_index": int(k),
                    "mean_curve": list(mean_curve),
                    "std_dev": list(std_dev),
                    "treal": list(1e9 * (treal - treal[find_zero(mean_curve)])),
                }
            )

    # Convert the results to a DataFrame
    data = (
        pl.DataFrame(results)
        .sort(by=["temperature", "scan_label", "scan_number", "file_index"])
        .explode("mean_curve", "std_dev", "treal")
    )

    return data


def extract_temperature(set_label):
    try:
        return int(re.match(r"(\d+)(\w+)", set_label).group(1))
    except AttributeError:
        return 60


def load_file(file):
    with h5py.File(file, "r") as f:
        reflectivity = np.array(f.get("Sig"))
    return reflectivity


def get_label(file):
    file.split("_")[-5]


def get_flake_number(filename, idname="flake"):
    return int(re.search(rf"{idname}(\d+)", filename).group(1))


def import_data(files, keys):
    data_dict = {}
    for i in range(len(files)):
        data_dict[keys[i]] = load_file(files[i])
    return pd.DataFrame.from_dict(data_dict)


def find_zero(waveform):
    """Finds the position at which the largest value jump occurs"""
    return np.argmax(-np.diff(waveform))


def remove_thermal(waveform, time):
    background_model = lm.models.StepModel(prefix="step_") * (
        lm.models.ExponentialModel(prefix="expfast_")
        + lm.models.ExponentialModel(prefix="expslow_")
    )
    background_params = background_model.make_params()
    params_dict = {
        "step_amplitude": {"value": 1, "vary": False},
        "step_center": {"value": 0.0, "min": -1e-1, "max": 1e-1, "vary": True},
        "step_sigma": {"value": 0.0, "min": 0.0, "max": 1.0, "vary": False},
        "expslow_amplitude": {"value": -1e-3, "min": -1e-2, "max": 0, "vary": True},
        "expslow_decay": {"value": 2, "min": 1, "max": 10.0},
        "expfast_amplitude": {"value": -1e-3, "min": -1e-2, "max": 0, "vary": True},
        "expfast_decay": {"value": 0.05},
    }
    for name, param in params_dict.items():
        background_params[name].set(**param)

    res = background_model.fit(waveform, background_params, x=time)

    return res


def model_constructor(hi=True, split=False, mid=False, lo=False):
    def lorentzian(x, amplitude, center, sigma):
        return (amplitude / np.pi) * (
            0.5 * sigma / ((0.5 * sigma) ** 2 + (x - center) ** 2)
        )

    # (NAME, VALUE, VARY, MIN, MAX,  EXPR,  BRUTE_STEP)
    noise_params = (("noise_c", 2e-4, True, 0.0, 1e-3),)
    hi_params = (
        ("hi_amplitude", 0.01, True, 0.0, 0.05),
        ("hi_center", 79, True, 65, 85),
        ("hi_sigma", 5, True, 2, 20),
    )
    hi2_params = (
        ("hi2_amplitude", 0.01, True, 0.0, 0.05),
        ("hi2_center", 75, True, 65, 85),
        ("hi2_sigma", 5, True, 2, 20),
    )
    mid_params = (
        ("mid_amplitude", 0, True, 0.0, 1e-2),
        ("mid_center", 47, True, 40, 51),
        ("mid_sigma", 5, True, 1, 30),
    )
    lo_params = (
        ("lo_amplitude", 0.01, True, 0.0, 0.05),
        ("lo_center", 5, True, 0, 10),
        ("lo_sigma", 5, True, 2, 20),
    )

    params = lm.Parameters()
    model = lm.models.ConstantModel(prefix="noise_")
    if hi:
        model += lm.Model(lorentzian, prefix="hi_")
        params.add_many(*hi_params, *noise_params)
    if split:
        model += lm.Model(lorentzian, prefix="hi2_")
        params.add_many(*hi2_params, *noise_params)
    if mid:
        model += lm.Model(lorentzian, prefix="mid_")
        params.add_many(*mid_params, *noise_params)
    if lo:
        model += lm.Model(lorentzian, prefix="lo_")
        params.add_many(*lo_params, *noise_params)
    return model, params


# def get_spectrum(waveform):
#     fft = np.fft.rfft(waveform)

#     phase = np.unwrap(np.angle(spectrum))
#     start_index, stop_index = np.argmin(np.abs(f - 30)), np.argmin(np.abs(f - 200))
#     a, b = np.polyfit(f[start_index:stop_index], phase[start_index:stop_index], 1)
#     phase -= a * f + b

#     amp = np.abs(spectrum)

#     return amp, phase


# def fit_spectrum(spectrum):
#     phase = np.unwrap(np.angle(spectrum))

#     model, params = model_constructor(hi=True, split=False, mid=True, lo=True)
#     res = model.fit(np.absspectrum[1:], params, x=f[1:], method="nelder-mead")
#     ax_freq[i].plot(f, spectrum, color=colors[i], marker="o", ms=1)
#     ax_freq[i].plot(f_cont, res.eval(x=f_cont), color="k", lw=0.5)

#     fit_results.append(res)

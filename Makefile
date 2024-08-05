ENV_NAME=acoustic_resonators

env: init_env compile_polars_splines install_polars_plugins install_experiment_package

init_env:
	conda env create -f environment.yml

compile_polars_splines:
	conda run --name $(ENV_NAME) maturin develop --release --manifest-path ./src/polars-splines/Cargo.toml

install_polars_plugins:
	conda run --name $(ENV_NAME) pip install ./src/scientific_mplstyle ./src/polars-complex ./src/polars-dataset --no-build-isolation

install_experiment_package:
	conda run --name $(ENV_NAME) pip install -e ./src/experiment --no-build-isolation

download_data:
	conda run --name $(ENV_NAME) python data_download.py

.PHONY: env init_env activate_env compile_polars_splines install_polars_plugins install_experiment_package
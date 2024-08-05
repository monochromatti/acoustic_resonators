import numpy as np
import polars as pl
from polars_dataset import Dataset


def create_dataset(
    paths: pl.DataFrame,
    column_names: list[str],
    index: str,
    lockin_schema: dict[str, tuple[str, str] | str],
    id_schema: dict[pl.DataType] = None,
    **kwargs,
) -> Dataset:
    """Create a Dataset from a list of data files.

    Parameters
    ----------
    paths : pl.DataFrame
        A DataFrame with at last one column, named "path", containing the paths to the files.
    column_names : list[str]
        A list of column names to use for the data. Order should match the order of columns in the data files.
    index : str
    lockin_schema : dict[tuple[str] | str, str]
        Data files may contain two channels (X and Y). If so, the relative phase
        will be adjusted to maximize the amplitude of the X channel, and only X will be
        retained. The dictionary entry of this data has the structure
            - key (str): new name for retained column
            - value (tuple[str]): 2-tuple of names of X and Y channel, respectively.
        If only one single channel, the dictionary entry has the structure
            - key (str): new name for column
            - value (str): name of column in data file
    id_schema : dict[pl.DataType], optional
        Polars data type of the columns of `path` representing id (not index) parameters.

    Returns
    -------
    Dataset
        A Dataset object.
    """

    kwargs = {
        "separator": kwargs.pop("separator", "\t"),
        "has_header": kwargs.pop("has_header", False),
        "comment_prefix": kwargs.pop("comment_prefix", "#"),
        **kwargs,
    }
    pair_dict = {k: v for k, v in lockin_schema.items() if isinstance(v, tuple)}
    lone_dict = {k: v for k, v in lockin_schema.items() if isinstance(v, str)}

    lockin_exprs = [
        pl.col(x, y).complex.struct().map_batches(zero_quadrature).alias(name)
        for name, (x, y) in pair_dict.items()
    ]
    lockin_exprs += [pl.col(col).alias(name) for name, col in lone_dict.items()]

    frames = []
    if not id_schema:
        id_schema = paths.schema
        del id_schema["path"]

    for *idvals, path in paths.iter_rows():
        id_exprs = [
            pl.lit(val).cast(dtype).alias(name)
            for val, (name, dtype) in zip(idvals, id_schema.items())
        ]

        df = (
            pl.read_csv(path, new_columns=column_names, **kwargs)
            .with_columns(*id_exprs, index, *lockin_exprs)
            .select(
                *(pl.col(name) for _, (name, _) in zip(idvals, id_schema.items())),
                pl.col(index),
                *(pl.col(name) for name in lockin_schema.keys()),
            )
            .sort(index, *id_schema.keys())
        )
        frames.append(df)
    return Dataset(frames, index, list(id_schema))


def zero_quadrature(s: pl.Series):
    real, imag = s.struct.unnest().get_columns()
    pha = np.arange(-1.571, 1.571, 1e-3)
    imag_outer = np.outer(real, np.sin(pha)) + np.outer(imag, np.cos(pha))
    pha_opt = pha[np.sum(np.abs(imag_outer) ** 2, axis=0).argmin()]
    return real * np.cos(pha_opt) - imag * np.sin(pha_opt)

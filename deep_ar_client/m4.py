"""Utility for loading M4 competition CSV files (Hourly/Daily/Monthly etc.)."""

from pathlib import Path
import pandas as pd
import numpy as np

def load_m4_csv(csv_path: str, min_length: int = 100):
    """Return list[np.ndarray] where each array is one time‑series.

    Parameters
    ----------
    csv_path : str
        Path to an M4 *train* CSV such as ``Hourly-train.csv``.
    min_length : int
        Filter out very short series (rare in M4 but useful for testing).

    Notes
    -----
    Each CSV row is: id, V1, V2, …, Vn
    with variable‑length rows padded by blank cells.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    series_list = []
    for _, row in df.iterrows():
        values = row.iloc[1:].dropna().astype(np.float32).values
        if len(values) >= min_length:
            series_list.append(values)
    return series_list

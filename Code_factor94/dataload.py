"""Quick loader and descriptive stats for datashare_OLD.zip (GKX_20201231.csv)."""
from __future__ import annotations
import zipfile
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd

DATA_ZIP = Path(r"D:\GNN in asset pricing\dataset\Autoencoder_data\datashare.zip")
CSV_MEMBER = "datashare.csv"


def _iter_csv_chunks(*, usecols: List[str] | None = None, chunksize: int = 500_000, **read_csv_kwargs) -> Iterable[pd.DataFrame]:
    """Stream the CSV inside the zipfile in chunks to avoid loading everything at once."""
    if not DATA_ZIP.exists():
        raise FileNotFoundError(f"Zip not found: {DATA_ZIP}")
    with zipfile.ZipFile(DATA_ZIP) as zf:
        if CSV_MEMBER not in zf.namelist():
            raise FileNotFoundError(f"{CSV_MEMBER} not found in {DATA_ZIP}")
        with zf.open(CSV_MEMBER) as handle:
            yield from pd.read_csv(handle, chunksize=chunksize, usecols=usecols, **read_csv_kwargs)


def get_columns() -> List[str]:
    """Return all column names without loading data."""
    with zipfile.ZipFile(DATA_ZIP) as zf:
        with zf.open(CSV_MEMBER) as handle:
            cols = list(pd.read_csv(handle, nrows=0).columns)
    return cols


def get_date_summary() -> Tuple[int, pd.Timestamp | None, pd.Timestamp | None]:
    """Count rows and compute min/max DATE using chunked scanning."""
    total = 0
    min_date = None
    max_date = None
    for chunk in _iter_csv_chunks(usecols=["DATE"], chunksize=500_000):
        dates = pd.to_datetime(chunk["DATE"], format="%Y%m%d", errors="coerce")
        total += len(dates)
        if dates.notna().any():
            dmin = dates.min()
            dmax = dates.max()
            min_date = dmin if min_date is None or (dmin is not pd.NaT and dmin < min_date) else min_date
            max_date = dmax if max_date is None or (dmax is not pd.NaT and dmax > max_date) else max_date
    return total, min_date, max_date


def get_head(n: int = 5) -> pd.DataFrame:
    """Read the first n rows for a quick peek."""
    with zipfile.ZipFile(DATA_ZIP) as zf:
        with zf.open(CSV_MEMBER) as handle:
            return pd.read_csv(handle, nrows=n)


def sample_describe(nrows: int = 20_000) -> pd.DataFrame:
    """Compute descriptive stats on a sample of rows (approximation, faster than full dataset)."""
    with zipfile.ZipFile(DATA_ZIP) as zf:
        with zf.open(CSV_MEMBER) as handle:
            sample = pd.read_csv(handle, nrows=nrows)
    desc = sample.describe(include="all", percentiles=[0.25, 0.5, 0.75]).transpose()
    return desc


def main() -> None:
    print(f"Zip path: {DATA_ZIP}")
    print(f"CSV member: {CSV_MEMBER}")

    columns = get_columns()
    print(f"Column count: {len(columns)}")
    print("Columns:")
    print(columns)

    total_rows, min_date, max_date = get_date_summary()
    print(f"\nRow count (scanned): {total_rows:,}")
    print(f"DATE range: {min_date.date() if min_date is not None else 'N/A'} -> {max_date.date() if max_date is not None else 'N/A'}")

    head = get_head()
    print("\nHead (first 5 rows):")
    print(head)

    stats = sample_describe()
    print("\nDescriptive stats on first 20k rows (approximate):")
    print(stats)


if __name__ == "__main__":
    main()

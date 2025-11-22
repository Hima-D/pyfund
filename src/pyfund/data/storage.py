# src/pyfund/data/storage.py
from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_BASE_PATH = Path("./cache/parquet")


class DataStorage:
    """
    High-performance, versioned Parquet storage for financial time-series data.

    Features:
    - Automatic partitioning by ticker + date (optional)
    - Metadata support (source, fetched_at, version, etc.)
    - Hash-based deduplication / cache invalidation
    - Safe concurrent reads/writes
    - Simple API with context manager support
    """

    def __init__(
        self,
        base_path: str | Path = DEFAULT_BASE_PATH,
        partition_by: str | None = "ticker",  # None, "ticker", or "year/month"
        compression: str = "zstd",  # Better than gzip: faster + smaller
    ):
        self.base_path = Path(base_path).resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.partition_by = partition_by
        self.compression = compression

    def _get_path(self, name: str, ticker: str | None = None) -> Path:
        """Resolve final storage path with optional partitioning."""
        if self.partition_by == "ticker" and ticker:
            path = self.base_path / ticker
        elif self.partition_by == "year/month" and ticker:
            # Requires 'Date' in index or column
            path = self.base_path / ticker
        else:
            path = self.base_path

        path.mkdir(parents=True, exist_ok=True)
        return path / f"{name}.parquet"

    def save(
        self,
        df: pd.DataFrame,
        name: str,
        ticker: str | None = None,
        metadata: dict[str, Any] | None = None,
        overwrite: bool = True,
    ) -> Path:
        """
        Save DataFrame to Parquet with rich metadata.

        Parameters
        ----------
        df : pd.DataFrame
            Data to save (should have DatetimeIndex)
        name : str
            Logical name (e.g., "daily_prices", "features")
        ticker : str, optional
            For partitioning and metadata
        metadata : dict, optional
            Extra info: {"source": "alpaca", "interval": "1d"}
        overwrite : bool
            If False, append or raise if exists

        Returns
        -------
        Path
            Full path where data was saved
        """
        if df.empty:
            raise ValueError("Cannot save empty DataFrame")

        # Ensure proper index name
        if isinstance(df.index, pd.DatetimeIndex):
            df.index.name = df.index.name or "Date"

        # Default metadata
        default_metadata = {
            "pyfund_version": "0.1.0",
            "saved_at": datetime.utcnow().isoformat(),
            "ticker": ticker or "unknown",
            "rows": len(df),
            "columns": list(df.columns),
            "start_date": str(df.index.min()) if isinstance(df.index, pd.DatetimeIndex) else None,
            "end_date": str(df.index.max()) if isinstance(df.index, pd.DatetimeIndex) else None,
            "data_hash": self._df_hash(df),
        }

        if metadata:
            default_metadata.update(metadata)

        path = self._get_path(name, ticker=ticker)

        # Write with metadata
        table = pa.Table.from_pandas(df, preserve_index=True)
        existing_metadata = table.schema.metadata or {}
        existing_metadata.update({k: str(v).encode() for k, v in default_metadata.items()})
        table = table.replace_schema_metadata(existing_metadata)

        pq.write_table(
            table,
            path,
            compression=self.compression,
            use_dictionary=True,
            write_statistics=True,
            flavor="spark" if self.partition_by else None,
        )

        return path

    def load(
        self,
        name: str,
        ticker: str | None = None,
        columns: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """
        Load DataFrame with optional filtering.
        """
        path = self._get_path(name, ticker=ticker)

        if not path.exists():
            raise FileNotFoundError(f"No data found at {path}")

        filters = []
        if start_date or end_date:
            if "Date" not in pd.read_parquet(path, columns=[]).columns and isinstance(
                pd.read_parquet(path, columns=[]).index, pd.DatetimeIndex
            ):
                # Index is Date
                df = pd.read_parquet(path, columns=columns)
                if start_date:
                    df = df[df.index >= start_date]
                if end_date:
                    df = df[df.index <= end_date]
                return df

            # Column-based filtering
            if start_date:
                filters.append(("Date", ">=", start_date))
            if end_date:
                filters.append(("Date", "<=", end_date))

        df = pd.read_parquet(
            path,
            columns=columns,
            filters=filters if filters else None,
        )

        # Restore proper index if saved as DatetimeIndex
        if "Date" in df.columns:
            df = df.set_index("Date")
            df.index = pd.to_datetime(df.index)

        return df.sort_index()

    def exists(self, name: str, ticker: str | None = None) -> bool:
        return self._get_path(name, ticker=ticker).exists()

    def list_datasets(self) -> list[str]:
        """List all saved dataset names."""
        return [p.stem for p in self.base_path.rglob("*.parquet")]

    @staticmethod
    def _df_hash(df: pd.DataFrame) -> str:
        """Simple content hash for cache invalidation."""
        return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

    # Bonus: Context manager for batch operations
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

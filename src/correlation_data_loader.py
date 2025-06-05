"""
correlation_data_loader.py – Task 3 Data Loader (Aligned Sentiment + Price)
--------------------------------------------------------------------------

Modular loader for pre-aligned sentiment and stock price datasets used in correlation analysis.

Features:
- Loads enriched_full_df and enriched_aligned_df from CSV
- Validates presence of key columns ('ticker', 'cleaned_date', etc.)
- Converts 'cleaned_date' to datetime with coercion fallback
- Optional diagnostics for shape, date range, and null stats

Author: Nabil Mohamed
"""

import os
import pandas as pd
from pandas.errors import EmptyDataError, ParserError

# ----------------------------------------------------------------------
# 📥 CorrelationDataLoader – Defensive, verbose, and reusable
# ----------------------------------------------------------------------


class CorrelationDataLoader:
    """
    Modular loader for aligned sentiment–price datasets.

    Attributes:
    -----------
    full_path : str
        Path to enriched_full_df CSV file.
    aligned_path : str
        Path to enriched_aligned_df CSV file.
    verbose : bool
        If True, print diagnostics during loading.
    """

    def __init__(self, full_path: str, aligned_path: str, verbose: bool = True):
        self.full_path = full_path
        self.aligned_path = aligned_path
        self.verbose = verbose

    def load_full_df(self) -> pd.DataFrame:
        """
        Loads the enriched full sentiment–event–headline dataset.

        Returns:
        --------
        pd.DataFrame : Full dataset with news sentiment and stock context.
        """
        return self._load_and_validate(self.full_path, "enriched_full_df")

    def load_aligned_df(self) -> pd.DataFrame:
        """
        Loads the sentiment–price aligned dataset (main for correlation).

        Returns:
        --------
        pd.DataFrame : Aligned dataset ready for correlation analysis.
        """
        return self._load_and_validate(self.aligned_path, "enriched_aligned_df")

    def _load_and_validate(self, path: str, label: str) -> pd.DataFrame:
        """
        Internal loader that reads, validates, and logs CSV contents.

        Parameters:
        -----------
        path : str
            File path to load.
        label : str
            Diagnostic label for display.

        Returns:
        --------
        pd.DataFrame : Validated DataFrame.

        Raises:
        -------
        FileNotFoundError : If path does not exist.
        ValueError : If file is empty or parsing fails.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ File not found: {path}")

        try:
            df = pd.read_csv(path)
        except (EmptyDataError, ParserError) as e:
            raise ValueError(f"🚫 Could not parse {label}: {e}") from e

        if df.empty:
            raise ValueError(f"🚫 File {label} is empty: {path}")

        # Coerce 'cleaned_date' into datetime format
        if "cleaned_date" in df.columns:
            df["cleaned_date"] = pd.to_datetime(df["cleaned_date"], errors="coerce")
            df = df.dropna(subset=["cleaned_date"])

        if self.verbose:
            self._log_summary(df, label, path)

        return df

    def _log_summary(self, df: pd.DataFrame, label: str, path: str):
        """
        Prints diagnostic info about the DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            Loaded data.
        label : str
            Label for type of data (full or aligned).
        path : str
            Source file path.
        """
        print(f"\n📄 Loaded {label} from {path}")
        print(f"🔢 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(
            f"🗓️ Date range: {df['cleaned_date'].min().date()} → {df['cleaned_date'].max().date()}"
        )
        print(
            f"🧪 Null values:\n{df.isnull().sum().sort_values(ascending=False).head(10)}\n"
        )

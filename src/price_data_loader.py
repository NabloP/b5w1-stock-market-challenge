"""
price_data_loader.py – Stock Price CSV Loader
---------------------------------------------

Modular loader for historical OHLCV stock data from CSV files.

Features:
- Parses standard 'Date' column into datetime format
- Renames to 'cleaned_date' for alignment with sentiment
- Handles encoding fallbacks (utf-8, latin1)
- Logs ticker, date range, shape, and column names
- Includes bulk loader for multiple tickers

Author: Nabil Mohamed
"""

import os
import pandas as pd
from pandas.errors import EmptyDataError, ParserError


# ------------------------------------------------------------------------------
# 📈 PriceDataLoader – Modular, elegant, fault-tolerant
# ------------------------------------------------------------------------------


class PriceDataLoader:
    """
    Elegant loader for OHLCV stock price CSV files.

    Attributes:
    -----------
    folder_path : str
        Path to the folder containing ticker-level CSVs.
    verbose : bool
        Whether to print diagnostic output after loading.
    """

    def __init__(self, folder_path: str, verbose: bool = True):
        self.folder_path = folder_path  # Path to directory of CSVs
        self.verbose = verbose  # Toggle logging
        self.encoding_used = None  # Track last successful encoding

    def load_price_data(self, ticker: str) -> pd.DataFrame:
        """
        Loads OHLCV price data for a given stock ticker.

        Parameters:
        -----------
        ticker : str
            Stock ticker symbol (e.g., 'AAPL', 'TSLA')

        Returns:
        --------
        pd.DataFrame : Parsed DataFrame with 'cleaned_date' and OHLCV columns
        """
        # Construct expected file path
        file_path = os.path.join(self.folder_path, f"{ticker}_historical_data.csv")

        # Raise error if file missing
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ Price file not found: {file_path}")

        # Try UTF-8 first
        try:
            df = self._read_csv(file_path, encoding="utf-8")
            self.encoding_used = "utf-8"
        except UnicodeDecodeError:
            # Fallback to latin1
            df = self._read_csv(file_path, encoding="latin1")
            self.encoding_used = "latin1"
            if self.verbose:
                print(f"⚠️ Encoding issue for {ticker}. Retried with latin1.")

        # Empty check
        if df.empty:
            raise ValueError(f"🚫 Empty price file: {file_path}")

        # Parse 'Date' column and rename to 'cleaned_date'
        df["cleaned_date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["cleaned_date"]).reset_index(drop=True)

        # Diagnostics
        if self.verbose:
            self._print_summary(df, ticker, file_path)

        return df

    def load_all(self) -> pd.DataFrame:
        """
        Loads all ticker CSVs in the folder and returns one combined DataFrame.

        Returns:
        --------
        pd.DataFrame : Combined price data with a 'ticker' column added
        """
        all_dfs = []

        for file in os.listdir(self.folder_path):
            if file.endswith("_historical_data.csv"):
                ticker = file.split("_")[0]
                try:
                    df = self.load_price_data(ticker)
                    df["ticker"] = ticker
                    all_dfs.append(df)
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️ Skipping {ticker} due to error: {e}")

        if not all_dfs:
            raise ValueError("🚫 No price data could be loaded from directory.")

        return pd.concat(all_dfs, ignore_index=True)

    def _read_csv(self, path: str, encoding: str) -> pd.DataFrame:
        """
        Reads a CSV file with the specified encoding.

        Parameters:
        -----------
        path : str
            File path
        encoding : str
            Encoding to use

        Returns:
        --------
        pd.DataFrame : Loaded DataFrame

        Raises:
        -------
        ValueError : If the file cannot be parsed
        """
        try:
            return pd.read_csv(path, encoding=encoding)
        except (EmptyDataError, ParserError) as e:
            raise ValueError(
                f"❌ Failed to parse price file: {path} ({encoding})\n{str(e)}"
            ) from e

    def _print_summary(self, df: pd.DataFrame, ticker: str, path: str):
        """
        Logs a diagnostic summary of the loaded price file.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to summarize
        ticker : str
            Ticker symbol
        path : str
            Source file path
        """
        print(f"\n📄 Loaded: {ticker} | Path: {path}")
        print(f"📦 Encoding used: {self.encoding_used}")
        print(f"🔢 Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")
        print(
            f"🗓️ Date range: {df['cleaned_date'].min().date()} → {df['cleaned_date'].max().date()}"
        )
        print(f"🧪 Columns: {', '.join(df.columns)}\n")

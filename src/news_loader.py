"""
news_loader.py – Financial News Dataset Loader
----------------------------------------------

Specialized CSV loader for financial news articles with:
- Timestamp parsing for the 'date' column
- UTF-8 with fallback to latin1 encoding
- Pretty diagnostic summaries (rows, columns, encoding)

Author: Nabil Mohamed
"""

import os
import pandas as pd
from pandas.errors import EmptyDataError, ParserError


# ------------------------------------------------------------------------------
# 🗞️ NewsDataLoader – Modular, elegant, fault-tolerant
# ------------------------------------------------------------------------------


class NewsDataLoader:
    """
    Elegant CSV loader for financial news headlines.

    Attributes:
    -----------
    path : str
        Full path to the CSV file on disk.
    verbose : bool
        Whether to print diagnostic output after loading.
    parse_dates : list
        List of columns to parse as datetime (defaults to ['date']).
    """

    DEFAULT_DATE_COLUMNS = ["date"]  # Class-level default

    def __init__(
        self, path: str, parse_dates=None, verbose: bool = True
    ):  # define constructor
        self.path = path  # ensure path is a string
        self.verbose = verbose  # ensure verbose is a boolean
        self.parse_dates = (
            parse_dates if parse_dates else self.DEFAULT_DATE_COLUMNS
        )  # default to ['date']
        self.encoding_used = None  # Track which encoding succeeded

    def load(self) -> pd.DataFrame:  # Load the dataset
        """
        Attempts to load the CSV with UTF-8 encoding, then falls back to latin1.

        Returns:
        --------
        pd.DataFrame : Loaded dataset with datetime parsing.

        Raises:
        -------
        FileNotFoundError : If the file does not exist at the specified path.
        ValueError : If the file is empty or fails parsing.
        """

        if not os.path.exists(self.path):  # check if the file exists
            raise FileNotFoundError(
                f"❌ File not found: {self.path}"
            )  # raise an error if not found

        try:  # try to read the CSV with UTF-8 encoding
            df = self._read_csv("utf-8")  # parse the CSV file
            self.encoding_used = "utf-8"  # track the encoding used
        except UnicodeDecodeError:  # handle encoding issues
            df = self._read_csv("latin1")  # fallback to latin1 encoding
            self.encoding_used = "latin1"  # track the encoding used
            if self.verbose:  # print a warning if verbose mode is enabled
                print(
                    f"⚠️ Encoding issue encountered. Retried with latin1."
                )  # print a warning message

        if df.empty:  # check if the DataFrame is empty
            raise ValueError(
                f"🚫 Loaded DataFrame is empty: {self.path}"
            )  # raise an error if empty

        if self.verbose:  # print a summary if verbose mode is enabled
            self._print_summary(df)  # print a summary of the DataFrame

        return df  # return the loaded DataFrame

    def _read_csv(
        self, encoding: str
    ) -> pd.DataFrame:  # read CSV with specific encoding
        """Internal helper to read CSV with specific encoding"""  # this function reads the CSV file with the specified encoding
        try:
            return pd.read_csv(
                self.path, parse_dates=self.parse_dates, encoding=encoding
            )  # parse the CSV file
        except (EmptyDataError, ParserError) as e:  # handle parsing errors
            raise ValueError(
                f"❌ Failed to parse CSV file: {self.path} ({encoding})\n{str(e)}"
            ) from e  # raise an error if parsing fails

    def _print_summary(self, df: pd.DataFrame):  # print summary of DataFrame
        """Prints a summary of the DataFrame after successful load"""
        print(f"\n📄 File loaded: {self.path}")  # print file path
        print(f"📦 Encoding used: {self.encoding_used}")  # prints encoding used
        print(
            f"🔢 Rows: {df.shape[0]:,} | Columns: {df.shape[1]}"
        )  # prints number of rows in dataframe
        print(
            f"🧪 Columns: {', '.join(df.columns)}\n"
        )  # prints number of columns in dataframe

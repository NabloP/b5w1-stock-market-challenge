"""
technical_indicator_calculator.py – Technical Indicator Engine
---------------------------------------------------------------

Adds standard TA indicators (RSI, MACD, SMA, EMA, ATR) to aligned stock–sentiment data.
Ensures ticker-level isolation and datetime order preservation.

Author: Nabil Mohamed
"""

import pandas as pd
import numpy as np

try:
    import talib  # TA-Lib is required for technical indicators
except ImportError as e:
    raise ImportError(
        "❌ TA-Lib is not installed. Please install using `pip install TA-Lib`."
    ) from e


# ------------------------------------------------------------------------------
# 📊 TechnicalIndicatorCalculator – Adds SMA, EMA, RSI, MACD, ATR per Ticker
# ------------------------------------------------------------------------------


class TechnicalIndicatorCalculator:
    """
    Computes technical indicators per stock ticker.

    Attributes:
    -----------
    df : pd.DataFrame
        Input DataFrame with OHLCV and aligned sentiment–return features.
    ticker_col : str
        Column name containing stock ticker symbol.
    date_col : str
        Column name containing datetime.
    verbose : bool
        Whether to print progress for each ticker.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        ticker_col: str = "ticker",
        date_col: str = "cleaned_date",
        verbose: bool = True,
    ):
        self.df = df.copy()
        self.ticker_col = ticker_col
        self.date_col = date_col
        self.verbose = verbose

    def add_indicators(self) -> pd.DataFrame:
        """
        Adds technical indicators to the DataFrame grouped by ticker.

        Returns:
        --------
        pd.DataFrame : Enriched DataFrame with added indicator columns.
        """
        enriched_dfs = []  # Store results per ticker

        # Loop through each ticker separately
        for ticker, group in self.df.groupby(self.ticker_col):
            if self.verbose:
                print(f"📈 Calculating indicators for {ticker}...")

            # Ensure proper sorting and index reset
            group = group.sort_values(by=self.date_col).reset_index(drop=True)

            # Required input arrays for TA-Lib
            close = group["Close"].values.astype(float)
            high = group["High"].values.astype(float)
            low = group["Low"].values.astype(float)
            volume = group["Volume"].values.astype(float)

            # Compute indicators using TA-Lib
            group["SMA_14"] = talib.SMA(close, timeperiod=14)
            group["EMA_14"] = talib.EMA(close, timeperiod=14)
            group["RSI_14"] = talib.RSI(close, timeperiod=14)
            group["MACD"], group["MACD_signal"], _ = talib.MACD(close)
            group["ATR_14"] = talib.ATR(high, low, close, timeperiod=14)

            enriched_dfs.append(group)

        # Combine all ticker-level enriched groups
        final_df = pd.concat(enriched_dfs, ignore_index=True)

        if self.verbose:
            print(f"✅ Technical indicators added for {len(enriched_dfs)} tickers.")

        return final_df

"""
technical_indicator_calculator.py – Technical Indicator & Performance Engine
-------------------------------------------------------------------------

Adds standard technical indicators (RSI, MACD, SMA, EMA, ATR) to a DataFrame
containing OHLCV data, grouped by ticker. Then computes a PyNance-inspired
performance summary (daily returns, annualized return, volatility, Sharpe,
max drawdown) per ticker using industry-standard formulas.

Ensures:
- Deduplication of columns
- Ticker-level isolation
- Chronological ordering by date
- Robust error handling with clear messages

Author: Nabil Mohamed
"""

# -------------------- LIBRARIES --------------------
import pandas as pd  # For handling tabular data structures
import numpy as np  # For numerical calculations, arrays, and statistics
import talib  # TA-Lib provides technical indicators like RSI, SMA, etc.
from tqdm import tqdm  # tqdm adds progress bars to loops

# --------------------------------------------------
# 📊 TechnicalIndicatorCalculator – Adds TA Features
# --------------------------------------------------


class TechnicalIndicatorCalculator:
    """
    Adds technical indicators per ticker (SMA, EMA, RSI, MACD, ATR).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        ticker_col: str = "ticker",
        date_col: str = "cleaned_date",
        verbose: bool = True,
        use_tqdm: bool = True,
    ):
        # Drop duplicate columns, if any
        deduped_df = df.loc[:, ~df.columns.duplicated()].copy()
        self.df = deduped_df  # Store clean DataFrame internally

        # Save config
        self.ticker_col = ticker_col  # Column name for ticker symbols
        self.date_col = date_col  # Column name for datetime
        self.verbose = verbose  # Whether to print diagnostic info
        self.use_tqdm = use_tqdm  # Whether to show progress bar

        # Ensure mandatory columns are present
        required_cols = {
            self.ticker_col,
            self.date_col,
            "High",
            "Low",
            "Close",
            "Volume",
        }
        missing = required_cols - set(self.df.columns)
        if missing:
            raise KeyError(
                f"Input DataFrame is missing required columns: {missing}"
            )  # Raise error if any are missing

        # Convert date column to datetime
        try:
            self.df[self.date_col] = pd.to_datetime(
                self.df[self.date_col], errors="raise"
            )
        except Exception as e:
            raise TypeError(
                f"Could not convert '{self.date_col}' to datetime: {e}"
            ) from e

        # Optional printout for confirmation
        if self.verbose:
            print(
                "✅ TechnicalIndicatorCalculator initialized. Required columns present, no duplicates."
            )

    def add_indicators(self) -> pd.DataFrame:
        """
        Adds TA indicators to self.df grouped by ticker.
        Returns a DataFrame with new columns: SMA_14, EMA_14, RSI_14, MACD, MACD_signal, ATR_14.
        """
        enriched_groups = []  # Will hold each ticker’s enriched DataFrame

        # Get unique tickers
        tickers = self.df[self.ticker_col].unique()

        # Wrap with tqdm progress bar if configured
        if self.use_tqdm:
            tickers = tqdm(
                tickers,
                desc="Calculating TA indicators",
                unit="ticker",
                mininterval=0.5,
            )

        # Loop through each ticker individually
        for ticker in tickers:
            if self.verbose:
                print(f"📈 Processing ticker: {ticker}")

            # Subset data and sort by date
            ticker_df = self.df[self.df[self.ticker_col] == ticker].copy()
            ticker_df = ticker_df.sort_values(by=self.date_col).reset_index(drop=True)

            # Try extracting relevant columns and converting to float
            try:
                high = ticker_df["High"].astype(float).values  # Convert to float array
                low = ticker_df["Low"].astype(float).values
                close = ticker_df["Close"].astype(float).values
                volume = ticker_df["Volume"].astype(float).values
            except Exception as e:
                # Identify which columns failed
                faulty_cols = [
                    col
                    for col in ["High", "Low", "Close", "Volume"]
                    if ticker_df[col].isnull().any()
                    or not np.issubdtype(ticker_df[col].dtype, np.number)
                ]
                raise ValueError(
                    f"Ticker '{ticker}' has non-numeric or NaN values in: {faulty_cols}"
                ) from e

            # Compute TA indicators with TA-Lib
            try:
                ticker_df["SMA_14"] = talib.SMA(
                    close, timeperiod=14
                )  # Simple Moving Average
                ticker_df["EMA_14"] = talib.EMA(
                    close, timeperiod=14
                )  # Exponential Moving Average
                ticker_df["RSI_14"] = talib.RSI(
                    close, timeperiod=14
                )  # Relative Strength Index
                macd, macd_signal, _ = talib.MACD(close)  # MACD and signal line
                ticker_df["MACD"] = macd
                ticker_df["MACD_signal"] = macd_signal
                ticker_df["ATR_14"] = talib.ATR(
                    high, low, close, timeperiod=14
                )  # Average True Range
            except Exception as e:
                raise RuntimeError(f"TA-Lib error for {ticker}: {e}") from e

            enriched_groups.append(ticker_df)  # Append enriched DataFrame

        # Combine all enriched ticker DataFrames
        try:
            final_df = pd.concat(enriched_groups, ignore_index=True)
        except Exception as e:
            raise RuntimeError(
                f"Error concatenating enriched ticker DataFrames: {e}"
            ) from e

        if self.verbose:
            print(
                f"✅ Technical indicators added for {len(enriched_groups)} tickers. Final shape: {final_df.shape}"
            )

        return final_df  # Return final enriched DataFrame


# ------------------------------------------------------------------
# 📈 PyNancePerformanceCalculatorHybrid – Volatility & Sharpe Metrics
# ------------------------------------------------------------------


class PyNancePerformanceCalculatorHybrid:
    """
    Computes per-ticker metrics:
    - Annualized Return
    - Annualized Volatility (std)
    - Sharpe Ratio (risk-free = 0)
    - Max Drawdown
    """

    def __init__(
        self,
        df: pd.DataFrame,
        date_col: str = "cleaned_date",
        returns_col: str = "daily_return",
        verbose: bool = True,
    ):
        # Deduplicate and copy input DataFrame
        self.df = df.loc[:, ~df.columns.duplicated()].copy()
        self.date_col = date_col  # Date column name
        self.returns_col = returns_col  # Return column name
        self.verbose = verbose  # Verbosity flag

        # Validate required columns
        required_cols = {"ticker", date_col, "Close"}
        if missing := required_cols - set(self.df.columns):
            raise KeyError(f"Missing required columns: {missing}")

        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.df[self.date_col]):
            self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])

        # If daily returns missing, compute from Close
        if self.returns_col not in self.df.columns:
            if self.verbose:
                print(
                    f"ℹ️ '{self.returns_col}' not found; computing from 'Close' per ticker."
                )
            self.df.sort_values(by=["ticker", self.date_col], inplace=True)
            self.df[self.returns_col] = (
                self.df.groupby("ticker")["Close"].pct_change().fillna(0)
            )

        if self.verbose:
            print("✅ PyNancePerformanceCalculatorHybrid initialized.")

    def compute_summary(self) -> pd.DataFrame:
        """
        Compute per-ticker summary metrics:
        - Annualized Return
        - Annualized Volatility
        - Sharpe Ratio (assumes r_f = 0)
        - Max Drawdown
        Returns:
            pd.DataFrame with metrics indexed by ticker
        """
        df = self.df.sort_values(["ticker", self.date_col]).reset_index(
            drop=True
        )  # Ensure order
        summaries = []  # Store results

        for ticker, group in df.groupby("ticker"):
            rets = group[self.returns_col].dropna().values  # Extract return series
            n_obs = len(rets)  # Count of observations

            if n_obs < 2:
                # If not enough data, skip this ticker
                summaries.append(
                    {
                        "ticker": ticker,
                        "annualized_return": np.nan,
                        "annualized_volatility": np.nan,
                        "sharpe_ratio": np.nan,
                        "max_drawdown": np.nan,
                    }
                )
                continue

            # a) Annualized geometric return
            total_return = np.prod(1 + rets)  # Compound return
            ann_return = total_return ** (252.0 / n_obs) - 1  # Annualize

            # b) Annualized volatility (sample std × sqrt(252))
            daily_std = np.std(rets, ddof=1)  # Sample standard deviation
            ann_vol = daily_std * np.sqrt(252)  # Annualized

            # c) Sharpe Ratio (r_f = 0)
            sharpe = (
                ann_return / ann_vol if ann_vol != 0 else np.nan
            )  # Avoid div-by-zero

            # d) Max drawdown
            cumulative = np.cumprod(1 + rets)  # Cumulative return path
            peak = np.maximum.accumulate(cumulative)  # Peak value
            drawdowns = (cumulative - peak) / peak  # Drawdown % from peak
            max_dd = drawdowns.min()  # Worst drawdown

            # Append results
            summaries.append(
                {
                    "ticker": ticker,
                    "annualized_return": ann_return,
                    "annualized_volatility": ann_vol,
                    "sharpe_ratio": sharpe,
                    "max_drawdown": max_dd,
                }
            )

        # Convert to DataFrame and return
        summary_df = pd.DataFrame(summaries).set_index("ticker")

        if self.verbose:
            print(
                f"✅ Hybrid performance summary computed for {summary_df.shape[0]} tickers."
            )

        return summary_df

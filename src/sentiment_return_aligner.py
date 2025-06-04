"""
sentiment_return_aligner.py – Sentiment–Price Alignment Module
---------------------------------------------------------------

Aligns sentiment scores from financial news with historical stock price data,
applying exponential decay and computing returns, volume-based signals, and
sentiment-volume divergence. Also provides lagged-feature creation for
sentiment and returns. Outputs a per-ticker, per-date enriched DataFrame
ready for predictive modeling.

Features:
- Validates and renames columns to ensure consistency
- Maps categorical sentiment labels to numeric [-1, 0, 1]
- Applies exponential time decay to collapse multiple headlines into one weighted score
- Merges decayed sentiment with OHLCV price data by ticker and date
- Computes day-over-day returns and forward returns (1, 3, 5 days)
- Calculates volume returns, 7-day volume averages, volume spikes
- Flags both bullish_low_vol and bearish_high_vol divergence cases
- Drops any rows with missing critical data
- Adds lagged features for sentiment and returns (1d, 3d, 5d)
- Returns final enriched DataFrame

Author: Nabil Mohamed
"""

import pandas as pd  # Data manipulation library
import numpy as np  # Numerical computing library
from tqdm import tqdm  # Progress bar utility for loops


class SentimentReturnAligner:
    """
    Aligns enriched sentiment data with historical stock price data and adds lagged features.

    Attributes:
    -----------
    sentiment_df : pd.DataFrame
        DataFrame of news headlines containing at least
        ['ticker', 'cleaned_date', 'ensemble_sentiment'] columns.
    price_df : pd.DataFrame
        DataFrame of OHLCV stock prices containing at least
        ['ticker', 'cleaned_date', 'Close', 'Volume'] columns.
    decay_lambda : float
        Lambda for exponential decay (higher value gives more weight to recent headlines).
    verbose : bool
        If True, prints diagnostic summaries at each step.
    use_tqdm : bool
        If True, displays progress bars for long-running operations.
    """

    def __init__(
        self,
        sentiment_df: pd.DataFrame,
        price_df: pd.DataFrame,
        decay_lambda: float = 1.5,
        verbose: bool = True,
        use_tqdm: bool = True,
    ):
        """
        Initialize the SentimentReturnAligner with DataFrames and parameters.

        Parameters:
        -----------
        sentiment_df : pd.DataFrame
            DataFrame containing raw sentiment data with
            'ticker', 'cleaned_date', 'ensemble_sentiment'.
        price_df : pd.DataFrame
            DataFrame containing price data with
            'ticker', 'cleaned_date', 'Close', 'Volume'.
        decay_lambda : float
            Exponential decay factor for weighting (default 1.5).
        verbose : bool
            If True, prints detailed logs (default True).
        use_tqdm : bool
            If True, shows progress bars for grouping and loops (default True).
        """

        # Copy inputs to avoid mutating original DataFrames
        self.sentiment_df = (
            sentiment_df.copy()
        )  # Duplicate sentiment DataFrame to internal attribute
        # Drop duplicate columns in price_df to ensure 'cleaned_date' is unique
        self.price_df = price_df.loc[
            :, ~price_df.columns.duplicated()
        ].copy()  # Remove any duplicate column names
        self.decay_lambda = decay_lambda  # Store exponential decay factor
        self.verbose = verbose  # Store verbosity flag
        self.use_tqdm = use_tqdm  # Store progress-bar flag

        # --------------------------------------------------------------------------
        # Validate that sentiment_df contains required columns
        # --------------------------------------------------------------------------
        required_sentiment_cols = {
            "ticker",
            "cleaned_date",
            "ensemble_sentiment",
        }  # Required sentiment columns
        missing_sent_cols = required_sentiment_cols - set(
            self.sentiment_df.columns
        )  # Check which are missing
        if missing_sent_cols:  # If there are any missing required sentiment columns
            # Raise error listing missing sentiment columns
            raise KeyError(
                f"Sentiment DataFrame is missing columns: {missing_sent_cols}"
            )

        # --------------------------------------------------------------------------
        # Validate that price_df contains required columns
        # --------------------------------------------------------------------------
        required_price_cols = {
            "ticker",
            "cleaned_date",
            "Close",
            "Volume",
        }  # Required price columns
        missing_price_cols = required_price_cols - set(
            self.price_df.columns
        )  # Check which are missing
        if missing_price_cols:  # If there are any missing required price columns
            # Raise error listing missing price columns
            raise KeyError(f"Price DataFrame is missing columns: {missing_price_cols}")

        # --------------------------------------------------------------------------
        # Convert 'cleaned_date' in sentiment_df to datetime, with error handling
        # --------------------------------------------------------------------------
        try:
            self.sentiment_df["cleaned_date"] = pd.to_datetime(
                self.sentiment_df["cleaned_date"]
            )  # Convert sentiment_df dates to datetime
        except Exception as e:  # If conversion fails
            sample = (
                self.sentiment_df["cleaned_date"].head(3).tolist()
            )  # Get sample values
            # Raise TypeError with sample info
            raise TypeError(
                f"Could not convert sentiment_df['cleaned_date'] to datetime: {e}\n"
                f"Sample values: {sample}"
            )

        # --------------------------------------------------------------------------
        # Convert 'cleaned_date' in price_df to datetime, with error handling
        # --------------------------------------------------------------------------
        try:
            self.price_df["cleaned_date"] = pd.to_datetime(
                self.price_df["cleaned_date"]
            )  # Convert price_df dates to datetime
        except Exception as e:  # If conversion fails
            sample = self.price_df["cleaned_date"].head(3).tolist()  # Get sample values
            # Raise TypeError with sample info
            raise TypeError(
                f"Could not convert price_df['cleaned_date'] to datetime: {e}\n"
                f"Sample values: {sample}"
            )

        # If verbose, print initialization success message
        if self.verbose:
            print(
                "✅ Initialization complete: Required columns validated and dates parsed."
            )

    def _map_sentiment_to_numeric(self):
        """
        Convert 'ensemble_sentiment' categorical labels to numeric scale [-1, 0, 1].
        Raises ValueError if any unexpected labels are found.
        """
        mapping = {
            "bullish": 1.0,
            "neutral": 0.0,
            "bearish": -1.0,
        }  # Define sentiment-to-number mapping

        # Map categorical sentiment to numeric scores
        self.sentiment_df["sentiment_score"] = self.sentiment_df[
            "ensemble_sentiment"
        ].map(
            mapping
        )  # Apply mapping

        # Identify any unmapped labels that resulted in NaN values
        unmapped = self.sentiment_df.loc[
            self.sentiment_df["sentiment_score"].isna(), "ensemble_sentiment"
        ].unique()  # Unique unmapped sentiment labels
        if unmapped.size > 0:  # If any unmapped labels exist
            # Raise error listing unexpected labels
            raise ValueError(
                f"Unexpected sentiment labels encountered: {list(unmapped)}. "
                "Allowed values are: ['bullish', 'neutral', 'bearish']"
            )

        # If verbose, confirm mapping succeeded
        if self.verbose:
            print("✅ Sentiment labels converted to numeric scale [-1, 0, 1].")

    def _apply_exponential_decay(self) -> pd.DataFrame:
        """
        Group by (ticker, cleaned_date) and apply exponential decay to sentiment scores.
        Returns a DataFrame with columns: ['ticker', 'cleaned_date', 'weighted_sentiment'].
        """
        # Sort sentiment_df by ticker and date for consistent grouping
        self.sentiment_df = self.sentiment_df.sort_values(
            by=["ticker", "cleaned_date"]
        )  # Sort in-place for grouping

        # Determine grouping iterator: with or without progress bar
        groups = self.sentiment_df.groupby(
            ["ticker", "cleaned_date"]
        )  # Group by ticker and date
        if self.use_tqdm:  # If progress bars are enabled
            groups = tqdm(
                groups,
                desc="Applying sentiment decay",
                unit="group",
                mininterval=1,
            )  # Wrap groupby with tqdm for progress bar

        results = []  # Initialize list to collect weighted results for each group
        for (ticker, date), group in groups:  # Iterate over each (ticker, date) group
            scores = group[
                "sentiment_score"
            ].values  # Array of numeric sentiment scores
            dates = group["cleaned_date"].values.astype(
                "datetime64[D]"
            )  # Convert dates to numpy datetime64[D]
            ref_date = dates.max()  # Reference date (should equal group date)
            days_diff = (ref_date - dates).astype(int)  # Compute days difference array
            weights = np.exp(
                -self.decay_lambda * days_diff
            )  # Compute exponential decay weights
            weighted_score = np.average(
                scores, weights=weights
            )  # Weighted average of scores
            # Append weighted result for this group
            results.append(
                {
                    "ticker": ticker,  # Ticker symbol
                    "cleaned_date": date,  # Date of sentiment
                    "weighted_sentiment": weighted_score,  # Weighted sentiment score
                }
            )

        # Create DataFrame from weighted results
        decayed_df = pd.DataFrame(
            results
        )  # DataFrame of weighted sentiment per ticker-date

        # If verbose, report number of rows processed
        if self.verbose:
            print(
                f"📉 Exponential decay applied (λ={self.decay_lambda}) → "
                f"{len(decayed_df):,} daily sentiment rows"
            )

        return decayed_df  # Return decayed sentiment DataFrame

    def _compute_returns_and_volume_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Given a merged DataFrame with 'Close', 'Volume', and 'weighted_sentiment',
        compute:
        - return_t: day-over-day percentage return
        - forward_return_1d, forward_return_3d, forward_return_5d
        - volume_return_t: day-over-day volume change
        - volume_7d_avg: 7-day rolling average volume
        - volume_spike: binary flag if Volume > 1.5 * volume_7d_avg
        - sentiment_volume_divergence: 'bullish_low_vol', 'bearish_high_vol', or 'aligned'
        """
        # Sort DataFrame by ticker and date for time-series operations
        df = df.sort_values(
            by=["ticker", "cleaned_date"]
        ).copy()  # Copy to avoid in-place modification

        tickers = df["ticker"].unique()  # Get unique ticker symbols
        if self.use_tqdm:  # If progress bars are enabled
            tickers = tqdm(
                tickers,
                desc="Processing tickers",
                unit="ticker",
                mininterval=0.5,
            )  # Wrap tickers list with tqdm

        enriched_list = (
            []
        )  # Initialize list to collect each ticker's enriched DataFrame
        for ticker in tickers:  # Iterate through each ticker
            ticker_df = df[
                df["ticker"] == ticker
            ].copy()  # Subset DataFrame for one ticker

            # -------------------------------
            # 1. Day-over-day returns
            # -------------------------------
            ticker_df["return_t"] = ticker_df[
                "Close"
            ].pct_change()  # Compute daily % return

            # -------------------------------
            # 2. Forward returns (1d, 3d, 5d)
            # -------------------------------
            ticker_df["forward_return_1d"] = (
                ticker_df["Close"].shift(-1) / ticker_df["Close"] - 1
            )  # 1-day forward return
            ticker_df["forward_return_3d"] = (
                ticker_df["Close"].shift(-3) / ticker_df["Close"] - 1
            )  # 3-day forward return
            ticker_df["forward_return_5d"] = (
                ticker_df["Close"].shift(-5) / ticker_df["Close"] - 1
            )  # 5-day forward return

            # -------------------------------
            # 3. Volume returns & rolling average
            # -------------------------------
            ticker_df["volume_return_t"] = ticker_df[
                "Volume"
            ].pct_change()  # Compute daily volume % change
            ticker_df["volume_7d_avg"] = (
                ticker_df["Volume"].rolling(7, min_periods=1).mean()
            )  # Compute 7-day rolling average volume

            # -------------------------------
            # 4. Volume spike flag
            # -------------------------------
            ticker_df["volume_spike"] = (
                ticker_df["Volume"] > 1.5 * ticker_df["volume_7d_avg"]
            ).astype(
                int
            )  # Flag = 1 if today's volume > 1.5 * 7-day avg

            # -------------------------------
            # 5. Sentiment-volume divergence
            # -------------------------------
            conditions = [
                (ticker_df["weighted_sentiment"] > 0)
                & (ticker_df["volume_return_t"] < 0),
                (ticker_df["weighted_sentiment"] < 0)
                & (ticker_df["volume_return_t"] > 0),
            ]  # Define conditions for divergence
            choices = [
                "bullish_low_vol",
                "bearish_high_vol",
            ]  # Labels for each condition
            ticker_df["sentiment_volume_divergence"] = np.select(
                conditions, choices, default="aligned"
            )  # Apply np.select to assign divergence label

            enriched_list.append(ticker_df)  # Collect this ticker's enriched DataFrame

        # Concatenate all ticker-level DataFrames into a single enriched DataFrame
        enriched = pd.concat(
            enriched_list, ignore_index=True
        )  # Merge into one DataFrame

        # If verbose, print summary of enriched signals
        if self.verbose:
            print("📈 Computed returns, volume-based signals, and divergence flags.")

        return enriched  # Return the enriched DataFrame

    def align(self) -> pd.DataFrame:
        """
        Main alignment pipeline:
        1. Map sentiment labels → numeric
        2. Apply exponential decay per (ticker, date)
        3. Merge decayed sentiment with price_df on ['ticker','cleaned_date']
        4. Compute returns & volume signals
        5. Drop rows with missing critical values
        6. Return intermediate enriched DataFrame (before lagging)
        """
        # -------------------------------
        # 1. Convert sentiment labels to numeric
        # -------------------------------
        self._map_sentiment_to_numeric()  # Map and validate sentiment labels

        # -------------------------------
        # 2. Apply exponential decay
        # -------------------------------
        decayed_df = self._apply_exponential_decay()  # Get weighted sentiment DataFrame

        # -------------------------------
        # 3. Merge price_df with decayed sentiment
        # -------------------------------
        for key in ["ticker", "cleaned_date"]:  # Check required merge keys
            if key not in self.price_df.columns:  # If price_df missing key
                raise KeyError(f"price_df missing merge key: '{key}'")
            if key not in decayed_df.columns:  # If decayed_df missing key
                raise KeyError(f"decayed_df missing merge key: '{key}'")

        merged = pd.merge(
            self.price_df,
            decayed_df,
            on=["ticker", "cleaned_date"],
            how="inner",  # Keep only matching rows
        )  # Perform inner merge

        if self.verbose:
            print(
                f"🔗 Merged price+sentiment → {merged.shape[0]:,} rows, {merged.shape[1]} columns"
            )

        # -------------------------------
        # 4. Compute returns & volume signals
        # -------------------------------
        enriched = self._compute_returns_and_volume_signals(merged)  # Compute signals

        # -------------------------------
        # 5. Drop rows with missing critical values
        # -------------------------------
        pre_drop = enriched.shape[0]  # Number of rows before dropping
        enriched = enriched.dropna(
            subset=["weighted_sentiment", "return_t", "forward_return_1d"]
        ).reset_index(
            drop=True
        )  # Drop rows missing key columns
        post_drop = enriched.shape[0]  # Number of rows after dropping

        if self.verbose:
            print(f"🗑 Dropped {pre_drop - post_drop:,} rows with missing key values")
            print(
                f"🧩 Intermediate enriched dataset ready: {enriched.shape[0]:,} rows, {enriched.shape[1]} columns"
            )

        return enriched  # Return the intermediate enriched DataFrame

    def add_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add lagged features for sentiment and returns to the enriched DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame returned by align(), containing:
            ['ticker', 'cleaned_date', ..., 'weighted_sentiment', 'return_t', etc.]

        Returns:
        --------
        pd.DataFrame
            DataFrame with additional columns:
            ['sentiment_lag_1d', 'sentiment_lag_3d', 'sentiment_lag_5d',
             'return_lag_1d', 'return_lag_3d'] and any other requested lags.
        """
        # Sort by ticker and date to ensure correct chronological order for shift operations
        df = df.sort_values(["ticker", "cleaned_date"]).reset_index(drop=True)

        # Lagged weighted sentiment (1, 3, 5 days)
        df["sentiment_lag_1d"] = df.groupby("ticker")["weighted_sentiment"].shift(
            1
        )  # 1-day lag
        df["sentiment_lag_3d"] = df.groupby("ticker")["weighted_sentiment"].shift(
            3
        )  # 3-day lag
        df["sentiment_lag_5d"] = df.groupby("ticker")["weighted_sentiment"].shift(
            5
        )  # 5-day lag

        # Lagged day-over-day return
        df["return_lag_1d"] = df.groupby("ticker")["return_t"].shift(
            1
        )  # 1-day return lag
        df["return_lag_3d"] = df.groupby("ticker")["return_t"].shift(
            3
        )  # 3-day return lag

        if self.verbose:
            print(
                "📊 Added lagged features: sentiment_lag_(1,3,5)d and return_lag_(1,3)d."
            )

        return df  # Return DataFrame with added lagged features

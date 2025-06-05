"""
correlation_sentiment_aggregator.py – Sentiment Aggregation Engine
-------------------------------------------------------------------

Aggregates daily sentiment signals from news headlines for use in correlation
analysis with stock price movements. Designed for Task 3 of B5W1.

Features:
- Groups news sentiment by cleaned_date and ticker
- Supports multiple aggregation strategies: mean, median, exponential weighted
- Optionally propagates volume–sentiment divergence signals
- Preserves all numeric return and volume features per ticker-date
- Retains raw `weighted_sentiment` for direct correlation testing
- Validates input schema and handles missing data gracefully

Author: Nabil Mohamed
"""

import pandas as pd  # For DataFrame operations
import numpy as np  # For numeric operations
from typing import Optional  # For type hinting


# ------------------------------------------------------------------
# 🧠 CorrelationSentimentAggregator – Aggregates Daily Sentiment
# ------------------------------------------------------------------


class CorrelationSentimentAggregator:
    """
    Aggregates sentiment features per ticker per date for correlation tasks.

    Attributes:
    -----------
    df : pd.DataFrame
        DataFrame containing at least 'ticker', 'cleaned_date', and sentiment columns.
    verbose : bool
        If True, prints diagnostic messages during processing.
    """

    def __init__(self, df: pd.DataFrame, verbose: bool = True):
        self.df = df.copy()  # Preserve original input
        self.verbose = verbose  # Enable verbose mode

        # Validate required columns
        required_cols = {"ticker", "cleaned_date", "weighted_sentiment"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Input DataFrame is missing required columns: {missing}")

        # Ensure 'cleaned_date' is datetime
        self.df["cleaned_date"] = pd.to_datetime(
            self.df["cleaned_date"], errors="coerce"
        )
        self.df.dropna(subset=["cleaned_date"], inplace=True)

        if self.verbose:
            print("✅ CorrelationSentimentAggregator initialized successfully.")

    def aggregate_daily_sentiment(
        self, method: str = "mean", decay_factor: float = 0.5
    ) -> pd.DataFrame:
        """
        Aggregates sentiment per ticker-date using a specified method.

        Parameters:
        -----------
        method : str
            Aggregation method: "mean", "median", or "ewm" (exponential weighted mean).
        decay_factor : float
            Alpha decay rate for 'ewm'. Smaller = faster decay.

        Returns:
        --------
        pd.DataFrame
            Aggregated sentiment and feature-rich frame per ticker and cleaned_date.
        """
        if method not in {"mean", "median", "ewm"}:
            raise ValueError(
                f"Unsupported method '{method}'. Use 'mean', 'median', or 'ewm'."
            )

        if self.verbose:
            print(f"🔄 Aggregating sentiment using method: {method}")

        # Pre-sort for consistent EWM behavior
        self.df.sort_values(by=["ticker", "cleaned_date"], inplace=True)

        # Set base columns to keep (rich numeric features)
        base_cols = [
            "ticker",
            "cleaned_date",
            "return_t",
            "forward_return_1d",
            "forward_return_3d",
            "forward_return_5d",
            "return_lag_1d",
            "return_lag_3d",
            "volume_return_t",
            "volume_7d_avg",
            "volume_spike",
        ]

        # Extract numeric features per ticker-date (one row per pair)
        rich_numeric_df = self.df[
            ["ticker", "cleaned_date"]
            + [
                col
                for col in self.df.columns
                if col in base_cols and col not in {"ticker", "cleaned_date"}
            ]
        ].drop_duplicates(subset=["ticker", "cleaned_date"])

        # Aggregate the sentiment using the specified method
        if method == "mean":
            sentiment_agg = (
                self.df.groupby(["ticker", "cleaned_date"])["weighted_sentiment"]
                .mean()
                .reset_index()
                .rename(columns={"weighted_sentiment": f"agg_sentiment_{method}"})
            )
        elif method == "median":
            sentiment_agg = (
                self.df.groupby(["ticker", "cleaned_date"])["weighted_sentiment"]
                .median()
                .reset_index()
                .rename(columns={"weighted_sentiment": f"agg_sentiment_{method}"})
            )
        elif method == "ewm":
            sentiment_agg = (
                self.df.groupby("ticker", group_keys=False)
                .apply(
                    lambda g: g.set_index("cleaned_date")["weighted_sentiment"]
                    .ewm(alpha=decay_factor)
                    .mean()
                    .reset_index()
                    .assign(ticker=g["ticker"].iloc[0])
                )
                .reset_index(drop=True)
                .rename(columns={"weighted_sentiment": f"agg_sentiment_{method}"})
            )

        # Optional: Add volume–sentiment divergence signals
        if "sentiment_volume_divergence" in self.df.columns:
            div_flag_df = (
                self.df.groupby(["ticker", "cleaned_date"])[
                    "sentiment_volume_divergence"
                ]
                .max()
                .reset_index()
                .rename(columns={"sentiment_volume_divergence": "divergence_flag"})
            )
            sentiment_agg = pd.merge(
                sentiment_agg, div_flag_df, on=["ticker", "cleaned_date"], how="left"
            )

        # Preserve raw weighted_sentiment (1 row per ticker/date)
        sentiment_raw = self.df[
            ["ticker", "cleaned_date", "weighted_sentiment"]
        ].drop_duplicates(subset=["ticker", "cleaned_date"])

        # Merge all into final frame
        enriched_df = sentiment_agg.merge(
            sentiment_raw, on=["ticker", "cleaned_date"], how="left"
        ).merge(rich_numeric_df, on=["ticker", "cleaned_date"], how="left")

        if self.verbose:
            print(f"📌 Columns in enriched_df: {enriched_df.columns.tolist()}")
            print(f"✅ Aggregated sentiment shape: {enriched_df.shape}")

        return enriched_df

"""
sentiment_return_aligner.py – Sentiment–Price Alignment Module
---------------------------------------------------------------

Aligns sentiment scores from financial news with historical stock price data,
applying exponential decay and computing returns for downstream modeling.

Features:
- Maps categorical sentiment labels to numerical range [-1, 1]
- Applies exponential time decay to prioritize recent news
- Aligns sentiment with OHLCV price data by ticker and date
- Computes daily returns and forward returns (t+1)
- Adds volume-based signals: volume returns, spikes, sentiment-volume divergence
- Outputs a merged dataset ready for predictive modeling

Author: Nabil Mohamed
"""

import pandas as pd
import numpy as np


class SentimentReturnAligner:
    """
    Class to align enriched sentiment data with historical stock price data.

    Attributes:
    -----------
    sentiment_df : pd.DataFrame
        News headlines with 'cleaned_date', 'stock'/'ticker', and 'ensemble_sentiment'
    price_df : pd.DataFrame
        OHLCV stock price data with 'cleaned_date', 'ticker', 'Close', 'Volume'
    decay_lambda : float
        Lambda for exponential decay (higher = faster decay)
    verbose : bool
        Whether to print diagnostic summaries
    """

    def __init__(
        self,
        sentiment_df: pd.DataFrame,
        price_df: pd.DataFrame,
        decay_lambda: float = 1.5,
        verbose: bool = True,
    ):
        # Normalize ticker naming
        self.sentiment_df = sentiment_df.copy().rename(columns={"stock": "ticker"})
        self.price_df = price_df.copy()
        self.decay_lambda = decay_lambda
        self.verbose = verbose

    def _map_sentiment_to_numeric(self):
        mapping = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}
        self.sentiment_df["sentiment_score"] = self.sentiment_df[
            "ensemble_sentiment"
        ].map(mapping)
        if self.verbose:
            print("✅ Sentiment labels converted to numeric scale [-1, 0, 1]")

    def _apply_exponential_decay(self):
        self.sentiment_df["cleaned_date"] = pd.to_datetime(
            self.sentiment_df["cleaned_date"]
        )
        self.sentiment_df = self.sentiment_df.sort_values(by=["ticker", "cleaned_date"])

        def decay_group(group):
            scores = group["sentiment_score"].values
            dates = pd.to_datetime(group["cleaned_date"]).values.astype("datetime64[D]")
            ref_date = dates.max()
            days_diff = (ref_date - dates).astype(int)
            weights = np.exp(-self.decay_lambda * days_diff)
            weighted_score = np.average(scores, weights=weights)
            return pd.Series({"weighted_sentiment": weighted_score, "date": ref_date})

        decayed_df = (
            self.sentiment_df.groupby(["ticker", "cleaned_date"])
            .apply(decay_group)
            .reset_index(drop=True)
        )

        if self.verbose:
            print(f"📉 Exponential decay applied with λ = {self.decay_lambda}")

        return decayed_df

    def _compute_returns_and_volume_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(by=["ticker", "cleaned_date"])
        df["return_t"] = df.groupby("ticker")["Close"].pct_change()
        df["return_t+1"] = df.groupby("ticker")["Close"].pct_change().shift(-1)

        df["volume_return_t"] = df.groupby("ticker")["Volume"].pct_change()
        df["volume_7d_avg"] = df.groupby("ticker")["Volume"].transform(
            lambda x: x.rolling(7).mean()
        )
        df["volume_spike"] = (df["Volume"] > 1.5 * df["volume_7d_avg"]).astype(int)
        df["sentiment_volume_divergence"] = np.sign(df["weighted_sentiment"]) * (
            df["volume_return_t"] < 0
        ).astype(int)

        if self.verbose:
            print("📈 Returns and 📊 volume signals computed.")

        return df

    def align(self) -> pd.DataFrame:
        self._map_sentiment_to_numeric()
        decayed_sentiment = self._apply_exponential_decay()

        price_plus = pd.merge(
            self.price_df,
            decayed_sentiment.rename(columns={"date": "cleaned_date"}),
            on=["ticker", "cleaned_date"],
            how="inner",
        )

        enriched = self._compute_returns_and_volume_signals(price_plus)

        enriched = enriched.dropna(
            subset=["weighted_sentiment", "return_t", "return_t+1"]
        ).reset_index(drop=True)

        if self.verbose:
            print(
                f"🧩 Final enriched dataset: {enriched.shape[0]:,} rows, {enriched.shape[1]} columns"
            )

        return enriched

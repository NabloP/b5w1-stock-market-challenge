"""
text_stock_eda.py – Stock-Level Headline Sentiment & Volume Diagnostics
----------------------------------------------------------------------

This module provides tools for analyzing sentiment and event headline trends
at the stock level, using cleaned news data from the B5W1 project.

Features:
- Summarize headline frequency and sentiment category distribution per stock
- Merge sentiment and event data per stock
- Visualize news flow and sentiment category shifts over time
- Detect headline volume spikes and sentiment volatility by ticker
- Modular class-based design with diagnostics and visual output

Author: Nabil Mohamed
"""

from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class StockHeadlineProfiler:
    """
    Analyzes headline sentiment and frequency patterns at the stock level.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        stock_col: str = "stock",
        headline_col: str = "cleaned_headline",
        date_col: str = "cleaned_date",
        sentiment_col: str = "ensemble_sentiment",
        verbose: bool = True,
    ):
        self.df = df.copy()
        self.stock_col = stock_col
        self.headline_col = headline_col
        self.date_col = date_col
        self.sentiment_col = sentiment_col
        self.verbose = verbose

        self._validate_inputs()

    def _validate_inputs(self):
        required_cols = {
            self.stock_col,
            self.headline_col,
            self.date_col,
            self.sentiment_col,
        }
        missing = required_cols - set(self.df.columns)
        if missing:
            raise KeyError(f"Missing required column(s): {missing}")

        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col], errors="coerce")
        if self.verbose:
            print(f"✅ StockHeadlineProfiler initialized with {len(self.df)} rows.")

    def get_headline_volume(self) -> pd.DataFrame:
        counts = (
            self.df.groupby(self.stock_col)[self.headline_col].count().reset_index()
        )
        counts.columns = [self.stock_col, "headline_count"]
        return counts

    def get_sentiment_distribution(self) -> pd.DataFrame:
        sentiment_dist = (
            self.df.groupby([self.stock_col, self.sentiment_col])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        return sentiment_dist

    def join_with_events(self, event_df: pd.DataFrame) -> pd.DataFrame:
        join_col = self.headline_col
        event_col = (
            "cleaned_headline" if "cleaned_headline" in event_df.columns else "headline"
        )
        merged = pd.merge(
            self.df,
            event_df,
            left_on=join_col,
            right_on=event_col,
            how="left",
        )
        return merged

    def plot_sentiment_distribution(self, stock: str):
        subset = self.df[self.df[self.stock_col] == stock]
        sentiment_counts = (
            subset[self.sentiment_col]
            .value_counts()
            .reindex(["bullish", "neutral", "bearish"], fill_value=0)
        )

        sentiment_counts.plot(kind="bar", color=["green", "gray", "red"])
        plt.title(f"Sentiment Distribution – {stock}")
        plt.xlabel("Sentiment")
        plt.ylabel("Headline Count")
        plt.tight_layout()
        plt.show()

    def plot_top_stocks_by_volume(self, n: int = 10):
        volume_df = (
            self.get_headline_volume()
            .sort_values("headline_count", ascending=False)
            .head(n)
        )
        plt.figure(figsize=(10, 5))
        sns.barplot(
            data=volume_df, x="headline_count", y=self.stock_col, palette="Blues_r"
        )
        plt.title(f"Top {n} Stocks by Headline Volume")
        plt.xlabel("Headline Count")
        plt.ylabel("Stock")
        plt.tight_layout()
        plt.show()

    def plot_sentiment_share_by_stock(self, n: int = 10):
        sentiment_df = self.get_sentiment_distribution()
        top_stocks = (
            self.get_headline_volume()
            .sort_values("headline_count", ascending=False)
            .head(n)[self.stock_col]
        )
        sentiment_df = sentiment_df[sentiment_df[self.stock_col].isin(top_stocks)]

        sentiment_df.set_index(self.stock_col).plot(
            kind="barh",
            stacked=True,
            color={"bullish": "green", "neutral": "gray", "bearish": "red"},
        )
        plt.title(f"Sentiment Share – Top {n} Stocks")
        plt.xlabel("Headline Count")
        plt.ylabel("Stock")
        plt.tight_layout()
        plt.show()


class StockVolatilityAnalyzer:
    """
    Detects headline volume spikes and categorical sentiment shifts for each stock.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        stock_col: str = "stock",
        date_col: str = "cleaned_date",
        sentiment_col: str = "ensemble_sentiment",
        verbose: bool = True,
    ):
        self.df = df.copy()
        self.stock_col = stock_col
        self.date_col = date_col
        self.sentiment_col = sentiment_col
        self.verbose = verbose

        self._validate_inputs()

    def _validate_inputs(self):
        required_cols = {self.stock_col, self.date_col, self.sentiment_col}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise KeyError(f"Missing required column(s): {missing}")

        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col], errors="coerce")
        if self.verbose:
            print(f"✅ StockVolatilityAnalyzer initialized with {len(self.df)} rows.")

    def detect_headline_spikes(self) -> pd.DataFrame:
        volume_df = (
            self.df.groupby([self.stock_col, self.date_col])
            .size()
            .reset_index(name="headline_count")
        )

        volume_df.sort_values([self.stock_col, self.date_col], inplace=True)
        volume_df["rolling_avg"] = volume_df.groupby(self.stock_col)[
            "headline_count"
        ].transform(lambda x: x.rolling(3).mean())

        volume_df["spike"] = volume_df["headline_count"] > 2 * volume_df["rolling_avg"]
        return volume_df

    def detect_sentiment_flips(self) -> pd.DataFrame:
        df = self.df.copy()
        df.sort_values([self.stock_col, self.date_col], inplace=True)
        df["prev_sentiment"] = df.groupby(self.stock_col)[self.sentiment_col].shift(1)
        df["flip"] = df[self.sentiment_col] != df["prev_sentiment"]
        df["flip"] = df["flip"].fillna(False)
        return df[df["flip"]]

    def plot_volume_spikes(self, stock: str):
        df = self.detect_headline_spikes()
        subset = df[df[self.stock_col] == stock].copy()
        if subset.empty:
            print(f"⚠️ No volume data for {stock}")
            return

        plt.figure(figsize=(10, 4))
        plt.plot(
            subset[self.date_col],
            subset["headline_count"],
            label="Headline Count",
            marker="o",
        )
        plt.plot(
            subset[self.date_col],
            subset["rolling_avg"],
            label="Rolling Avg",
            linestyle="--",
        )
        spike_dates = subset[subset["spike"]][self.date_col]
        spike_values = subset[subset["spike"]]["headline_count"]
        plt.scatter(spike_dates, spike_values, color="red", label="Spike", zorder=5)
        plt.title(f"Headline Volume & Spikes – {stock}")
        plt.xlabel("Date")
        plt.ylabel("Headline Count")
        plt.legend()
        plt.grid(True, linestyle=":")
        plt.tight_layout()
        plt.show()

    def plot_sentiment_flips(self, stock: str):
        df = self.detect_sentiment_flips()
        subset = df[df[self.stock_col] == stock].copy()
        if subset.empty:
            print(f"⚠️ No sentiment flips detected for {stock}")
            return

        plt.figure(figsize=(10, 4))
        plt.plot(
            subset[self.date_col],
            subset["flip"].astype(int),
            label="Sentiment Flip",
            marker="o",
            linestyle="",
            color="purple",
        )
        plt.title(f"Sentiment Flips Over Time – {stock}")
        plt.xlabel("Date")
        plt.ylabel("Flip Detected")
        plt.tight_layout()
        plt.show()

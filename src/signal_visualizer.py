"""
signal_visualizer.py – Signal Diagnostics Visualization Module
---------------------------------------------------------------

Visualizes relationships between sentiment signals, stock prices, and technical indicators
to support quantitative diagnostics.

Features:
- Time-series plots of price vs. sentiment
- Scatterplots of sentiment vs. return
- Technical indicator overlays (RSI, MACD, SMA, etc.)

Author: Nabil Mohamed
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------------------------------
# 📊 SignalVisualizer – Plots Price, Sentiment, and Technical Indicators
# --------------------------------------------------------------------------


class SignalVisualizer:
    """
    Visualizes sentiment, return, and technical indicator signals for a given stock ticker.

    Attributes:
    -----------
    df : pd.DataFrame
        Merged dataset containing price, sentiment, returns, and indicators.
    date_col : str
        Column name representing the date.
    """

    def __init__(self, df: pd.DataFrame, date_col: str = "cleaned_date"):
        self.df = df.copy()
        self.date_col = date_col

    def plot_sentiment_vs_price(self, ticker: str):
        """
        Plots stock price and sentiment signal over time for a given ticker.
        """
        data = self.df[self.df["ticker"] == ticker]

        fig, ax1 = plt.subplots(figsize=(12, 5))

        # Plot price
        sns.lineplot(
            data=data,
            x=self.date_col,
            y="Close",
            ax=ax1,
            label="Close Price",
            color="blue",
        )
        ax1.set_ylabel("Price", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")

        # Create second axis for sentiment
        ax2 = ax1.twinx()
        sns.lineplot(
            data=data,
            x=self.date_col,
            y="weighted_sentiment",
            ax=ax2,
            label="Sentiment",
            color="red",
        )
        ax2.set_ylabel("Sentiment", color="red")
        ax2.tick_params(axis="y", labelcolor="red")

        plt.title(f"📈 {ticker} – Price vs. Sentiment")
        fig.tight_layout()
        plt.show()

    def plot_sentiment_return_scatter(self, ticker: str):
        """
        Plots scatterplot of sentiment signal vs. forward return.
        """
        data = self.df[self.df["ticker"] == ticker]

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=data,
            x="weighted_sentiment",
            y="return_t+1",
            hue="weighted_sentiment",
            palette="coolwarm",
            alpha=0.6,
        )
        plt.axhline(0, linestyle="--", color="gray")
        plt.axvline(0, linestyle="--", color="gray")
        plt.title(f"🔍 {ticker} – Sentiment vs. Next-Day Return")
        plt.xlabel("Weighted Sentiment")
        plt.ylabel("Return (t+1)")
        plt.tight_layout()
        plt.show()

    def plot_technical_indicators(self, ticker: str):
        """
        Plots SMA, EMA, RSI, MACD indicators for diagnostics.
        """
        data = self.df[self.df["ticker"] == ticker]

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        # Price + SMAs
        sns.lineplot(
            data=data,
            x=self.date_col,
            y="Close",
            ax=axes[0],
            label="Close",
            color="black",
        )
        sns.lineplot(
            data=data,
            x=self.date_col,
            y="SMA_14",
            ax=axes[0],
            label="SMA 14",
            color="blue",
        )
        sns.lineplot(
            data=data,
            x=self.date_col,
            y="EMA_14",
            ax=axes[0],
            label="EMA 14",
            color="orange",
        )
        axes[0].set_title("SMA/EMA")

        # RSI
        sns.lineplot(
            data=data,
            x=self.date_col,
            y="RSI_14",
            ax=axes[1],
            label="RSI 14",
            color="green",
        )
        axes[1].axhline(70, linestyle="--", color="red", label="Overbought")
        axes[1].axhline(30, linestyle="--", color="blue", label="Oversold")
        axes[1].set_title("RSI")

        # MACD
        sns.lineplot(
            data=data,
            x=self.date_col,
            y="MACD",
            ax=axes[2],
            label="MACD",
            color="purple",
        )
        sns.lineplot(
            data=data,
            x=self.date_col,
            y="MACD_signal",
            ax=axes[2],
            label="Signal",
            color="brown",
        )
        axes[2].set_title("MACD")

        plt.suptitle(f"📊 {ticker} – Technical Indicators", fontsize=14)
        plt.tight_layout()
        plt.show()

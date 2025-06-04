"""
signal_visualizer.py – Signal Diagnostics Visualization Module
---------------------------------------------------------------

Visualizes relationships between sentiment signals, stock prices, technical indicators,
and multiple time‐series decompositions for diagnostic insights.

Features:
- Time‐series plots of price vs. sentiment
- Scatterplots of sentiment vs. return
- Technical indicator overlays (RSI, MACD, SMA, etc.)
- Additive and multiplicative seasonal decomposition of price
- STL (Loess‐based) decomposition of price
- HP Filter decomposition into trend and cycle

Author: Nabil Mohamed
"""

import pandas as pd  # Data manipulation
import matplotlib.pyplot as plt  # Plotting
import seaborn as sns  # High‐level plotting
from statsmodels.tsa.seasonal import (
    seasonal_decompose,
)  # Additive/Multiplicative decomposition
from statsmodels.tsa.seasonal import STL  # STL decomposition
from statsmodels.tsa.filters.hp_filter import hpfilter  # HP Filter for trend/cycle


# --------------------------------------------------------------------------
# 📊 SignalVisualizer – Comprehensive Plots & Decompositions
# --------------------------------------------------------------------------


class SignalVisualizer:
    """
    Visualizes sentiment, return, technical indicator signals,
    and multiple decompositions of the Close price.

    Attributes:
    -----------
    df : pd.DataFrame
        Merged dataset containing price, sentiment, returns, and indicators.
    date_col : str
        Column name representing the date (must be datetime).
    """

    def __init__(self, df: pd.DataFrame, date_col: str = "cleaned_date"):
        """
        Initialize the SignalVisualizer.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing enriched features (aligned sentiment, returns, indicators,
            volume flags, etc.). Must include columns used by each plotting method.
        date_col : str, default="cleaned_date"
            Name of the date column in df (must be datetime).
        """
        self.df = df.copy()  # Copy to avoid modifying original
        self.date_col = date_col

        # Ensure date column is datetime type
        if not pd.api.types.is_datetime64_any_dtype(self.df[self.date_col]):
            try:
                self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
            except Exception as e:
                raise TypeError(f"Could not convert {self.date_col} to datetime: {e}")

        # Set Seaborn style once for consistency
        sns.set_style("whitegrid")

    def plot_sentiment_vs_price(self, ticker: str):
        """
        Plots stock Close price (blue) and weighted_sentiment (red) on dual y‐axes.

        Parameters:
        -----------
        ticker : str
            Stock ticker to filter on.
        """
        data = self.df[self.df["ticker"] == ticker]  # Filter to the selected ticker

        fig, ax1 = plt.subplots(figsize=(12, 5))  # Create figure and axis

        # Plot Close price on primary y‐axis (left)
        sns.lineplot(
            data=data,
            x=self.date_col,
            y="Close",
            ax=ax1,
            label="Close Price",
            color="blue",
        )
        ax1.set_ylabel("Price", color="blue")  # Label for price axis
        ax1.tick_params(axis="y", labelcolor="blue")  # Color the y‐axis labels blue

        # Create second y‐axis for weighted_sentiment (right)
        ax2 = ax1.twinx()
        sns.lineplot(
            data=data,
            x=self.date_col,
            y="weighted_sentiment",
            ax=ax2,
            label="Weighted Sentiment",
            color="red",
        )
        ax2.set_ylabel("Weighted Sentiment", color="red")  # Label for sentiment axis
        ax2.tick_params(axis="y", labelcolor="red")  # Color the y‐axis labels red

        plt.title(f"📈 {ticker} – Price vs. Weighted Sentiment")  # Chart title
        fig.tight_layout()  # Adjust layout
        plt.show()  # Render plot

    def plot_sentiment_return_scatter(self, ticker: str):
        """
        Plots scatterplot of weighted_sentiment vs. forward_return_1d.

        Parameters:
        -----------
        ticker : str
            Stock ticker to filter on.
        """
        data = self.df[self.df["ticker"] == ticker]  # Filter to ticker

        plt.figure(figsize=(8, 6))  # Create figure
        sns.scatterplot(
            data=data,
            x="weighted_sentiment",
            y="forward_return_1d",
            hue="weighted_sentiment",
            palette="coolwarm",
            alpha=0.6,
        )
        plt.axhline(0, linestyle="--", color="gray")  # Horizontal at 0 return
        plt.axvline(0, linestyle="--", color="gray")  # Vertical at 0 sentiment
        plt.title(f"🔍 {ticker} – Sentiment vs. Next-Day Return")
        plt.xlabel("Weighted Sentiment")
        plt.ylabel("Return (t+1)")
        plt.tight_layout()
        plt.show()

    def plot_technical_indicators(self, ticker: str):
        """
        Plots SMA_14, EMA_14, RSI_14, MACD, and MACD_signal for diagnostics.

        Parameters:
        -----------
        ticker : str
            Stock ticker to filter on.
        """
        data = self.df[self.df["ticker"] == ticker]  # Filter to ticker

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)  # 3-panel subplot

        # -------------------------
        # 1) Price with SMA & EMA
        # -------------------------
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
        axes[0].set_title("SMA & EMA over Close Price")  # Title for top subplot

        # -------------------------
        # 2) RSI with Overbought/Oversold
        # -------------------------
        sns.lineplot(
            data=data,
            x=self.date_col,
            y="RSI_14",
            ax=axes[1],
            label="RSI 14",
            color="green",
        )
        axes[1].axhline(
            70, linestyle="--", color="red", label="Overbought"
        )  # Overbought threshold
        axes[1].axhline(
            30, linestyle="--", color="blue", label="Oversold"
        )  # Oversold threshold
        axes[1].set_title("RSI (14-day)")  # Title for middle subplot

        # -------------------------
        # 3) MACD & Signal Line
        # -------------------------
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
            label="MACD Signal",
            color="brown",
        )
        axes[2].set_title("MACD & Signal Line")  # Title for bottom subplot

        plt.suptitle(f"📊 {ticker} – Technical Indicators", fontsize=14)
        plt.tight_layout()
        plt.show()

    def plot_volume_spike_vs_price(self, ticker: str):
        """
        Plots a bar chart of volume_spike under a line of Close price,
        coloring bars by sentiment_volume_divergence.

        Parameters:
        -----------
        ticker : str
            Stock ticker to filter on.
        """
        data = self.df[self.df["ticker"] == ticker]  # Filter to ticker

        # Create 2-row subplot: upper for price, lower for volume spike
        fig, (ax_price, ax_vol) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(12, 7),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )

        # -------------------------
        # Top: Close Price Line
        # -------------------------
        sns.lineplot(
            data=data,
            x=self.date_col,
            y="Close",
            ax=ax_price,
            label="Close Price",
            color="tab:blue",
        )
        ax_price.set_ylabel("Close Price", color="tab:blue")
        ax_price.tick_params(axis="y", labelcolor="tab:blue")

        # -------------------------
        # Bottom: Volume Spike Bars
        # -------------------------
        # Color bars based on sentiment_volume_divergence
        colors = data["sentiment_volume_divergence"].apply(
            lambda x: (
                "green"
                if x == "bullish_low_vol"
                else ("red" if x == "bearish_high_vol" else "gray")
            )
        )
        ax_vol.bar(
            data[self.date_col],
            data["volume_spike"],
            color=colors,
            label="Volume Spike",
        )
        ax_vol.set_ylabel("Vol Spike\n(1=Yes, 0=No)", color="black")
        ax_vol.set_ylim(-0.1, 1.1)  # Y‐axis limits for clarity
        ax_vol.set_xlabel("Date")
        plt.xticks(rotation=45)

        plt.suptitle(f"{ticker}: Price & Volume Spikes (Colored by Divergence)", y=0.95)
        plt.tight_layout()
        plt.show()

    def plot_price_decomposition(
        self, ticker: str, model: str = "additive", freq: int = 252
    ):
        """
        Decomposes the Close price time series into trend, seasonal, and residual
        components using seasonal_decompose (additive or multiplicative).

        Parameters:
        -----------
        ticker : str
            Stock ticker to filter on.
        model : str, default="additive"
            "additive" or "multiplicative" decomposition.
        freq : int, default=252
            Periodicity (e.g. ~252 trading days per year) used for seasonal extraction.
        """
        data = self.df[self.df["ticker"] == ticker].sort_values(
            self.date_col
        )  # Filter & sort
        ts = data.set_index(self.date_col)["Close"]  # Convert to time series

        # Perform seasonal decomposition
        decomposition = seasonal_decompose(ts, model=model, period=freq)

        # Create 4‐panel subplot: observed, trend, seasonal, residual
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

        # Observed
        decomposition.observed.plot(ax=axes[0], color="black", legend=False)
        axes[0].set_ylabel("Observed")

        # Trend
        decomposition.trend.plot(ax=axes[1], color="blue", legend=False)
        axes[1].set_ylabel("Trend")

        # Seasonal
        decomposition.seasonal.plot(ax=axes[2], color="orange", legend=False)
        axes[2].set_ylabel("Seasonal")

        # Residual
        decomposition.resid.plot(ax=axes[3], color="red", legend=False)
        axes[3].set_ylabel("Residual")

        plt.suptitle(
            f"{ticker}: {model.capitalize()} Decomposition (seasonal_period={freq})",
            fontsize=14,
        )
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_STL_decomposition(
        self, ticker: str, period: int = 252, robust: bool = True
    ):
        """
        Decomposes the Close price time series using STL (Loess‐based) into trend,
        seasonal, and resid components.

        Parameters:
        -----------
        ticker : str
            Stock ticker to filter on.
        period : int, default=252
            Length of seasonal cycle (e.g. ~252 trading days for yearly seasonality).
        robust : bool, default=True
            If True, uses robust fitting to reduce the influence of outliers.
        """
        data = self.df[self.df["ticker"] == ticker].sort_values(
            self.date_col
        )  # Filter & sort
        ts = data.set_index(self.date_col)["Close"].astype(
            float
        )  # Time series must be float

        # Perform STL decomposition
        stl = STL(ts, period=period, robust=robust)
        result = stl.fit()  # Fit the STL model

        # Create 4‐panel subplot: observed, trend, seasonal, resid
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

        # Observed
        axes[0].plot(ts.index, ts.values, color="black")
        axes[0].set_ylabel("Observed")

        # Trend
        axes[1].plot(ts.index, result.trend, color="blue")
        axes[1].set_ylabel("Trend")

        # Seasonal
        axes[2].plot(ts.index, result.seasonal, color="orange")
        axes[2].set_ylabel("Seasonal")

        # Residual
        axes[3].plot(ts.index, result.resid, color="red")
        axes[3].set_ylabel("Residual")

        plt.suptitle(
            f"{ticker}: STL Decomposition (period={period}, robust={robust})",
            fontsize=14,
        )
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_HP_filter(self, ticker: str, lamb: float = 1600):
        """
        Applies Hodrick‐Prescott (HP) filter to decompose Close price into trend and cycle.

        Parameters:
        -----------
        ticker : str
            Stock ticker to filter on.
        lamb : float, default=1600
            Smoothing parameter for HP filter (higher → smoother trend).
        """
        data = self.df[self.df["ticker"] == ticker].sort_values(
            self.date_col
        )  # Filter & sort
        ts = data.set_index(self.date_col)["Close"].astype(
            float
        )  # Time series must be float

        # Apply HP filter: returns (cycle, trend)
        cycle, trend = hpfilter(ts, lamb=lamb)

        # Create 2-panel subplot: trend and cycle
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

        # Trend
        axes[0].plot(ts.index, trend, color="blue", label="Trend")
        axes[0].plot(ts.index, ts.values, color="gray", alpha=0.4, label="Observed")
        axes[0].set_ylabel("Price")
        axes[0].legend()
        axes[0].set_title("HP Filter Trend")

        # Cycle
        axes[1].plot(ts.index, cycle, color="red", label="Cycle (Residual)")
        axes[1].axhline(0, linestyle="--", color="black", alpha=0.5)
        axes[1].set_ylabel("Cycle")
        axes[1].set_xlabel("Date")
        axes[1].legend()
        axes[1].set_title("HP Filter Cycle")

        plt.suptitle(f"{ticker}: HP Filter Decomposition (λ={lamb})", fontsize=14)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

"""
correlation_visualizer.py – Sentiment–Return Correlation Diagnostics
---------------------------------------------------------------------

Provides visualizations for inspecting the relationship between aggregated
sentiment signals and stock return metrics. Supports heatmaps, bar plots,
and scatter diagnostics per method or ticker.

Author: Nabil Mohamed
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# 📊 CorrelationVisualizer – High-Impact Signal Diagnostics
# ---------------------------------------------------------------------


class CorrelationVisualizer:
    """
    Visualizes sentiment–return correlation results from Task 3.

    Parameters:
    -----------
    correlation_df : pd.DataFrame
        Output from CorrelationAnalyzer module with expected columns.
    raw_df : pd.DataFrame, optional
        Raw aligned dataset (e.g., final_df or enriched_aligned_df) to use for scatter plots.
    verbose : bool
        Whether to print status messages.
    """

    def __init__(
        self,
        correlation_df: pd.DataFrame,
        raw_df: pd.DataFrame = None,
        verbose: bool = True,
    ):
        self.df = correlation_df.copy()
        self.raw_df = raw_df.copy() if raw_df is not None else None
        self.verbose = verbose

        # Validate expected columns in correlation summary
        expected_cols = {
            "ticker",
            "sentiment_feature",
            "return_feature",
            "method",
            "correlation",
            "n_obs",
        }
        if not expected_cols.issubset(self.df.columns):
            raise ValueError(f"Input DataFrame must contain columns: {expected_cols}")

        if self.verbose:
            print(f"✅ CorrelationVisualizer initialized with {len(self.df):,} rows.")
            if self.raw_df is not None:
                print(
                    f"📦 Raw time series DataFrame provided with {len(self.raw_df):,} rows."
                )

    def plot_heatmap(self, method: str = "pearson", figsize: tuple = (10, 6)):
        """Displays a heatmap of correlation scores for a given method."""
        df_method = self.df[self.df["method"] == method]

        if df_method.empty:
            print(f"⚠️ No correlations found for method: {method}")
            return

        pivot = df_method.pivot_table(
            index="sentiment_feature",
            columns="return_feature",
            values="correlation",
            aggfunc="mean",
        )

        plt.figure(figsize=figsize)
        sns.heatmap(pivot, annot=True, cmap="coolwarm", center=0, fmt=".2f")
        plt.title(f"Sentiment–Return Correlation Heatmap ({method})", fontsize=14)
        plt.tight_layout()
        plt.show()

    def plot_top_correlations(self, top_n: int = 10, method: str = "pearson"):
        """Bar plot of top absolute correlation scores."""
        df_method = self.df[self.df["method"] == method].copy()

        if df_method.empty:
            print(f"⚠️ No correlations found for method: {method}")
            return

        df_method["abs_corr"] = df_method["correlation"].abs()
        top_df = df_method.sort_values(by="abs_corr", ascending=False).head(top_n)

        plt.figure(figsize=(10, 5))
        sns.barplot(
            data=top_df,
            x="correlation",
            y=top_df.apply(
                lambda row: f"{row['sentiment_feature']} vs {row['return_feature']}",
                axis=1,
            ),
            hue="ticker",
            dodge=False,
        )
        plt.axvline(0, color="black", linestyle="--")
        plt.title(f"Top {top_n} Correlations ({method})", fontsize=14)
        plt.xlabel("Correlation")
        plt.ylabel("Sentiment–Return Pair")
        plt.tight_layout()
        plt.show()

    def plot_scatter_pairs(
        self,
        sentiment_col: str,
        return_col: str,
        ticker: str = None,
        method: str = "pearson",
    ):
        """
        Scatterplot with regression fit for a single sentiment–return pair.

        Requires raw_df to be passed during initialization.

        Parameters:
        -----------
        sentiment_col : str
            Name of sentiment column.
        return_col : str
            Name of return metric column.
        ticker : str
            Optional ticker to filter by.
        method : str
            Correlation method used (for title only).
        """
        if self.raw_df is None:
            print("🚫 raw_df not provided. Scatter plot requires raw time series data.")
            return

        df = self.raw_df.copy()
        if ticker:
            df = df[df["ticker"] == ticker]

        # Check if the required columns exist
        if sentiment_col not in df.columns or return_col not in df.columns:
            print(f"🚫 Columns missing in raw_df: {sentiment_col} or {return_col}")
            return

        df = df[[sentiment_col, return_col]].dropna()
        if df.empty:
            print(
                f"⚠️ No data available for plotting {sentiment_col} vs {return_col} (ticker={ticker})"
            )
            return

        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=sentiment_col, y=return_col, data=df)
        sns.regplot(x=sentiment_col, y=return_col, data=df, scatter=False, color="red")
        title = f"{sentiment_col} vs {return_col}"
        if ticker:
            title += f" | {ticker}"
        plt.title(f"{title} ({method})", fontsize=12)
        plt.tight_layout()
        plt.show()

    def filter_by(self, method: str = None, ticker: str = None) -> pd.DataFrame:
        """
        Helper to return subset of correlation_df for review or reuse.

        Parameters:
        -----------
        method : str
            Correlation method to filter by.
        ticker : str
            Ticker to filter by.

        Returns:
        --------
        pd.DataFrame
            Filtered correlation result.
        """
        df = self.df.copy()
        if method:
            df = df[df["method"] == method]
        if ticker:
            df = df[df["ticker"] == ticker]
        return df.reset_index(drop=True)

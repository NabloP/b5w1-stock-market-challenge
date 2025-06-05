"""
correlation_analyzer.py – Multimethod Correlation Engine
---------------------------------------------------------

Computes correlation coefficients between sentiment signals and stock returns
using multiple methods: Pearson, Spearman, Kendall's Tau.

Features:
- Supports ticker-level or global analysis
- Handles NaNs and type safety automatically
- Applies method-appropriate assumptions (parametric vs non-parametric)
- Returns long-format DataFrame for downstream plotting or sorting

Author: Nabil Mohamed
"""

import pandas as pd
import numpy as np
from typing import List, Literal


# ------------------------------------------------------------------
# 🔁 CorrelationAnalyzer – Flexible, Multimethod, Modular
# ------------------------------------------------------------------


class CorrelationAnalyzer:
    """
    Computes correlations between sentiment signals and return metrics.

    Parameters:
    -----------
    df : pd.DataFrame
        The full aligned and enriched dataset.
    methods : list[str]
        Correlation methods to compute: ['pearson', 'spearman', 'kendall']
    by_ticker : bool
        Whether to compute per-ticker correlations or aggregate across all.
    verbose : bool
        Toggle diagnostics.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        methods: List[Literal["pearson", "spearman", "kendall"]] = ["pearson"],
        by_ticker: bool = True,
        verbose: bool = True,
    ):
        self.df = df.copy()
        self.methods = methods
        self.by_ticker = by_ticker
        self.verbose = verbose

        # Drop rows missing key fields to avoid runtime errors
        self.df.dropna(subset=["ticker", "cleaned_date"], inplace=True)

        # Confirm numeric fields are clean
        self.df = self.df.apply(pd.to_numeric, errors="ignore")

        if self.verbose:
            print(f"✅ Initialized CorrelationAnalyzer with {len(self.df):,} rows.")

    def compute(
        self,
        sentiment_cols: List[str],
        return_cols: List[str],
    ) -> pd.DataFrame:
        """
        Computes correlation matrix for all combinations of sentiment and return metrics.

        Parameters:
        -----------
        sentiment_cols : list of str
            Names of sentiment signal columns (e.g., weighted_sentiment, agg_sentiment_ewm)
        return_cols : list of str
            Names of return metric columns (e.g., forward_return_1d, return_t)

        Returns:
        --------
        pd.DataFrame : Long-format correlation summary
        """
        results = []

        if self.by_ticker:
            tickers = self.df["ticker"].unique()
            for ticker in tickers:
                sub_df = self.df[self.df["ticker"] == ticker]
                res = self._compute_pairwise(
                    sub_df, sentiment_cols, return_cols, ticker
                )
                results.extend(res)
        else:
            res = self._compute_pairwise(
                self.df, sentiment_cols, return_cols, ticker="ALL"
            )
            results.extend(res)

        return pd.DataFrame(results)

    def _compute_pairwise(
        self,
        df: pd.DataFrame,
        sentiment_cols: List[str],
        return_cols: List[str],
        ticker: str,
    ) -> List[dict]:
        """
        Helper to compute all correlation pairs for a single ticker or full set.

        Returns list of dicts with correlation results.
        """
        output = []

        for method in self.methods:
            for s_col in sentiment_cols:
                for r_col in return_cols:
                    # Drop missing pairs
                    temp = df[[s_col, r_col]].dropna()

                    if temp.empty or temp.shape[0] < 3:
                        continue

                    try:
                        corr = temp[s_col].corr(temp[r_col], method=method)
                        output.append(
                            {
                                "ticker": ticker,
                                "sentiment_feature": s_col,
                                "return_feature": r_col,
                                "method": method,
                                "correlation": corr,
                                "n_obs": temp.shape[0],
                            }
                        )
                    except Exception as e:
                        if self.verbose:
                            print(
                                f"❌ Failed {method} on {ticker} | {s_col} vs {r_col}: {e}"
                            )

        return output

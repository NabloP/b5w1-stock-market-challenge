# scripts/quantitative_analysis_orchestrator.py

"""
🧠 Task 2 – Quantitative Sentiment–Price Analysis
----------------------------------------------------
This script performs alignment between financial news sentiment and historical stock prices,
computes technical indicators for both full history and aligned-only subsets (with progress bars enabled),
and visualizes key signals for downstream correlation analysis.
"""

# ------------------------------------------------------------------------------
# 📂 Make sure `src/` is on PYTHONPATH so that `import src.xxx` works
# ------------------------------------------------------------------------------
import os
import sys

# Insert project root (one level up from this script) into sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ------------------------------------------------------------------------------
# 📂 Project Directory & File Checks
# ------------------------------------------------------------------------------
import pandas as pd

# Ensure project root context (in case someone double-clicks the script)
if os.path.basename(os.getcwd()) == "scripts":
    os.chdir("..")
print("📂 Working directory is now:", os.getcwd())

# File paths
SENTIMENT_FILE = "data/cleaned_headlines_sample.csv"
PRICE_DIR = "data/yfinance_data"
EXPECTED_TICKERS = ["AAPL", "AMZN", "GOOG", "META", "MSFT", "NVDA", "TSLA"]

print("\n📁 Checking dataset files:")
print(
    "✅ Sentiment file found"
    if os.path.exists(SENTIMENT_FILE)
    else f"❌ Missing: {SENTIMENT_FILE}"
)
for ticker in EXPECTED_TICKERS:
    path = os.path.join(PRICE_DIR, f"{ticker}_historical_data.csv")
    print(f"✅ {ticker} file found" if os.path.exists(path) else f"❌ Missing: {path}")

# ------------------------------------------------------------------------------
# 🔌 Import Core Modules & Custom Loaders
# ------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import importlib
from tqdm import tqdm

# Reloadable modules
import src.price_data_loader
import src.news_loader
import src.sentiment_return_aligner
import src.technical_indicator_calculator
import src.signal_visualizer

importlib.reload(src.price_data_loader)
importlib.reload(src.news_loader)
importlib.reload(src.sentiment_return_aligner)
importlib.reload(src.technical_indicator_calculator)
importlib.reload(src.signal_visualizer)

from src.price_data_loader import PriceDataLoader
from src.news_loader import NewsDataLoader
from src.sentiment_return_aligner import SentimentReturnAligner
from src.technical_indicator_calculator import TechnicalIndicatorCalculator
from src.signal_visualizer import SignalVisualizer

# ------------------------------------------------------------------------------
# 📥 Load Data
# ------------------------------------------------------------------------------
try:
    price_loader = PriceDataLoader(folder_path=PRICE_DIR, verbose=True)
    prices_df = price_loader.load_all()
    print("✅ Stock prices loaded")
except Exception as e:
    print(f"❌ Stock price loading failed: {e}")
    prices_df = None

try:
    sentiment_loader = NewsDataLoader(
        path=SENTIMENT_FILE, parse_dates=["cleaned_date"], verbose=True
    )
    sentiment_df = sentiment_loader.load()
    print("✅ Sentiment data loaded")
except Exception as e:
    print(f"❌ Sentiment loading failed: {e}")
    sentiment_df = None


# ------------------------------------------------------------------------------
# ✅ Sanity Check Function
# ------------------------------------------------------------------------------
def run_sanity_check(df, name="DataFrame"):
    print(f"\n🧪 Sanity Check – {name}")
    print("-" * 50)
    if df is not None:
        display(df.head(3))
        print(df.dtypes)
        missing = df.isna().sum()
        print(missing if missing.any() else "✅ No missing values")
        dups = df.duplicated().sum()
        print(f"⚠️ {dups} duplicate rows" if dups else "✅ No duplicate rows")
    else:
        print(f"🚫 {name} is None")


run_sanity_check(prices_df, "Stock Prices")
run_sanity_check(sentiment_df, "Sentiment Data")

# ------------------------------------------------------------------------------
# 🔗 Sentiment Alignment Pipeline (with progress bar)
# ------------------------------------------------------------------------------
aligned_df = None
if sentiment_df is not None and prices_df is not None:
    # 1) Rename 'stock' to 'ticker' if needed
    if "stock" in sentiment_df.columns:
        sentiment_df.rename(columns={"stock": "ticker"}, inplace=True)

    # 2) Normalize ticker case and strip whitespace
    sentiment_df["ticker"] = sentiment_df["ticker"].astype(str).str.strip().str.upper()
    prices_df["ticker"] = prices_df["ticker"].astype(str).str.strip().str.upper()

    # 3) Convert and normalize dates
    sentiment_df["cleaned_date"] = pd.to_datetime(
        sentiment_df["cleaned_date"], errors="coerce"
    ).dt.normalize()
    prices_df["cleaned_date"] = pd.to_datetime(
        prices_df["cleaned_date"], errors="coerce"
    ).dt.normalize()

    try:
        aligner = SentimentReturnAligner(
            sentiment_df=sentiment_df,
            price_df=prices_df,
            decay_lambda=0.7,
            verbose=True,
            use_tqdm=True,  # enable progress bar
        )
        aligned_df = aligner.align()
        print("\n✅ Alignment complete. `aligned_df` shape:", aligned_df.shape)
    except Exception as e:
        print(f"❌ Alignment failed: {type(e).__name__} – {e}")
        aligned_df = None
else:
    print("🚫 Skipping alignment – sentiment_df or prices_df is None")

run_sanity_check(aligned_df, "Aligned Data")

# ─────────────────────────────────────────────────────────────────────────────
# 🧮 Compute Technical Indicators & Hybrid Performance Metrics on “Full History” and “Aligned-Only” Frames
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd  # For DataFrame operations

# Import our indicator calculator and the hybrid performance calculator that uses PyNance.stderr if available
from src.technical_indicator_calculator import (
    TechnicalIndicatorCalculator,
    PyNancePerformanceCalculatorHybrid,
)

# Step 1: Validate input DataFrames
if not isinstance(aligned_df, pd.DataFrame) or not isinstance(prices_df, pd.DataFrame):
    raise TypeError("❌ Both aligned_df and prices_df must be pandas DataFrames")

print("Step 1: aligned_df type →", type(aligned_df))
print("Step 2: prices_df   type →", type(prices_df))

# Step 2: Drop duplicate columns
aligned_df = aligned_df.loc[:, ~aligned_df.columns.duplicated()].copy()
prices_df = prices_df.loc[:, ~prices_df.columns.duplicated()].copy()

# Step 3: Check for required columns
required_price_cols = {"ticker", "cleaned_date", "High", "Low", "Close", "Volume"}
required_aligned_cols = {"ticker", "cleaned_date"}

if missing := required_price_cols - set(prices_df.columns):
    raise KeyError(f"prices_df is missing required columns: {missing}")
if missing := required_aligned_cols - set(aligned_df.columns):
    raise KeyError(f"aligned_df is missing required columns: {missing}")

# Step 4: Compute technical indicators on full price history
full_calc = TechnicalIndicatorCalculator(
    df=prices_df,
    ticker_col="ticker",
    date_col="cleaned_date",
    verbose=True,
    use_tqdm=True,
)
indicators_full = full_calc.add_indicators()
print("✅ indicators_full type →", type(indicators_full))
display(indicators_full.head(3))

# Step 5: Compute hybrid performance metrics per ticker
try:
    perf_calc = PyNancePerformanceCalculatorHybrid(
        df=indicators_full[["ticker", "cleaned_date", "Close"]].copy(),
        verbose=True,
    )
    performance_summary = perf_calc.compute_summary()
    print("\n✅ Hybrid performance summary:")
    display(performance_summary)
except ImportError as ie:
    print(f"⚠️ Skipping hybrid performance metrics: {ie}")
    performance_summary = pd.DataFrame()
except Exception as e:
    print(f"⚠️ Error computing hybrid performance metrics: {type(e).__name__} – {e}")
    performance_summary = pd.DataFrame()

# Step 5b: Merge performance summary into full and aligned enriched DataFrames
if not performance_summary.empty:
    enriched_perf = performance_summary.reset_index()
else:
    enriched_perf = pd.DataFrame(
        columns=[
            "ticker",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
        ]
    )

# Step 6: Merge indicators onto full price DataFrame
enriched_full_df = pd.merge(
    prices_df,
    indicators_full,
    on=["ticker", "cleaned_date"],
    how="left",
    validate="one_to_one",
)
enriched_full_df = pd.merge(
    enriched_full_df,
    enriched_perf,
    on="ticker",
    how="left",
    validate="many_to_one",
)
print("\nStep 6: enriched_full_df type →", type(enriched_full_df))
print("         enriched_full_df shape →", enriched_full_df.shape)

# Step 7: Merge indicators onto aligned sentiment DataFrame
enriched_aligned_df = pd.merge(
    aligned_df,
    indicators_full,
    on=["ticker", "cleaned_date"],
    how="left",
    validate="many_to_one",
)
enriched_aligned_df = pd.merge(
    enriched_aligned_df,
    enriched_perf,
    on="ticker",
    how="left",
    validate="many_to_one",
)
print("\nStep 7: enriched_aligned_df type →", type(enriched_aligned_df))
print("         enriched_aligned_df shape →", enriched_aligned_df.shape)
print("         Columns in enriched_aligned_df →", list(enriched_aligned_df.columns))

# Step 8: Check for missing values in indicator columns
indicator_cols = ["SMA_14", "EMA_14", "RSI_14", "MACD", "MACD_signal", "ATR_14"]
print("\nStep 8: Null counts in enriched_aligned_df indicator columns:")
for col in indicator_cols:
    if col not in enriched_aligned_df.columns:
        print(f"  ❌ Column '{col}' not found in enriched_aligned_df")
    else:
        print(f"  → {col}: {enriched_aligned_df[col].isna().sum():,} missing values")

# Step 9: Preview both enriched DataFrames
print("\n✅ enriched_full_df (every trading day + indicators) preview:")
display(enriched_full_df.head(3))
print("\n✅ enriched_aligned_df (aligned dates only + indicators) preview:")
display(enriched_aligned_df.head(3))

run_sanity_check(enriched_aligned_df, "Enriched Aligned Data")

# ------------------------------------------------------------------------------
# 📊 Visualization (With Pre-Checks)
# ------------------------------------------------------------------------------
if enriched_full_df is not None:
    # Fix merged suffixes so that 'Close' appears in enriched_full_df
    rename_map = {}
    for col in ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"]:
        if f"{col}_x" in enriched_full_df.columns:
            rename_map[f"{col}_x"] = col
    enriched_full_df.rename(columns=rename_map, inplace=True)
    to_drop = [c for c in enriched_full_df.columns if c.endswith("_y")]
    if to_drop:
        enriched_full_df.drop(columns=to_drop, inplace=True)
else:
    print("🚫 enriched_full_df is None; will skip TA-only plots.")

if enriched_aligned_df is not None:
    final_df = enriched_aligned_df  # renamed for clarity
else:
    final_df = None
    print("🚫 enriched_aligned_df is None; will skip sentiment-based plots.")

# Instantiate visualizers
full_vis = (
    SignalVisualizer(df=enriched_full_df) if enriched_full_df is not None else None
)
aligned_vis = SignalVisualizer(df=final_df) if final_df is not None else None

# Choose ticker for diagnostics
selected_ticker = "AAPL"

print(f"\n🔎 Visualizing signals for '{selected_ticker}':\n")

# 1) SENTIMENT-BASED PLOTS on final_df
if aligned_vis is None or final_df is None:
    print("⚠️ Cannot run sentiment-based plots – `aligned_vis` or `final_df` is None.")
else:
    if selected_ticker not in final_df["ticker"].unique():
        print(
            f"⚠️ Ticker '{selected_ticker}' not found in final_df. Skipping sentiment-based plots."
        )
    else:
        # Confirm required columns
        missing_cols = [
            c for c in ["Close", "weighted_sentiment"] if c not in final_df.columns
        ]
        if missing_cols:
            print(
                f"⚠️ final_df missing columns {missing_cols}. Skipping sentiment vs. price."
            )
        else:
            subset = final_df[final_df["ticker"] == selected_ticker]
            if subset["Close"].notna().sum() == 0:
                print(
                    f"⚠️ No non-null 'Close' for '{selected_ticker}' in final_df. Skipping price plot."
                )
            elif subset["weighted_sentiment"].notna().sum() == 0:
                print(
                    f"⚠️ No non-null 'weighted_sentiment' for '{selected_ticker}' in final_df. Skipping sentiment plot."
                )
            else:
                aligned_vis.plot_sentiment_vs_price(ticker=selected_ticker)
                if (
                    "forward_return_1d" in final_df.columns
                    and subset["forward_return_1d"].notna().sum() > 0
                ):
                    aligned_vis.plot_sentiment_return_scatter(ticker=selected_ticker)
                else:
                    print(
                        f"⚠️ No valid 'forward_return_1d' for '{selected_ticker}'. Skipping scatter."
                    )

# 2) TA-ONLY PLOTS on enriched_full_df
if full_vis is None or enriched_full_df is None:
    print("⚠️ Cannot run TA-only plots – `full_vis` or `enriched_full_df` is None.")
else:
    if selected_ticker not in enriched_full_df["ticker"].unique():
        print(
            f"⚠️ Ticker '{selected_ticker}' not found in enriched_full_df. Skipping TA-only plots."
        )
    else:
        # Confirm 'Close' exists
        if "Close" not in enriched_full_df.columns:
            print(
                "⚠️ 'Close' missing in enriched_full_df. Skipping technical indicators."
            )
        else:
            subset_full = enriched_full_df[
                enriched_full_df["ticker"] == selected_ticker
            ]
            if subset_full["Close"].notna().sum() == 0:
                print(
                    f"⚠️ No non-null 'Close' for '{selected_ticker}' in enriched_full_df. Skipping TA plot."
                )
            else:
                ta_cols = ["SMA_14", "EMA_14", "RSI_14", "MACD", "MACD_signal"]
                missing_ta = [c for c in ta_cols if c not in enriched_full_df.columns]
                if missing_ta:
                    print(
                        f"⚠️ Missing TA columns {missing_ta} in enriched_full_df. Skipping TA plot."
                    )
                else:
                    full_vis.plot_technical_indicators(ticker=selected_ticker)

        # Price decomposition (need ≥504 rows)
        n_points = subset_full.shape[0]
        if n_points >= 504:
            full_vis.plot_price_decomposition(
                ticker=selected_ticker, model="additive", freq=252
            )
            full_vis.plot_price_decomposition(
                ticker=selected_ticker, model="multiplicative", freq=252
            )
            full_vis.plot_STL_decomposition(
                ticker=selected_ticker, period=252, robust=True
            )
        else:
            print(
                f"⚠️ Only {n_points} rows for '{selected_ticker}'. Need ≥504 for decomposition. Skipping."
            )

        # HP filter (requires ≥1 row)
        if n_points > 0:
            full_vis.plot_HP_filter(ticker=selected_ticker, lamb=1600)
        else:
            print(
                f"⚠️ No data for '{selected_ticker}' in enriched_full_df. Skipping HP filter."
            )

# -------------------------------------------------------------------
# 💾 Save Outputs to Disk (DataFrames and Plots)
# -------------------------------------------------------------------

# Define output directory
output_dir = "data/outputs"
plot_dir = os.path.join(output_dir, "plots")

# Create directories if they don't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# Save DataFrames
try:
    enriched_full_df.to_csv(
        os.path.join(output_dir, "enriched_full_df.csv"), index=False
    )
    print("✅ Saved: enriched_full_df.csv")
except Exception as e:
    print(f"❌ Failed to save enriched_full_df: {e}")

try:
    enriched_aligned_df.to_csv(
        os.path.join(output_dir, "enriched_aligned_df.csv"), index=False
    )
    print("✅ Saved: enriched_aligned_df.csv")
except Exception as e:
    print(f"❌ Failed to save enriched_aligned_df: {e}")

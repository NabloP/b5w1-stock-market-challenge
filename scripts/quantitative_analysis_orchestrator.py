# quantitative_analysis_orchestrator.py

"""
🧠 Task 2 – Quantitative Sentiment–Price Analysis
----------------------------------------------------
This script performs alignment between financial news sentiment and historical stock prices,
computes technical indicators, and visualizes signals for downstream correlation analysis.
"""

# ------------------------------------------------------------------------------
# 📂 Project Directory & File Checks
# ------------------------------------------------------------------------------
import os
import pandas as pd

# Ensure project root context
if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")
print("📂 Working directory is now:", os.getcwd())

# File paths
SENTIMENT_FILE = "data/cleaned_headlines_sample.csv"
PRICE_DIR = "data/yfinance_data"
EXPECTED_TICKERS = ["AAPL", "AMZN", "GOOG", "META", "MSFT", "NVDA", "TSLA"]

# File existence checks
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
        print(df.isna().sum() if df.isna().sum().any() else "✅ No missing values")
        dups = df.duplicated().sum()
        print(f"⚠️ {dups} duplicate rows" if dups else "✅ No duplicate rows")
    else:
        print(f"🚫 {name} is None")


run_sanity_check(prices_df, "Stock Prices")
run_sanity_check(sentiment_df, "Sentiment Data")

# ------------------------------------------------------------------------------
# 🔗 Sentiment Alignment Pipeline
# ------------------------------------------------------------------------------
if sentiment_df is not None:
    sentiment_df.rename(columns={"stock": "ticker"}, inplace=True)

try:
    aligner = SentimentReturnAligner(
        price_df=prices_df,
        sentiment_df=sentiment_df,
        decay_lambda=0.7,
        verbose=True,
    )
    aligned_df = aligner.align()
    display(aligned_df.head())
    print("✅ Alignment complete")
except Exception as e:
    print(f"❌ Alignment failed: {e}")
    aligned_df = None

# ------------------------------------------------------------------------------
# 🧮 Technical Indicator Computation
# ------------------------------------------------------------------------------

try:
    indicator_calc = TechnicalIndicatorCalculator(
        df=aligned_df,
        ticker_col="ticker",
        date_col="cleaned_date",
        verbose=True,
    )
    enriched_df = indicator_calc.add_indicators()
    display(enriched_df.head())
    print("✅ Indicators added")
except Exception as e:
    print(f"❌ Indicator computation failed: {e}")
    enriched_df = None

# ------------------------------------------------------------------------------
# 📊 Visualization
# ------------------------------------------------------------------------------

if enriched_df is not None:
    visualizer = SignalVisualizer(df=enriched_df)
    ticker = "AAPL"  # Can be parameterized
    visualizer.plot_sentiment_vs_price(ticker)
    visualizer.plot_sentiment_return_scatter(ticker)
    visualizer.plot_technical_indicators(ticker)
else:
    print("🚫 Visualization skipped – Enriched DataFrame unavailable")

# eda_orchestrator.py – Full Pipeline Orchestration Script for B5W1 Task 1
# -----------------------------------------------------------------------
# This script runs the full exploratory data analysis pipeline for the
# financial news sentiment challenge from start to finish.
# Author: Nabil Mohamed | Date: 2025-06-01

# -----------------------------
# 🛠 Environment Setup
# -----------------------------
import os  # OS operations (e.g., working directory check)
import sys  # System path access for imports
import pandas as pd  # Data manipulation
import numpy as np  # Numerical ops (used in modules)
import matplotlib.pyplot as plt  # Plotting
import seaborn as sns  # Enhanced plots

# Ensure correct working directory when run from /notebooks/
if os.path.basename(os.getcwd()) == "notebooks":  # If in 'notebooks/' folder
    os.chdir("..")  # Go up to project root
print("📂 Working directory set to:", os.getcwd())  # Confirm directory

# -----------------------------
# 📥 Load Dataset
# -----------------------------
from src.news_loader import NewsDataLoader  # Custom loader module

DATA_PATH = "data/raw_analyst_ratings.csv"  # Path to source dataset

try:
    loader = NewsDataLoader(path=DATA_PATH)  # Initialize loader
    df = loader.load()  # Load CSV and return DataFrame
    print("✅ Dataset loaded successfully.")
except Exception as e:
    print(f"❌ Dataset loading failed: {e}")  # Catch loading issues
    df = None  # Avoid crashing pipeline

# -----------------------------
# ⏰ Timestamp Cleaning
# -----------------------------
from src.text_timestamp_cleaner import TimestampCleaner  # Timestamp module

if df is not None:  # Proceed only if dataset loaded
    try:
        ts_cleaner = TimestampCleaner(verbose=True)  # Enable logs
        df = ts_cleaner.clean(df)  # Add cleaned_date column
        print("✅ Timestamp parsing succeeded.")
    except Exception as e:
        print(f"❌ Timestamp parsing failed: {e}")

# -----------------------------
# 🧼 Headline Text Cleaning
# -----------------------------
from src.text_cleaner import TextCleaner  # Headline cleaner

try:
    cleaner = TextCleaner(  # Configure cleaner
        text_col="headline",
        output_col="cleaned_headline",
        lowercase=True,
        remove_html=True,
        remove_punctuation=True,
        remove_stopwords=True,
        verbose=True,
    )
    df = cleaner.transform(df)  # Apply transformation
    print("✅ Headline cleaning completed.")
except Exception as e:
    print(f"❌ Headline cleaning failed: {e}")

# -----------------------------
# 🧠 Feature Engineering – Headline, Publisher
# -----------------------------
from src.text_features import TextFeatureExtractor  # For word/length stats
from src.text_publishers import PublisherAnalyzer  # For publisher analytics

try:
    feature_extractor = TextFeatureExtractor(headline_col="headline", verbose=True)
    df = feature_extractor.transform(df)  # Add 'word_count', 'headline_length'

    publisher_analyzer = PublisherAnalyzer(publisher_col="publisher", verbose=True)
    publisher_freq_df = publisher_analyzer.analyze(df)  # Count publishers
    publisher_analyzer.plot_top_publishers(top_n=10)  # Plot most active sources
    publisher_analyzer.plot_publication_dates(  # Plot time activity
        publisher_analyzer.analyze_publication_dates(df, date_col="cleaned_date")
    )
    print("✅ Feature engineering completed.")
except Exception as e:
    print(f"❌ Feature engineering failed: {e}")

# -----------------------------
# 📊 Text Distribution Visualization
# -----------------------------
from src.text_distributions import TextDistributionPlotter

try:
    plotter = TextDistributionPlotter(  # Configure visualizer
        length_col="headline_length",
        word_col="word_count",
        bins=40,
        title_prefix="📝",
        style="seaborn-v0_8-muted",
        use_latex_style=False,
        verbose=True,
    )
    plotter.plot(df, show=True)  # Plot length and word count histograms
    print("✅ Headline distribution visualization completed.")
except Exception as e:
    print(f"❌ Failed to visualize headline distributions: {e}")

# -----------------------------
# 🔍 Sentiment Labeling Pipeline
# -----------------------------
from src.text_keywords import (
    flag_bullish,
    flag_bearish,
    KeywordSentimentAnalyzer,
    EnsembleSentimentAnalyzer,
)

try:
    df = flag_bullish(df, column="cleaned_headline")  # Apply bullish flag
    df = flag_bearish(df, column="cleaned_headline")  # Apply bearish flag

    vader_analyzer = KeywordSentimentAnalyzer([], "cleaned_headline", verbose=True)
    df = vader_analyzer.run_vader_sentiment(df)  # Run VADER

    ensemble_analyzer = EnsembleSentimentAnalyzer(verbose=True)  # Initialize
    df = ensemble_analyzer.run(df)  # Compute ensemble labels
    print("✅ Sentiment analysis completed.")
except Exception as e:
    print(f"❌ Sentiment analysis failed: {e}")

# -----------------------------
# 🧠 Event Extraction Pipeline
# -----------------------------
from src.text_event_analysis import (
    EventExtractor,
    EventTimelineAnalyzer,
    EventExtractorREBEL,
)

try:
    event_extractor = EventExtractor(
        df["cleaned_headline"].dropna().tolist(), verbose=True
    )
    entity_df = event_extractor.extract_named_entities()
    noun_df = event_extractor.extract_noun_phrases()

    rebel_model = EventExtractorREBEL(verbose=True)
    combined_events_df = event_extractor.extract_combined_events(rebel_model)

    freq_df = event_extractor.compute_event_frequencies()
    event_extractor.visualize_top_events(top_n=15)

    combined_events_df["cleaned_date"] = df.loc[
        combined_events_df.index, "cleaned_date"
    ]
    timeline_analyzer = EventTimelineAnalyzer(combined_events_df, verbose=True)
    timeline_analyzer.plot_event_timeline()
    print("✅ Event extraction pipeline completed.")
except Exception as e:
    print(f"❌ Event extraction pipeline failed: {e}")

# -----------------------------
# 💹 Stock-Level Diagnostics
# -----------------------------
from src.text_stock_eda import (
    StockHeadlineProfiler,
    StockVolatilityAnalyzer,
)

try:
    profiler = StockHeadlineProfiler(df, verbose=True)  # Initialize
    volume_df = profiler.get_headline_volume()  # Count headlines
    sentiment_df = profiler.get_sentiment_distribution()  # Count bullish/bearish

    profiler.plot_top_stocks_by_volume(n=10)  # Show high-volume stocks
    profiler.plot_sentiment_share_by_stock(n=10)  # Show sentiment mix
    profiler.plot_sentiment_distribution("AAPL")  # Plot AAPL's sentiment

    volatility_analyzer = StockVolatilityAnalyzer(df, verbose=True)  # Initialize
    volatility_analyzer.plot_volume_spikes("AAPL")  # Show AAPL spikes
    volatility_analyzer.plot_sentiment_flips("AAPL")  # Show flip zones
    print("✅ Stock-level diagnostics completed.")
except Exception as e:
    print(f"❌ Stock-level diagnostics failed: {e}")

# -----------------------------
# ✅ Pipeline Completion Log
# -----------------------------
print("🏁 All pipeline stages completed successfully.")

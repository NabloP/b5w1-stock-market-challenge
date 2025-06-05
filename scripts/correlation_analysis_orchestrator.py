"""
correlation_analysis_orchestrator.py – Task 3 Orchestration Script
------------------------------------------------------------------

Executes full sentiment–return correlation pipeline for B5W1:
- Loads enriched sentiment and price data
- Aggregates sentiment signals daily using exponential decay
- Computes correlation between sentiment and returns
- Visualizes signal strength and correlation diagnostics

Author: Nabil Mohamed
"""

# ------------------------------------------------------------------
# ✅ Imports: Standard Libraries
# ------------------------------------------------------------------
import os  # Used to inspect and manipulate the working directory
import sys  # Used to append custom module paths to the system path
from pathlib import Path  # Provides platform-independent path handling
import pandas as pd  # Used for all DataFrame operations
from IPython.display import display  # Used to display DataFrame previews in notebooks

# ------------------------------------------------------------------
# 🔧 Path Fix: Add 'src/' to sys.path so custom imports work
# ------------------------------------------------------------------
project_root = (
    Path(__file__).resolve().parents[1]
)  # Resolve the absolute path to the project root
src_path = project_root / "src"  # Define the path to the src directory

if (
    src_path.exists() and str(src_path) not in sys.path
):  # If the src path exists and isn't already in sys.path
    sys.path.append(
        str(src_path)
    )  # Append it to sys.path so Python can find the modules
    print(f"🔧 Added to sys.path: {src_path}")  # Confirm addition
else:
    print(
        f"⚠️ src path does not exist or already in sys.path: {src_path}"
    )  # Warn if not added

# ------------------------------------------------------------------
# ✅ Imports: Custom Modules (Task 3)
# ------------------------------------------------------------------
from correlation_data_loader import (
    CorrelationDataLoader,
)  # Module to load enriched aligned data
from correlation_sentiment_aggregator import (
    CorrelationSentimentAggregator,
)  # Module to aggregate daily sentiment
from correlation_analyzer import CorrelationAnalyzer  # Module to compute correlations
from correlation_visualizer import CorrelationVisualizer  # Module to visualize results

# ------------------------------------------------------------------
# 🔁 1. Set Working Directory and Define Paths
# ------------------------------------------------------------------
if (
    os.path.basename(os.getcwd()) == "notebooks"
):  # If script is run from notebooks directory
    os.chdir("..")  # Move up one level to reach project root
print("📂 Working directory is now:", os.getcwd())  # Confirm working directory

ALIGNED_PATH = "data/outputs/enriched_aligned_df.csv"  # Path to aligned sentiment–price merged file
FULL_PATH = (
    "data/outputs/enriched_full_df.csv"  # Path to the full enriched news dataset
)

# ------------------------------------------------------------------
# 📥 2. Load Data Using CorrelationDataLoader
# ------------------------------------------------------------------
try:
    loader = CorrelationDataLoader(  # Instantiate the custom data loader class
        full_path=FULL_PATH,
        aligned_path=ALIGNED_PATH,
        verbose=True,  # Provide file paths and set verbosity
    )
    aligned_df = loader.load_aligned_df()  # Load the aligned sentiment–price dataset
    full_df = (
        loader.load_full_df()
    )  # Load the full news dataset (optional for exploration)
except Exception as e:
    print(f"❌ Failed to load datasets: {e}")  # Print any errors encountered
    aligned_df, full_df = None, None  # Set fallback to None for robustness

# ------------------------------------------------------------------
# 🧪 3. Sanity Check for Aligned Dataset
# ------------------------------------------------------------------
if aligned_df is not None:  # Proceed only if dataset loaded successfully
    print("\n🔍 Preview of aligned_df:")  # Print visual preview label
    display(aligned_df.head(3))  # Display top 3 rows of DataFrame
    print("📑 Columns:", aligned_df.columns.tolist())  # List all column names
    print(
        f"📊 Shape: {aligned_df.shape}"
    )  # Print shape of the DataFrame (rows, columns)

# ------------------------------------------------------------------
# 🧠 4. Aggregate Daily Sentiment using CorrelationSentimentAggregator
# ------------------------------------------------------------------
try:
    aligned_df["cleaned_date"] = pd.to_datetime(  # Convert cleaned_date to datetime
        aligned_df["cleaned_date"], errors="coerce"  # Coerce errors to NaT
    ).dt.normalize()  # Normalize datetime to remove time component

    aligned_df["ticker"] = (
        aligned_df["ticker"].astype(str).str.strip().str.upper()
    )  # Standardize ticker formatting

    required_cols = {
        "ticker",
        "cleaned_date",
        "weighted_sentiment",
    }  # Define required columns for aggregation
    missing = required_cols - set(aligned_df.columns)  # Check for any missing columns
    if missing:  # If any required column is missing
        raise KeyError(
            f"❌ Missing required columns in aligned_df: {missing}"
        )  # Raise error with message

    aggregator = CorrelationSentimentAggregator(
        aligned_df, verbose=True
    )  # Instantiate sentiment aggregator
    final_df = (
        aggregator.aggregate_daily_sentiment(  # Aggregate using exponential weighting
            method="ewm", decay_factor=0.5  # Use λ=0.5 decay for exponential weighting
        )
    )

    print("\n✅ Aggregation succeeded. Preview of final_df:")  # Confirm success
    display(final_df.head(3))  # Show top rows of aggregated result
    print("📦 Shape:", final_df.shape)  # Print shape of aggregated DataFrame

except Exception as e:
    print(
        f"❌ Sentiment aggregation failed: {type(e).__name__} – {e}"
    )  # Catch and print error message
    final_df = None  # Fallback to None if aggregation fails

# ------------------------------------------------------------------
# 📊 5. Run Correlation Analysis using CorrelationAnalyzer
# ------------------------------------------------------------------
try:
    if final_df is None or final_df.empty:  # Validate presence of final_df
        raise ValueError(
            "❌ final_df is empty or not defined."
        )  # Raise error if missing

    all_columns = set(final_df.columns)  # Convert all column names to a set

    sentiment_features = [  # Select only available sentiment columns
        col for col in ["weighted_sentiment", "agg_sentiment_ewm"] if col in all_columns
    ]

    return_features = [  # Select only available return-related columns
        col
        for col in [
            "return_t",
            "forward_return_1d",
            "forward_return_3d",
            "forward_return_5d",
            "return_lag_1d",
            "return_lag_3d",
        ]
        if col in all_columns
    ]

    if not sentiment_features or not return_features:  # Ensure both sets exist
        raise ValueError(
            "❌ Missing sentiment or return columns for correlation."
        )  # Raise error if not

    analyzer = CorrelationAnalyzer(  # Instantiate correlation analyzer
        df=final_df,  # Use aggregated dataset
        methods=["pearson", "spearman", "kendall"],  # Correlation methods to compute
        by_ticker=True,  # Compute per ticker
        verbose=True,  # Enable verbose logging
    )

    correlation_df = analyzer.compute(  # Run correlation computation
        sentiment_cols=sentiment_features,  # Provide sentiment columns
        return_cols=return_features,  # Provide return columns
    )

    print("\n✅ Correlation results (top 10):")  # Confirm success
    display(
        correlation_df.sort_values(by="correlation", ascending=False).head(10)
    )  # Display top correlations

except Exception as e:
    print(
        f"❌ Correlation analysis failed: {type(e).__name__} – {e}"
    )  # Print error message
    correlation_df = None  # Fallback to None on failure

# ------------------------------------------------------------------
# 📈 6. Visualize Correlation Results with CorrelationVisualizer
# ------------------------------------------------------------------
try:
    if correlation_df is None or final_df is None:  # Ensure all necessary data exists
        raise ValueError(
            "❌ Missing correlation_df or final_df for visualization."
        )  # Raise error if not

    visualizer = CorrelationVisualizer(  # Instantiate the visualizer
        correlation_df=correlation_df,  # Provide correlation matrix
        raw_df=final_df,  # Provide original aggregated data
        verbose=True,  # Enable verbose mode
    )

    visualizer.plot_heatmap(
        method="pearson", figsize=(10, 6)
    )  # Generate Pearson heatmap

    visualizer.plot_top_correlations(
        top_n=10, method="spearman"
    )  # Show top 10 strongest correlations (Spearman)

    visualizer.plot_scatter_pairs(  # Generate scatter plot for a specific sentiment-return pair
        sentiment_col="agg_sentiment_ewm",  # Use exponentially weighted sentiment
        return_col="forward_return_1d",  # Correlate with 1-day forward return
        ticker="AAPL",  # Focus on AAPL
        method="kendall",  # Use Kendall correlation
    )

except Exception as e:
    print(
        f"❌ Correlation visualization failed: {type(e).__name__} – {e}"
    )  # Catch and report visualization error

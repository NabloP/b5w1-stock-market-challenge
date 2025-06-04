# Stock Market Challenge Week 1 - 10 Academy

## 🗂 Challenge Context
This repository contains the submission for 10 Academy’s **B5W1: Predicting Price Moves with News Sentiment** challenge. The objective is to analyze how financial news sentiment impacts stock price movements using a combination of NLP, quantitative analysis, and financial indicators.

The project includes:

✨ Sentiment extraction from news headlines

⚖️ Daily stock return computation and correlation analysis

⚙️ Technical indicator calculation using TA-Lib and PyNance

🪨 CI/CD pipeline and modular analysis scripts

📈 Reproducible EDA workflow

## 🔧 Project Setup

To reproduce this environment:

1. Clone the repository:
   ```bash
   git clone https://github.com/NabloP/b5w1-stock-market-challenge.git
   cd b5w1-stock-market-challenge
   ```

2. Create and activate a virtual environment:
   
   **On Windows:**
    ```bash
    python -m venv .venv/stock-market-challenge
    .venv/stock-market-challenge/Scripts/activate
    ```

    **On macOS/Linux:**
    ```bash
    python3 -m venv .venv/stock-market-challenge
    source .venv/stock-market-challenge/bin/activate
    ```

3. Install dependencies
    ```bash
    pip install -r requirements.txt
    ```

## ⚙️ CI/CD (GitHub Actions)

This project uses GitHub Actions for Continuous Integration. On every `push` or `pull_request` event, the following workflow is triggered:

- Checkout repo

- Set up Python 3.10

- Install dependencies from `requirements.txt`

CI workflow is defined at:

    `.github/workflows/unittests.yml`

## 📁 Project Structure

<!-- TREE START -->
📁 Project Structure

solar-challenge-week1/
├── LICENSE
├── README.md
├── requirements.txt
├── .github/
│   └── workflows/
│       ├── unittests.yml
├── data/
│   ├── cleaned_headlines_sample.csv
│   ├── raw_analyst_ratings.csv
│   ├── exports/
│   ├── outputs/
│   │   ├── enriched_aligned_df.csv
│   │   ├── enriched_full_df.csv
│   │   └── plots/
│   └── yfinance_data/
│       ├── AAPL_historical_data.csv
│       ├── AMZN_historical_data.csv
│       ├── GOOG_historical_data.csv
│       ├── META_historical_data.csv
│       ├── MSFT_historical_data.csv
│       ├── NVDA_historical_data.csv
│       ├── TSLA_historical_data.csv
├── notebooks/
│   ├── README.md
│   ├── __init__.py
│   ├── task-1-news-sentiment-eda.ipynb
│   ├── task-2-quantitative-analysis.ipynb
├── scripts/
│   ├── __init__.py
│   ├── eda_orchestrator.py
│   ├── generate_tree.py
│   ├── quantitative_analysis_orchestrator.py
├── src/
│   ├── __init__.py
│   ├── news_loader.py
│   ├── price_data_loader.py
│   ├── sentiment_return_aligner.py
│   ├── signal_visualizer.py
│   ├── technical_indicator_calculator.py
│   ├── text_cleaner.py
│   ├── text_distributions.py
│   ├── text_event_analysis.py
│   ├── text_features.py
│   ├── text_keywords.py
│   ├── text_publishers.py
│   ├── text_publishers_domains.py
│   ├── text_stock_eda.py
│   ├── text_time_series.py
│   ├── text_timestamp_cleaner.py
└── tests/
    ├── __init__.py
<!-- TREE END -->

## ✅ Status
- ☑️ Repository initialized and environment activated
- ☑️ Folder structure scaffolded per 10 Academy guidelines
- ☑️ GitHub Actions CI configured
- ☑️ generate_tree.py integrated to auto-update README
- ☑️ Notebook and script directories prepared for modular development
- ☑️ generate_tree.py integrated to auto-update README
- ☑️ Modular development enforced via `src/`, `scripts/`, and `notebooks/`
- ☑️ Task 1 EDA pipeline (`eda_orchestrator.py`) completed and fully documented
- ☑️ Headline timestamp parsing, text cleaning, feature engineering, and sentiment labeling implemented
- ☑️ Event extraction pipeline (NER, noun phrases, REBEL) with timeline visualization completed
- ☑️ Stock-level headline diagnostics (volume spikes, sentiment flips, per-ticker profiles) operational
- ☑️ Task 2 alignment pipeline (`quantitative_analysis_orchestrator.py`) implemented and validated
- ☑️ Forward return computation (1D, 3D, 5D) and volume–sentiment divergence tagging functional
- ☑️ Technical indicators (SMA, EMA, RSI, MACD, ATR) computed via `TA-Lib`
- ☑️ Per-ticker performance diagnostics (annual return, Sharpe, drawdown) generated
- ☑️ Final outputs exported:
    - `data/outputs/enriched_full_df.csv`
    - `data/outputs/enriched_aligned_df.csv`
- ☑️ Notebook diagnostics and visuals integrated (sentiment vs price, indicators, divergence zones)


## 📦 What's in This Repo

This repository documents the Week 1 challenge for 10 Academy’s AI Mastery Bootcamp. It includes:

- 📁 **Scaffolded directory structure** using best practices for `src/`, `notebooks/`, `scripts/`, and `tests/`

- 🧪 **CI/CD integration** via GitHub Actions for reproducibility and reliability

- 🧹 **README auto-updating** via `scripts/generate_tree.py` to keep documentation aligned with project layout

- 📊 **Modular EDA workflows** for headline sentiment analysis and stock market behavior

- 📚 **Clear Git hygiene** (no committed `.venv` or `.csv`), commit messages and pull request usage

- 🧠 **My Contributions:** All project scaffolding, README setup, automation scripts, and CI configuration were done from scratch by me

## 🧪 Usage

**🔍 How the `eda_orchestrator.py` Pipeline Works**

This script orchestrates the full Task 1 exploratory data analysis (EDA) pipeline for the B5W1 challenge. It covers data loading, text cleaning, sentiment labeling, event extraction, and stock-level headline diagnostics.

📍 Pipeline is intended to be run from the project root. Adjusts automatically if run from `/notebooks/`.

**🔁 Pipeline Steps**

1. Dataset Load

- Loads `raw_analyst_ratings.csv` using a custom loader class.
- Ensures error handling for file not found or format issues.

2. Timestamp Cleaning

- Adds c`leaned_date` by parsing various datetime formats.
- Standardizes timestamps for alignment with OHLCV data.

3. Headline Cleaning

- Applies lowercasing, punctuation removal, HTML stripping, and stopword filtering to headline text.
- Adds `cleaned_headline` column.

4. Feature Extraction

- Computes `word_count` and `headline_length` for textual diagnostics.
- Analyzes top publishers and visualizes their activity by time.

5. Distribution Plots

- Plots histogram distributions of headline lengths and word counts.
- Uses Seaborn styling and verbose labeling for interpretation.

6. Sentiment Labeling

- Flags bullish and bearish keywords.
- Applies VADER for sentence-level polarity scoring.
- Combines multiple sentiment cues into an ensemble label (`ensemble_sentiment`).

7. Event Extraction

- Extracts named entities and noun phrases.
- Uses REBEL for structured event detection.
- Plots most frequent financial event types and their timeline.

8. Stock-Level Diagnostics

- Analyzes headline volumes per ticker and visualizes sentiment share.
- Plots ticker-specific sentiment shifts and headline bursts (e.g. for AAPL).

Outputs are used as enriched input for downstream Task 2 alignment and modeling.


**📈 How the `quantitative_analysis_orchestrator.py` Pipeline Works**

This script runs the Task 2 pipeline for the B5W1 challenge: aligning enriched sentiment signals with historical price data, computing forward returns, calculating technical indicators, and preparing a diagnostic-ready output for each stock.

📍 All outputs are saved to `data/outputs/`. The pipeline is designed to be run from the project root.

**🔁 Pipeline Steps**

1. Load Historical Price Data

- Loads all OHLCV `.csv` files from `data/yfinance_data/`, one per ticker (e.g., `AAPL_historical_data.csv`).
- Validates schema, deduplicates columns, and ensures datetime ordering.

2. Load Enriched Sentiment Data

- Uses `data/cleaned_headlines_sample.csv` from Task 1 as input.
- Parses tickers, converts dates, and aligns schema for join with price data.

3. Sentiment–Price Alignment

- Merges headline sentiment data with OHLCV time series using a ticker-date key.
- Applies exponential decay to aggregate lagged sentiment signals over a configurable window.
- Adds daily forward returns (1-day, 3-day, 5-day) for correlation diagnostics.

4. Volume–Sentiment Divergence Tagging

- Detects abnormal volume spikes using Z-score thresholds.
- Tags days where sentiment signals and volume direction disagree, suggesting hidden divergences.

5. Technical Indicator Calculation

- Computes core TA indicators per ticker using `TA-Lib`:
    - Simple Moving Average (SMA 14)
    - Exponential Moving Average (EMA 14)
    - Relative Strength Index (RSI 14)
    - MACD and Signal line
    - Average True Range (ATR 14)
- Appends results to the aligned dataframe.

6. Hybrid Performance Summary

- Computes per-ticker:
    - Annualized Return
    - Annualized Volatility
    - Sharpe Ratio (risk-free rate = 0)
    - Max Drawdown
- Summary is printed for inspection, not saved.

7. Final Output Saved

 - ✅ `enriched_full_df.csv`: Full dataset before merge
 - ✅ `enriched_aligned_df.csv`: Final sentiment–price–TA dataframe ready for modeling
- 📂 Location: `data/outputs/`

**📊 Visual Diagnostics (Optional)**
While plots were not auto-saved, the script includes logic for:

- Plotting sentiment vs price overlays
- Visualizing technical indicators over time
- Displaying divergence signals

These can be manually run via notebook for exploratory analysis or integrated into future automated runs.

## 🧠 Design Philosophy
This project was developed with a focus on:

- ✅ Modular Python design using classes, helper modules, and runners (clean script folders and testable code)
- ✅ High commenting density to meet AI and human readability expectations
- ✅ Clarity (in folder structure, README, and docstrings)
- ✅ Reproducibility through consistent Git hygiene and generate_tree.py
- ✅ Rubric-alignment (clear deliverables, EDA, and insights)

## 🚀 Author
Nabil Mohamed
AIM Bootcamp Participant
GitHub: [NabloP](https://github.com/NabloP)
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
├── project-tree.txt
├── requirements.txt
├── .github/
│   └── workflows/
│       ├── ci.yml
│       ├── unittests.yml
├── data/
│   ├── benin_clean.csv
│   ├── sierra_leone_clean.csv
│   ├── togo_clean.csv
├── notebooks/
│   ├── benin_eda.ipynb
│   ├── compare_countries.ipynb
│   ├── sierra_leone_eda.ipynb
│   ├── togo_eda.ipynb
├── reports/
│   ├── benin_missing_report.csv
│   ├── benin_summary_stats.csv
│   ├── sierra_leone_missing_report.csv
│   ├── sierra_leone_summary_stats.csv
│   ├── togo_missing_report.csv
│   ├── togo_summary_stats.csv
├── scripts/
│   ├── README.md
│   ├── __init__.py
│   ├── generate_tree.py
│   ├── run_benin_pipeline.py
│   ├── run_compare_pipeline.py
│   ├── run_sierra_leone_pipeline.py
│   ├── run_togo_pipeline.py
├── src/
│   ├── __init__.py
│   ├── clean.py
│   ├── compare_pipeline.py
│   ├── loader.py
│   ├── plots.py
│   ├── report.py
│   ├── Benin/
│   │   ├── __init__.py
│   │   ├── benin-malanville.csv
│   │   ├── load.py
│   ├── Sierra_Leone/
│   │   ├── __init__.py
│   │   ├── load.py
│   │   ├── sierraleone-bumbuna.csv
│   └── Togo/
│       ├── __init__.py
│       ├── load.py
│       ├── togo-dapaong_qc.csv
└── tests/
    ├── __init__.py
<!-- TREE END -->

## ✅ Status
- ☑️ Repository initialized and environment activated
- ☑️ Folder structure scaffolded per 10 Academy guidelines
- ☑️ GitHub Actions CI configured
- ☑️ generate_tree.py integrated to auto-update README
- ☑️ Notebook and script directories prepared for modular development


## 📦 What's in This Repo

This repository documents the Week 1 challenge for 10 Academy’s AI Mastery Bootcamp. It includes:

- 📁 **Scaffolded directory structure** using best practices for `src/`, `notebooks/`, `scripts/`, and `tests/`

- 🧪 **CI/CD integration** via GitHub Actions for reproducibility and reliability

- 🧹 **README auto-updating** via `scripts/generate_tree.py` to keep documentation aligned with project layout

- 📊 **Modular EDA workflows** for headline sentiment analysis and stock market behavior

- 📚 **Clear Git hygiene** (no committed `.venv` or `.csv`), commit messages and pull request usage

- 🧠 **My Contributions:** All project scaffolding, README setup, automation scripts, and CI configuration were done from scratch by me

## 🧪 Usage



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
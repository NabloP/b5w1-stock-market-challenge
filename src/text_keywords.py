"""
text_keywords.py – Enhanced Headline Keyword & Sentiment Analysis Engine
-------------------------------------------------------------------------

A robust, modular system for detecting keyword presence and
computing sentiment scores in financial news headlines using
tokenized fuzzy matching, keyword frequency extraction, VADER and TextBlob sentiment analysis.

Features:
- Fixes joined phrases like 'betterthanexpected' to 'better than expected'
- Tokenizes headline text for precise fuzzy keyword detection
- Supports configurable case sensitivity and fuzzy matching thresholds
- Counts keyword frequencies and plots top keywords by frequency
- Computes VADER and TextBlob sentiment scores and ensembles them
- Assigns sentiment labels with confidence scoring and mixed signal detection
- Filters headlines by stock ticker symbol
- Verbose diagnostics and null-safe operations
- Clean, modular OOP design enabling extensibility

Author: Nabil Mohamed
"""

from typing import List, Optional, Tuple, Dict
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from rapidfuzz import fuzz, process

# Download required NLTK data quietly if not present
nltk.download("punkt", quiet=True)
nltk.download("vader_lexicon", quiet=True)


class KeywordSentimentAnalyzer:
    """
    Modular keyword presence and sentiment analyzer for financial news headlines.

    Parameters:
    -----------
    keywords : Optional[List[str]]
        List of keywords/phrases to detect. Can be empty if only running sentiment analysis.
    case_sensitive : bool
        Whether keyword matching is case sensitive (default False).
    text_col : str
        Column name containing the text data (default 'headline').
    fuzzy_threshold : int
        Minimum fuzzy matching score (0-100) for keyword detection (default 85).
    verbose : bool
        Enable detailed logging and diagnostics (default True).
    joined_phrase_list : Optional[List[str]]
        List of known multi-word phrases to fix if joined without spaces (default None).
    """

    def __init__(
        self,
        keywords: Optional[List[str]] = None,
        case_sensitive: bool = False,
        text_col: str = "headline",
        fuzzy_threshold: int = 85,
        verbose: bool = True,
        joined_phrase_list: Optional[List[str]] = None,
    ):
        self.keywords = keywords or []
        self.case_sensitive = case_sensitive
        self.text_col = text_col
        self.fuzzy_threshold = fuzzy_threshold
        self.verbose = verbose
        self.joined_phrase_list = joined_phrase_list or []

        # Normalize keywords for matching per case sensitivity
        self._normalized_keywords = (
            self.keywords if self.case_sensitive else [k.lower() for k in self.keywords]
        )

        # Initialize VADER sentiment analyzer instance
        self._vader = SentimentIntensityAnalyzer()

        # Cache for keyword frequency DataFrame
        self.freq_table: Optional[pd.DataFrame] = None

    def fix_joined_phrases(self, text: str) -> str:
        """
        Replace known joined multi-word phrases in the text with spaced versions.
        For example, 'betterthanexpected' → 'better than expected'.
        """
        for phrase in self.joined_phrase_list:
            joined = phrase.replace(" ", "")
            pattern = re.compile(re.escape(joined), re.IGNORECASE)
            text = pattern.sub(phrase, text)
        return text

    def _normalize_text(self, text: str) -> str:
        """Normalize text for case sensitivity and type safety."""
        if not isinstance(text, str):
            return ""
        return text if self.case_sensitive else text.lower()

    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text using NLTK's tokenizer; fallback to split."""
        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = text.split()
        return tokens

    def _fuzzy_keyword_match(self, token: str) -> bool:
        """
        Fuzzy match token against keywords with configured threshold.
        Returns True if a match above threshold is found.
        """
        if not self._normalized_keywords:
            return False
        best_match = process.extractOne(
            token,
            self._normalized_keywords,
            scorer=fuzz.partial_ratio,
            score_cutoff=self.fuzzy_threshold,
        )
        return best_match is not None

    def flag_keywords(
        self, df: pd.DataFrame, output_col: str = "keyword_flag"
    ) -> pd.DataFrame:
        """
        Flag rows where any keyword fuzzily matches a token in the text column.

        Adds a boolean column to the DataFrame.

        Args:
            df: Input DataFrame
            output_col: Name of output boolean flag column

        Returns:
            DataFrame with flag column added.
        """
        if self.text_col not in df.columns:
            raise KeyError(f"Column '{self.text_col}' not found in DataFrame.")

        # Ensure text column has no NaNs and is string type
        df[self.text_col] = df[self.text_col].fillna("").astype(str)

        def match_func(text: str) -> bool:
            # Fix joined phrases before tokenization
            text = self.fix_joined_phrases(text)
            norm_text = self._normalize_text(text)
            tokens = self._tokenize_text(norm_text)
            # Return True if any token fuzzily matches a keyword
            return any(self._fuzzy_keyword_match(tok) for tok in tokens)

        # Apply token-level fuzzy matching for keywords across all rows
        df[output_col] = df[self.text_col].apply(match_func)

        # Verbose logging of matches
        if self.verbose:
            hits = df[output_col].sum()
            total = len(df)
            print(
                f"✅ '{output_col}' column added with {hits:,} matches out of {total:,} rows."
            )

        return df

    def compute_keyword_frequencies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute fuzzy keyword frequency counts over the text corpus.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with columns ['keyword', 'count'] sorted descending.
        """
        if not self.keywords:
            if self.verbose:
                print("⚠️ No keywords defined, skipping frequency computation.")
            return pd.DataFrame(columns=["keyword", "count"])

        if self.text_col not in df.columns:
            raise KeyError(f"Column '{self.text_col}' not found in DataFrame.")

        df[self.text_col] = df[self.text_col].fillna("").astype(str)

        counter = {kw: 0 for kw in self.keywords}

        for text in df[self.text_col]:
            text = self.fix_joined_phrases(text)
            norm_text = self._normalize_text(text)
            tokens = self._tokenize_text(norm_text)
            for tok in tokens:
                for kw in self.keywords:
                    norm_kw = kw if self.case_sensitive else kw.lower()
                    if fuzz.partial_ratio(tok, norm_kw) >= self.fuzzy_threshold:
                        counter[kw] += 1

        freq_df = pd.DataFrame(
            sorted(counter.items(), key=lambda x: x[1], reverse=True),
            columns=["keyword", "count"],
        )

        self.freq_table = freq_df

        if self.verbose:
            print(f"✅ Computed fuzzy keyword frequencies for {len(freq_df)} keywords:")
            print(freq_df.head(10).to_string(index=False))

        return freq_df

    def plot_keyword_frequencies(
        self, top_n: int = 20, figsize: Tuple[int, int] = (12, 6)
    ) -> None:
        """
        Plot horizontal bar chart of top N keywords by frequency.

        Args:
            top_n: Number of top keywords to plot.
            figsize: Figure size.

        Raises:
            ValueError: If frequency table is empty or not computed.
        """
        if self.freq_table is None or self.freq_table.empty:
            raise ValueError(
                "Frequency table not computed. Run compute_keyword_frequencies first."
            )

        top_df = self.freq_table.head(top_n)

        plt.figure(figsize=figsize)
        sns.barplot(x="count", y="keyword", data=top_df, palette="viridis")
        plt.title(f"Top {top_n} Keywords by Frequency")
        plt.xlabel("Count")
        plt.ylabel("Keyword")
        plt.tight_layout()
        plt.show()

        if self.verbose:
            print(f"📊 Plotted top {top_n} keywords by frequency.")

    def run_vader_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute VADER sentiment scores and label sentiment as bullish/bearish/neutral.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with added columns 'vader_scores', 'vader_compound', 'vader_sentiment'.
        """
        if self.text_col not in df.columns:
            raise KeyError(f"Column '{self.text_col}' not found in DataFrame.")

        texts = df[self.text_col].fillna("").astype(str)

        df["vader_scores"] = texts.apply(lambda t: self._vader.polarity_scores(t))
        df["vader_compound"] = df["vader_scores"].apply(lambda s: s["compound"])

        def sentiment_label(score: float) -> str:
            if score >= 0.05:
                return "bullish"
            elif score <= -0.05:
                return "bearish"
            else:
                return "neutral"

        df["vader_sentiment"] = df["vader_compound"].apply(sentiment_label)

        if self.verbose:
            counts = df["vader_sentiment"].value_counts()
            print(f"✅ VADER sentiment labels assigned: {counts.to_dict()}")

        return df

    def run_textblob_sentiment(
        self, df: pd.DataFrame, output_col: str = "textblob_polarity"
    ) -> pd.DataFrame:
        """
        Compute TextBlob polarity scores for text column.

        Args:
            df: Input DataFrame
            output_col: Column name to store polarity scores

        Returns:
            DataFrame with added polarity scores column.
        """
        if self.text_col not in df.columns:
            raise KeyError(f"Column '{self.text_col}' not found in DataFrame.")

        texts = df[self.text_col].fillna("").astype(str)

        df[output_col] = texts.apply(lambda t: TextBlob(t).sentiment.polarity)

        if self.verbose:
            counts = df[output_col].describe()
            print(f"✅ TextBlob polarity scores computed: {counts}")

        return df

    def run_ensemble_sentiment(
        self,
        df: pd.DataFrame,
        vader_col: str = "vader_compound",
        textblob_col: str = "textblob_polarity",
        bullish_flag_col: str = "bullish_flag",
        bearish_flag_col: str = "bearish_flag",
        output_label_col: str = "ensemble_sentiment",
        output_confidence_col: str = "ensemble_confidence",
        mixed_signal_col: str = "mixed_signals_flag",
    ) -> pd.DataFrame:
        """
        Combine keyword flags and VADER/TextBlob sentiment scores into an ensemble sentiment label.

        Rules:
        - If bullish_flag=True and bearish_flag=False → sentiment = bullish
        - If bearish_flag=True and bullish_flag=False → sentiment = bearish
        - If both bullish_flag and bearish_flag are True → mixed_signals_flag=True and sentiment=neutral
        - Else fallback to VADER/TextBlob ensemble sentiment

        Confidence:
        - 1.0 if VADER and TextBlob agree and no mixed signals
        - 0.5 if one is neutral and the other positive or negative
        - 0.0 if VADER and TextBlob disagree or mixed signals

        Args:
            df: Input DataFrame with sentiment and flag columns.
            vader_col: VADER compound score column name.
            textblob_col: TextBlob polarity score column name.
            bullish_flag_col: Column with bullish keyword flags.
            bearish_flag_col: Column with bearish keyword flags.
            output_label_col: Output column for ensemble sentiment label.
            output_confidence_col: Output column for confidence score.
            mixed_signal_col: Output column for mixed signals flag.

        Returns:
            DataFrame with new ensemble sentiment, confidence, and mixed signal columns.
        """
        # Defensive checks for required columns
        for col in [
            vader_col,
            textblob_col,
            bullish_flag_col,
            bearish_flag_col,
        ]:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")

        def polarity_to_label(score: float) -> str:
            if score >= 0.05:
                return "bullish"
            elif score <= -0.05:
                return "bearish"
            else:
                return "neutral"

        def ensemble_logic(row) -> Tuple[str, float, bool]:
            bullish = row[bullish_flag_col]
            bearish = row[bearish_flag_col]

            # Detect mixed signals
            mixed = bullish and bearish

            # Apply rules in order
            if mixed:
                return (
                    "neutral",
                    0.0,
                    True,
                )  # Mixed signals: neutral sentiment, flag set
            if bullish and not bearish:
                return ("bullish", 1.0, False)
            if bearish and not bullish:
                return ("bearish", 1.0, False)

            # Fallback to VADER/TextBlob ensemble
            vader_label = polarity_to_label(row[vader_col])
            tb_label = polarity_to_label(row[textblob_col])

            if vader_label == tb_label and vader_label != "neutral":
                return (vader_label, 1.0, False)
            if "neutral" in (vader_label, tb_label):
                label = vader_label if vader_label != "neutral" else tb_label
                return (label, 0.5, False)
            # Conflict in labels means low confidence and neutral sentiment
            return ("neutral", 0.0, False)

        # Apply ensemble logic row-wise
        results = df.apply(ensemble_logic, axis=1, result_type="expand")
        df[output_label_col] = results[0]
        df[output_confidence_col] = results[1]
        df[mixed_signal_col] = results[2]

        if self.verbose:
            counts = df[output_label_col].value_counts()
            mixed_count = df[mixed_signal_col].sum()
            print(f"✅ Ensemble sentiment assigned: {counts.to_dict()}")
            print(f"ℹ️ Mixed signals detected in {mixed_count} rows.")

        return df


class HeadlineFilter:
    """
    Filters headlines by stock ticker.

    Parameters:
    -----------
    stock_col : str
        Column name containing stock ticker symbols (default: 'stock').
    """

    def __init__(self, stock_col: str = "stock"):
        self.stock_col = stock_col

    def filter_by_stock(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Filter DataFrame rows by stock ticker symbol.

        Args:
            df: Input DataFrame
            ticker: Stock ticker symbol to filter by.

        Returns:
            Filtered DataFrame with only rows for the ticker.
        """
        if self.stock_col not in df.columns:
            raise KeyError(f"Column '{self.stock_col}' not found in DataFrame.")
        return df[df[self.stock_col].str.upper() == ticker.upper()]


# -------------------------
# Extended financial bullish and bearish lexicons with jargon
# -------------------------

BULLISH_TERMS = [
    "upgrade",
    "strong buy",
    "outperform",
    "bullish",
    "beats expectations",
    "positive outlook",
    "raises guidance",
    "accelerates",
    "record high",
    # Extended financial jargon:
    "eps beat",
    "earnings beat",
    "price target raised",
    "guidance increased",
    "upside potential",
    "margin expansion",
    "revenue growth",
    "market rally",
    "share buyback",
    "dividend increase",
    "analyst upgrade",
    "improved outlook",
]

BEARISH_TERMS = [
    "downgrade",
    "underperform",
    "misses expectations",
    "bearish",
    "negative outlook",
    "cuts forecast",
    "slows",
    "plunges",
    "record low",
    # Extended financial jargon:
    "eps miss",
    "earnings miss",
    "price target lowered",
    "guidance cut",
    "downside risk",
    "margin compression",
    "revenue decline",
    "market selloff",
    "share dilution",
    "dividend cut",
    "analyst downgrade",
    "weak guidance",
]


# -------------------------
# Convenience wrapper functions for quick usage
# -------------------------


def flag_bullish(df: pd.DataFrame, column: str = "headline") -> pd.DataFrame:
    """
    Convenience wrapper to flag bullish keywords in a DataFrame column.
    """
    return KeywordSentimentAnalyzer(
        keywords=BULLISH_TERMS, case_sensitive=False, text_col=column, verbose=True
    ).flag_keywords(df, output_col="bullish_flag")


def flag_bearish(df: pd.DataFrame, column: str = "headline") -> pd.DataFrame:
    """
    Convenience wrapper to flag bearish keywords in a DataFrame column.
    """
    return KeywordSentimentAnalyzer(
        keywords=BEARISH_TERMS, case_sensitive=False, text_col=column, verbose=True
    ).flag_keywords(df, output_col="bearish_flag")

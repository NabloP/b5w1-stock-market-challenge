"""
text_cleaner.py – Headline Text Cleaning Module
-----------------------------------------------

Encapsulates a configurable pipeline for cleaning financial news headlines.
Supports:
- Lowercasing
- HTML tag and special character removal
- Whitespace normalization
- Optional punctuation and stopword removal

Design Principles:
- Full OOP encapsulation with meaningful config
- Safe null handling and verbose diagnostics
- Plug-and-play compatibility with other EDA modules

Author: Nabil Mohamed
"""

import pandas as pd  # Core data structure
import re  # Regex utilities for pattern removal
import string  # Punctuation lists
from typing import Optional, List  # Type hinting
from nltk.corpus import stopwords  # Built-in stopword list
from nltk.tokenize import (
    word_tokenize,
)  # Word-level tokenization (used in stopword filtering)


# ------------------------------------------------------------------------------
# 🧼 TextCleaner – Configurable, Production-Grade Cleaning Pipeline
# ------------------------------------------------------------------------------


class TextCleaner:
    """
    A robust and modular text cleaner for financial headlines.

    Parameters:
    -----------
    text_col : str
        Name of column containing raw text (default: 'headline')
    output_col : str
        Name of new column for cleaned text (default: 'cleaned_headline')
    lowercase : bool
        Whether to convert text to lowercase (default: True)
    remove_html : bool
        Whether to remove HTML tags and escape sequences (default: True)
    remove_punctuation : bool
        Whether to strip punctuation symbols (default: False)
    remove_stopwords : bool
        Whether to remove stopwords using NLTK (default: False)
    verbose : bool
        Whether to print summaries and sample transformations (default: True)
    """

    def __init__(
        self,
        text_col: str = "headline",
        output_col: str = "cleaned_headline",
        lowercase: bool = True,
        remove_html: bool = True,
        remove_punctuation: bool = False,
        remove_stopwords: bool = False,
        verbose: bool = True,
    ):
        self.text_col = text_col  # Column containing raw text
        self.output_col = output_col  # Output column to write cleaned text to
        self.lowercase = lowercase  # Flag for lowercasing text
        self.remove_html = remove_html  # Flag for removing HTML tags/entities
        self.remove_punctuation = remove_punctuation  # Flag for punctuation stripping
        self.remove_stopwords = remove_stopwords  # Flag for removing stopwords
        self.verbose = verbose  # Toggle for logging/inspection

        # Preload stopwords set only if enabled
        self._stopwords = (
            set(stopwords.words("english")) if self.remove_stopwords else set()
        )

    def _validate_input(self, df: pd.DataFrame):
        """
        Check if the specified input column exists in the DataFrame.
        Raise a clear and user-friendly KeyError if not.
        """
        if self.text_col not in df.columns:
            raise KeyError(f"🛑 Column '{self.text_col}' not found in DataFrame.")

    def _clean_text(self, text: str) -> str:
        """
        Run all configured cleaning operations on a single string.
        Handles casing, punctuation, HTML tags, stopwords, and whitespace.
        """
        if not isinstance(text, str):  # Defensive fallback for malformed rows
            return ""

        if self.lowercase:  # Optionally convert to lowercase
            text = text.lower()

        if self.remove_html:  # Optionally remove HTML tags/entities
            text = re.sub(r"<.*?>", " ", text)  # Remove tags like <div>
            text = re.sub(r"&\w+;", " ", text)  # Remove HTML-encoded chars like &nbsp;

        if self.remove_punctuation:  # Strip out punctuation if enabled
            text = text.translate(str.maketrans("", "", string.punctuation))

        if self.remove_stopwords:  # Tokenize and filter out stopwords
            tokens = word_tokenize(text)  # Word-level tokenize
            text = " ".join([w for w in tokens if w not in self._stopwords])

        text = re.sub(r"\s+", " ", text).strip()  # Normalize whitespace

        return text

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the configured text cleaning logic to a DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            The input DataFrame containing raw text.

        Returns:
        --------
        pd.DataFrame
            A new column is added with cleaned text.
        """
        self._validate_input(df)  # Confirm the input column exists
        df[self.text_col] = df[self.text_col].fillna(
            ""
        )  # Replace NaNs with empty strings

        df[self.output_col] = df[self.text_col].apply(
            self._clean_text
        )  # Clean each row

        if self.verbose:  # Optional diagnostic logging
            print(f"✅ Column '{self.output_col}' successfully added.")
            print("🔍 Sample Before → After:")
            for i in range(min(3, len(df))):  # Show up to 3 sample transformations
                raw = df[self.text_col].iloc[i]
                cleaned = df[self.output_col].iloc[i]
                print(f"  - {raw}\n    → {cleaned}")
            print(f"🧼 Nulls replaced: {(df[self.text_col] == '').sum()}")

        return df

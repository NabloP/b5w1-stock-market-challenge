"""
text_keywords.py – Headline Keyword Flagging Engine
----------------------------------------------------

Elegant, modular system for detecting presence of keywords or phrases
within financial news headlines. Supports:

- Case sensitivity toggle
- Bullish/bearish keyword presets
- Clean diagnostic logging
- Null-safe operation

Author: Nabil Mohamed
"""

import pandas as pd  # DataFrame handling
from typing import List, Optional  # Type hinting for lists and optional parameters


# ------------------------------------------------------------------------------
# 🧠 KeywordFlagger – Encapsulated Matching Logic
# ------------------------------------------------------------------------------


class KeywordFlagger:
    """
    Keyword presence detector for text-based DataFrames.

    Parameters:
    -----------
    keywords : list of str
        List of keywords or phrases to flag.
    case_sensitive : bool
        Whether to perform case-sensitive matching (default: False).
    column : str
        Text column to evaluate (default: 'headline').
    output : str
        Name of the new boolean flag column (default: 'keyword_flag').
    verbose : bool
        Whether to log matching statistics (default: True).
    """

    def __init__(  # KeywordFlagger constructor
        self,  # self reference
        keywords: List[str],  # List of keywords to search for
        case_sensitive: bool = False,  # Whether to match case exactly
        column: str = "headline",  # Column containing text to search (default: 'headline')
        output: str = "keyword_flag",  # Name of the output flag column (default: 'keyword_flag')
        verbose: bool = True,  # Whether to print matching statistics (default: True)
    ):
        if not keywords:  # Check if keywords list is empty
            raise ValueError("🛑 Keyword list must not be empty.")

        self.keywords = keywords  # Store keywords
        self.case_sensitive = case_sensitive  # Store case sensitivity flag
        self.column = column  # Store column name to search
        self.output = output  # Store name of output flag column
        self.verbose = verbose  # Store verbosity flag for diagnostics

    def apply(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:  # Apply keyword matching to DataFrame
        """
        Flags rows in the DataFrame where the text contains any keyword.

        Returns:
        --------
        pd.DataFrame : DataFrame with added binary flag column.
        """

        if self.column not in df.columns:  # Check if specified column exists
            raise KeyError(
                f"🛑 Required column '{self.column}' not found in DataFrame."
            )

        # Null safety
        df[self.column] = df[self.column].fillna("")  # Fill NaNs with empty strings

        # Text normalization
        search_text = (  # Normalize text for matching
            df[self.column].str.lower() if not self.case_sensitive else df[self.column]
        )
        keyword_set = (  # Prepare keywords for matching
            [k.lower() for k in self.keywords]
            if not self.case_sensitive
            else self.keywords
        )

        # Match logic
        df[self.output] = search_text.apply(
            lambda x: any(k in x for k in keyword_set)
        )  # Check if any keyword is present

        # Diagnostics
        if self.verbose:  # Print summary statistics if verbose mode is enabled
            hits = df[self.output].sum()  # Count matches
            print(f"✅ '{self.output}' added: {hits:,} matches out of {len(df):,} rows")

        return df


# ------------------------------------------------------------------------------
# 🚀 Preset Utilities (Optional Shortcuts)
# ------------------------------------------------------------------------------

BULLISH_TERMS = [  # List of bullish sentiment phrases
    "upgrade",
    "strong buy",
    "outperform",
    "bullish",
    "beats expectations",
    "positive outlook",
    "raises guidance",
    "accelerates",
    "record high",
]

BEARISH_TERMS = [  # List of bearish sentiment phrases
    "downgrade",
    "underperform",
    "misses expectations",
    "bearish",
    "negative outlook",
    "cuts forecast",
    "slows",
    "plunges",
    "record low",
]


def flag_bullish(
    df: pd.DataFrame, column: str = "headline"
) -> pd.DataFrame:  # Flags rows with bullish sentiment phrases
    """
    Flags rows with bullish sentiment phrases in a given text column.
    """
    return KeywordFlagger(  # Create a KeywordFlagger instance for bullish terms
        keywords=BULLISH_TERMS,  # Use predefined bullish terms
        column=column,  # Specify the text column to search
        output="bullish_flag",  # Name of the output flag column
        verbose=True,  # Enable verbose logging
    ).apply(
        df
    )  # Apply the flagging logic to the DataFrame


def flag_bearish(
    df: pd.DataFrame, column: str = "headline"
) -> pd.DataFrame:  # Flags rows with bearish sentiment phrases
    """
    Flags rows with bearish sentiment phrases in a given text column.
    """
    return KeywordFlagger(  # Create a KeywordFlagger instance for bearish terms
        keywords=BEARISH_TERMS,  # Use predefined bearish terms
        column=column,  # Specify the text column to search
        output="bearish_flag",  # Name of the output flag column
        verbose=True,  # Enable verbose logging
    ).apply(
        df
    )  # Apply the flagging logic to the DataFrame

"""
text_features.py – Headline Text Feature Engineering
-----------------------------------------------------

Encapsulates a reusable, object-oriented class for extracting text-based features
from financial news headlines:

- Character length
- Word count (via NLTK tokenization)
- Configurable input/output columns
- Verbose diagnostics and null-safe operation

Author: Nabil Mohamed
"""

import pandas as pd  # For DataFrame operations
from nltk.tokenize import word_tokenize  # For word-level tokenization
from typing import Optional  # For flexible type hinting


# ------------------------------------------------------------------------------
# 🧠 TextFeatureExtractor – Headline Text Feature Class
# ------------------------------------------------------------------------------


class TextFeatureExtractor:
    """
    A reusable utility class for extracting basic textual features from
    a column of financial news headlines.

    Features extracted:
    - Character length
    - Word count (via NLTK)

    Example Usage:
    --------------
        extractor = TextFeatureExtractor(headline_col="headline", verbose=True)
        df = extractor.transform(df)
    """

    def __init__(
        self,
        headline_col: str = "headline",  # Name of the column containing text
        verbose: bool = True,  # Whether to log output during processing
        length_col: str = "headline_length",  # Output column name for character count
        word_col: str = "word_count",  # Output column name for token count
    ):
        # Store initialization parameters as instance attributes
        self.headline_col = headline_col
        self.verbose = verbose
        self.length_col = length_col
        self.word_col = word_col

    def _validate_input(self, df: pd.DataFrame) -> None:
        """
        Ensure that the required text column is present in the DataFrame.
        Raises KeyError with helpful debug info if not found.
        """
        if self.headline_col not in df.columns:
            # Provide a detailed error message including available columns
            raise KeyError(
                f"🛑 Column '{self.headline_col}' not found in DataFrame.\n"
                f"Available columns: {df.columns.tolist()}"
            )

    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute character and word counts for the target column.

        Returns:
        --------
        df : pd.DataFrame
            The same DataFrame with two new feature columns added.
        """
        # Fill any null or NaN values in the headline column with empty strings
        df[self.headline_col] = df[self.headline_col].fillna("")

        # Compute length of each headline (in characters)
        df[self.length_col] = df[self.headline_col].apply(len)

        # Compute word count using NLTK tokenizer (handles punctuation, spacing)
        df[self.word_col] = df[self.headline_col].apply(lambda x: len(word_tokenize(x)))

        return df  # Return the enriched DataFrame

    def _print_diagnostics(self, df: pd.DataFrame) -> None:
        """
        Print summary statistics for the extracted features, if verbose is enabled.
        Useful for sanity checks during EDA.
        """
        print("✅ Text features added:")  # Confirm that the operation completed
        print(f"📊 Input column: '{self.headline_col}'")  # Show source column
        # Display descriptive statistics for character lengths
        print(
            f"🔠 {self.length_col} (chars):", df[self.length_col].describe().to_dict()
        )
        # Display descriptive statistics for token counts
        print(f"🗨️ {self.word_col} (tokens):", df[self.word_col].describe().to_dict())

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Public method to apply the full feature extraction pipeline.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with at least one text column.

        Returns:
        --------
        pd.DataFrame
            Same DataFrame with two new feature columns added.
        """
        self._validate_input(df)  # Step 1: Ensure required column exists
        df = self._add_features(df)  # Step 2: Apply the feature computations
        if self.verbose:
            self._print_diagnostics(df)  # Step 3: Log stats if enabled
        return df  # Return the transformed DataFrame

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convenience alias for transform() to align with sklearn-style pipelines.

        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame.

        Returns:
        --------
        pd.DataFrame
            Transformed DataFrame with new features.
        """
        return self.transform(df)

    def __str__(self) -> str:
        """
        Human-readable representation of this instance.
        Useful when logging or debugging in Jupyter or CLI.
        """
        return (
            f"TextFeatureExtractor(input='{self.headline_col}', "
            f"outputs=('{self.length_col}', '{self.word_col}'), verbose={self.verbose})"
        )

    def __repr__(self) -> str:
        """
        Developer-friendly version of __str__.
        Ensures consistent object rendering in notebooks and logs.
        """
        return self.__str__()

"""
text_publishers.py – Publisher Activity Analyzer
------------------------------------------------

World-class publisher analytics engine for financial news datasets.

This module:
- Computes publisher article frequency
- Visualizes top publishers using horizontal bar charts
- Stores state (frequency table) for reuse
- Offers toggleable verbosity and null-safe handling
- Uses OOP beyond mere wrapping—supports future extensions (e.g., domain grouping)

Author: Nabil Mohamed
"""

import pandas as pd  # DataFrame manipulation
import matplotlib.pyplot as plt  # Visualization
import seaborn as sns  # Enhanced charting


# --------------------------------------------------------------------------
# 🏛️ PublisherAnalyzer – State-Preserving Class for Publisher Diagnostics
# --------------------------------------------------------------------------


class PublisherAnalyzer:
    """
    Class to analyze and visualize publisher activity in text-based datasets.

    Parameters:
    -----------
    publisher_col : str
        Name of the column containing publisher names (default: 'publisher').
    verbose : bool
        If True, prints human-readable diagnostic messages.
    """

    def __init__(self, publisher_col: str = "publisher", verbose: bool = True):
        self.publisher_col = publisher_col  # Store the target column
        self.verbose = verbose  # Enable logging if needed
        self._frequency_table = None  # Internal cache to store computed frequency

    def analyze(self, df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """
        Computes frequency of articles by publisher and stores results.

        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame containing a publisher column.
        top_n : int
            Number of top publishers to display in the summary (default: 10).

        Returns:
        --------
        pd.DataFrame
            A DataFrame with publisher names and corresponding article counts.

        Raises:
        -------
        KeyError:
            If the specified column does not exist in the input DataFrame.
        """
        # Step 1: Validation – ensure the target column exists
        if self.publisher_col not in df.columns:
            raise KeyError(
                f"🛑 Column '{self.publisher_col}' not found in the DataFrame."
            )

        # Step 2: Fill missing values with "Unknown" for safe aggregation
        publishers = df[self.publisher_col].fillna("Unknown")

        # Step 3: Compute value counts (frequency) and format into DataFrame
        freq = publishers.value_counts().reset_index()
        freq.columns = [self.publisher_col, "article_count"]

        # Step 4: Store for future use
        self._frequency_table = freq.copy()

        # Step 5: Optional diagnostics
        if self.verbose:
            print(f"✅ Found {freq.shape[0]} unique publishers.")
            print(f"📈 Top {top_n} publishers by article count:")
            display(freq.head(top_n))

        return freq

    def plot_top_publishers(self, top_n: int = 10, figsize: tuple = (10, 5)) -> None:
        """
        Plots a bar chart of the top N publishers by article volume.

        Parameters:
        -----------
        top_n : int
            Number of top publishers to include in the plot.
        figsize : tuple
            Size of the figure (default: (10, 5)).

        Raises:
        -------
        ValueError:
            If `analyze()` has not been called yet.
        """
        # Step 1: Ensure frequency table is available
        if self._frequency_table is None:
            raise ValueError(
                "🛑 No frequency table found. Call `.analyze(df)` before plotting."
            )

        # Step 2: Extract top N rows for plotting
        plot_data = self._frequency_table.head(top_n)

        # Step 3: Create horizontal bar plot
        plt.figure(figsize=figsize)
        sns.barplot(
            x="article_count", y=self.publisher_col, data=plot_data, palette="Blues_d"
        )
        plt.title(f"🏢 Top {top_n} Publishers by Article Volume")
        plt.xlabel("Number of Articles")
        plt.ylabel("Publisher")
        plt.tight_layout()
        plt.show()

        # Step 4: Optional diagnostics
        if self.verbose:
            print(f"📊 Chart rendered for top {top_n} publishers.")

    def get_frequency_table(self) -> pd.DataFrame:
        """
        Returns the most recently computed frequency table.

        Returns:
        --------
        pd.DataFrame
            Publisher frequency table.

        Raises:
        -------
        ValueError:
            If `.analyze()` hasn't been called yet.
        """
        if self._frequency_table is None:
            raise ValueError("🛑 No analysis results found. Run `.analyze(df)` first.")
        return self._frequency_table.copy()

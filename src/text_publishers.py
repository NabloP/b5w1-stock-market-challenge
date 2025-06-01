"""
text_publishers.py – Publisher Activity Analyzer
------------------------------------------------

World-class publisher analytics engine for financial news datasets.

This module:
- Computes publisher article frequency
- Analyzes and aggregates article counts by publication date
- Visualizes top publishers using horizontal bar charts
- Plots article counts over time with line charts
- Stores state (frequency tables) for reuse
- Offers toggleable verbosity and null-safe handling
- Uses OOP beyond mere wrapping—supports future extensions (e.g., domain grouping)

Author: Nabil Mohamed
"""

import pandas as pd  # Data manipulation library
import matplotlib.pyplot as plt  # Plotting library
import seaborn as sns  # Enhanced plotting aesthetics


# --------------------------------------------------------------------------
# 🏛️ PublisherAnalyzer – State-Preserving Class for Publisher Diagnostics
# --------------------------------------------------------------------------


class PublisherAnalyzer:
    """
    Class to analyze and visualize publisher activity and publication dates in text-based datasets.

    Parameters:
    -----------
    publisher_col : str
        Name of the column containing publisher names (default: 'publisher').
    verbose : bool
        If True, prints human-readable diagnostic messages.
    """

    def __init__(self, publisher_col: str = "publisher", verbose: bool = True):
        self.publisher_col = (
            publisher_col  # Store column name to analyze publisher data
        )
        self.verbose = verbose  # Flag to control verbosity of output
        self._frequency_table = None  # Internal cache for publisher frequency DataFrame
        self._publication_counts = (
            None  # Internal cache for article counts by date DataFrame
        )

    # ----------------------------------------------------------------------
    # Internal helper for environment-agnostic display of DataFrames
    # ----------------------------------------------------------------------

    def _display(self, df: pd.DataFrame):
        """
        Environment-agnostic display function.

        Attempts to use IPython.display.display if available; falls back to print.

        Args:
            df (pd.DataFrame): DataFrame to display
        """
        try:
            from IPython.display import display as ipy_display  # Try Jupyter display

            ipy_display(df)
        except ImportError:
            print(df)  # Fallback to plain print for non-Jupyter environments

    # ----------------------------------------------------------------------
    # Publisher Frequency Analysis
    # ----------------------------------------------------------------------

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
            DataFrame with publisher names and article counts.

        Raises:
        -------
        KeyError:
            If the specified column does not exist in the input DataFrame.
        """
        # Ensure the publisher column exists in the DataFrame
        if self.publisher_col not in df.columns:
            raise KeyError(
                f"🛑 Column '{self.publisher_col}' not found in the DataFrame."
            )

        # Replace null values with 'Unknown' to avoid errors during counting
        publishers = df[self.publisher_col].fillna("Unknown")

        # Calculate counts of each publisher and reset index for nicer DataFrame
        freq = publishers.value_counts().reset_index()
        freq.columns = [
            self.publisher_col,
            "article_count",
        ]  # Rename columns for clarity

        # Cache the frequency DataFrame internally for later reuse
        self._frequency_table = freq.copy()

        # If verbosity enabled, print total unique publishers and sample top counts
        if self.verbose:
            print(f"✅ Found {freq.shape[0]} unique publishers.")
            print(f"📈 Top {top_n} publishers by article count:")
            self._display(freq.head(top_n))  # Use environment-agnostic display

        return freq  # Return frequency DataFrame to caller

    # ----------------------------------------------------------------------
    # Publisher Frequency Plotting
    # ----------------------------------------------------------------------

    def plot_top_publishers(self, top_n: int = 10, figsize: tuple = (10, 5)) -> None:
        """
        Plot a horizontal bar chart of the top N publishers by article volume.

        Parameters:
        -----------
        top_n : int
            Number of top publishers to include in the plot.
        figsize : tuple
            Figure size, default (10, 5).

        Raises:
        -------
        ValueError:
            If .analyze() has not been called and frequency data is missing.
        """
        # Check internal cache exists
        if self._frequency_table is None:
            raise ValueError(
                "🛑 No frequency table found. Call `.analyze(df)` before plotting."
            )

        # Select top N publishers for plotting
        plot_data = self._frequency_table.head(top_n)

        # Create a matplotlib figure with specified size
        plt.figure(figsize=figsize)

        # Use seaborn to create a horizontal bar plot
        sns.barplot(
            x="article_count", y=self.publisher_col, data=plot_data, palette="Blues_d"
        )
        plt.title(f"🏢 Top {top_n} Publishers by Article Volume")  # Title for the chart
        plt.xlabel("Number of Articles")  # X-axis label
        plt.ylabel("Publisher")  # Y-axis label
        plt.tight_layout()  # Adjust layout to prevent clipping
        plt.show()  # Display the plot

        # Verbose confirmation message if enabled
        if self.verbose:
            print(f"📊 Chart rendered for top {top_n} publishers.")

    # ----------------------------------------------------------------------
    # Retrieve Cached Publisher Frequency Table
    # ----------------------------------------------------------------------

    def get_frequency_table(self) -> pd.DataFrame:
        """
        Return the cached frequency table.

        Returns:
        --------
        pd.DataFrame
            Cached publisher frequency table.

        Raises:
        -------
        ValueError:
            If analyze() hasn't been run before.
        """
        if self._frequency_table is None:
            raise ValueError("🛑 No analysis results found. Run `.analyze(df)` first.")
        return self._frequency_table.copy()

    # ----------------------------------------------------------------------
    # Publication Date Counts Analysis
    # ----------------------------------------------------------------------

    def analyze_publication_dates(
        self, df: pd.DataFrame, date_col: str = "cleaned_date"
    ) -> pd.DataFrame:
        """
        Aggregate number of articles by publication date.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing a datetime column for publication date.
        date_col : str
            Name of the date column to aggregate on (default 'cleaned_date').

        Returns:
        --------
        pd.DataFrame
            DataFrame indexed by publication date with article counts.

        Raises:
        -------
        KeyError:
            If specified date column does not exist in input DataFrame.
        """
        # Validate date column existence
        if date_col not in df.columns:
            raise KeyError(f"🛑 Column '{date_col}' not found in the DataFrame.")

        # Group by the date column and count articles per date
        pub_counts = (
            df.groupby(date_col)
            .size()  # Count articles per date group
            .reset_index(name="article_count")  # Rename the counts column
            .sort_values(by=date_col)  # Sort by date ascending for timeline
        )

        # Cache the publication counts internally
        self._publication_counts = pub_counts.copy()

        # Verbose diagnostics for count and sample display
        if self.verbose:
            print(f"✅ Computed article counts over {pub_counts.shape[0]} dates.")
            print("🔍 Sample of publication date counts:")
            self._display(pub_counts.head(5))

        return pub_counts  # Return publication date counts

    # ----------------------------------------------------------------------
    # Publication Date Counts Plotting
    # ----------------------------------------------------------------------

    def plot_publication_dates(
        self, pub_counts: pd.DataFrame = None, figsize: tuple = (12, 5)
    ) -> None:
        """
        Plot article counts over time as a line chart.

        Parameters:
        -----------
        pub_counts : pd.DataFrame, optional
            DataFrame with publication date counts to plot.
            If None, uses internally cached publication counts.
        figsize : tuple
            Figure size for the plot (default (12, 5)).

        Raises:
        -------
        ValueError:
            If no publication counts data is available for plotting.
        """
        # Use internal cached data if none provided
        if pub_counts is None:
            pub_counts = getattr(self, "_publication_counts", None)
            if pub_counts is None:
                raise ValueError(
                    "🛑 No publication counts data available. Run analyze_publication_dates() first."
                )

        # Import plotting libs locally to avoid polluting global namespace
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Create matplotlib figure
        plt.figure(figsize=figsize)

        # Use seaborn lineplot to visualize article counts over time
        sns.lineplot(data=pub_counts, x=pub_counts.columns[0], y="article_count")

        # Chart title and axis labels for clarity
        plt.title("🗓️ Article Counts Over Time")
        plt.xlabel(pub_counts.columns[0])
        plt.ylabel("Number of Articles")

        # Adjust layout and display the plot
        plt.tight_layout()
        plt.show()

        # Verbose confirmation message if enabled
        if self.verbose:
            print(f"📊 Publication date frequency chart rendered successfully.")

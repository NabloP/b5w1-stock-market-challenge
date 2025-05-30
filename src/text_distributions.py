"""
text_distributions.py – Headline Text Distribution Visuals
-----------------------------------------------------------

Encapsulates a reusable, object-oriented class for visualizing headline text features:
- Histogram of character length
- Histogram of word count

Supports diagnostic verbosity, styling overrides, and integration into EDA pipelines.

Author: Nabil Mohamed
"""

import pandas as pd  # For DataFrame handling
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For advanced statistical visualizations


# ------------------------------------------------------------------------------
# 📊 TextDistributionPlotter Class
# ------------------------------------------------------------------------------


class TextDistributionPlotter:  # Text distribution visualization class
    """
    A class for visualizing distribution plots of text-based features,
    such as headline character length and word count.

    Designed for plug-and-play use in modular EDA pipelines.
    """

    def __init__(  # TextDistributionPlotter constructor
        self,  # self reference
        length_col: str = "headline_length",  # Column name for character length
        word_col: str = "word_count",  # Column name for word count
        bins: int = 30,  # Number of bins in each histogram
        figsize: tuple = (12, 4),  # Matplotlib figure size
        style: str = "seaborn-v0_8-muted",  # Matplotlib/seaborn style
        use_latex_style: bool = False,  # Whether to format y-axis using LaTeX-style commas
        title_prefix: str = "📊",
        verbose: bool = True,  # Print diagnostic information
    ):
        """
        Initialize the plotter with configurable options.

        Parameters:
        -----------
        length_col : str
            Column name representing character length.
        word_col : str
            Column name representing word count.
        bins : int
            Number of bins in each histogram.
        figsize : tuple
            Matplotlib figure size.
        style : str
            Matplotlib/seaborn style.
        use_latex_style : bool
            Whether to format y-axis using LaTeX-style commas.
        title_prefix : str
            Optional prefix for subplot titles.
        verbose : bool
            Print diagnostic information.
        """
        self.length_col = length_col  # Store character length column name
        self.word_col = word_col  # Store word count column name
        self.bins = bins  # Store number of bins for histograms
        self.figsize = figsize  # Store figure size for plots
        self.style = style  # Store Matplotlib/seaborn style
        self.use_latex_style = (
            use_latex_style  # Whether to use LaTeX-style formatting for y-axis
        )
        self.title_prefix = title_prefix  # Store title prefix for subplots
        self.verbose = verbose  # Whether to print diagnostic information

    def _validate_columns(self, df: pd.DataFrame):  # Validate required columns
        """Ensure required columns exist in the DataFrame."""
        missing = [
            col for col in [self.length_col, self.word_col] if col not in df.columns
        ]  # Check for missing columns
        if missing:  # If any required columns are missing
            raise KeyError(f"🛑 Missing required column(s): {', '.join(missing)}")
        if self.verbose:  # If verbose mode is enabled
            print(f"✅ Columns validated: {self.length_col}, {self.word_col}")

    def _apply_styling(self):  # Apply Matplotlib styling
        """Set plotting style."""
        plt.style.use(self.style)  # Apply the specified style
        if self.verbose:  # If verbose mode is enabled
            print(f"🎨 Style applied: {self.style}")

    def _apply_latex_axis_format(
        self, ax: plt.Axes
    ):  # Apply LaTeX-style formatting to y-axis
        """Format y-axis ticks with commas (if enabled)."""
        if self.use_latex_style:  # If LaTeX-style formatting is enabled
            ax.ticklabel_format(
                style="plain", axis="y", useOffset=False
            )  # Disable scientific notation
            ax.yaxis.set_major_formatter(  # Set y-axis formatter
                plt.FuncFormatter(lambda x, _: f"{int(x):,}")  # Format with commas
            )

    def plot(
        self, df: pd.DataFrame, show: bool = True
    ):  # Plot histograms of text feature distributions
        """
        Plot histograms of text feature distributions.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with pre-computed text features.
        show : bool
            Whether to immediately render the plots.
        """
        self._validate_columns(df)  # Validate required columns
        self._apply_styling()  # Apply Matplotlib styling

        fig, axs = plt.subplots(
            1, 2, figsize=self.figsize
        )  # Create subplots for character length and word count

        # Character length plot
        sns.histplot(
            df[self.length_col], bins=self.bins, ax=axs[0], kde=True, color="steelblue"
        )  # Plot character length distribution
        axs[0].set_title(
            f"{self.title_prefix} Headline Character Length"
        )  # Set title for character length plot
        axs[0].set_xlabel("Characters")  # Set x-axis label for character length
        axs[0].set_ylabel("Frequency")  # Set y-axis label for character length
        self._apply_latex_axis_format(
            axs[0]
        )  # Apply LaTeX-style formatting to y-axis if enabled

        # Word count plot
        sns.histplot(
            df[self.word_col], bins=self.bins, ax=axs[1], kde=True, color="darkorange"
        )  # Plot word count distribution
        axs[1].set_title(
            f"{self.title_prefix} Headline Word Count"
        )  # Set title for word count plot
        axs[1].set_xlabel("Words")  # Set x-axis label for word count
        axs[1].set_ylabel("Frequency")  # Set y-axis label for word count
        self._apply_latex_axis_format(
            axs[1]
        )  # Apply LaTeX-style formatting to y-axis if enabled

        plt.tight_layout()  # Adjust layout to prevent overlap

        if show:  # If show is True, render the plots
            plt.show()
            if self.verbose:  # If verbose mode is enabled
                print("📈 Distribution plots rendered successfully.")

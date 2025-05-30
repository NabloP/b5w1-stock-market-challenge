"""
text_publishers_domains.py – Publisher Domain Extraction & Analytics
---------------------------------------------------------------------

Modular utility for extracting and analyzing publisher domains from financial news data.

Features:
- Robust domain parsing from URLs or free-form publisher names
- Graceful null handling and fallback heuristics
- Diagnostic-friendly verbosity and exception messaging
- Integrated analytics and plotting API

Author: Nabil Mohamed
"""

import pandas as pd  # For DataFrame operations
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For statistical data visualization
from urllib.parse import urlparse  # For parsing URLs


# ------------------------------------------------------------------------------
# 🌐 PublisherDomainAnalyzer – Full Encapsulation for Domain Analytics
# ------------------------------------------------------------------------------


class PublisherDomainAnalyzer:
    """
    Extracts domain names from publisher text, computes frequency statistics,
    and optionally visualizes the most frequent sources.

    Parameters:
    -----------
    publisher_col : str
        Column name in DataFrame that contains publisher URLs or names.
    verbose : bool
        Whether to log diagnostics to the console (default: True).
    """

    def __init__(
        self,
        publisher_col: str = "publisher",
        domain_col: str = "publisher_domain",
        verbose: bool = True,
    ):  # Function to initialize the analyzer
        self.publisher_col = publisher_col  # Column containing publisher names or URLs
        self.domain_col = domain_col  # Column to store extracted domains
        self.verbose = verbose  # Whether to print diagnostics and analysis results
        self._domain_freq_table = None  # Internal cache for domain frequency table

    def _extract_domain(self, publisher: str) -> str:
        """
        Internal logic to convert a publisher string into a domain-like form.
        Handles both raw URLs and plain text via fallback heuristics.

        Parameters:
        -----------
        publisher : str
            Raw input string from publisher column.

        Returns:
        --------
        str : Extracted domain string (e.g., 'yahoo.com').
        """
        try:
            # Normalize to valid URL (prepend scheme if missing)
            if not isinstance(publisher, str) or publisher.strip() == "":
                return "unknown"

            if "://" not in publisher:
                publisher = f"https://{publisher}"

            parsed = urlparse(publisher)
            host = parsed.hostname or publisher
            parts = host.lower().split(".")

            # Extract root domain (e.g., 'finance.yahoo.com' → 'yahoo.com')
            return ".".join(parts[-2:]) if len(parts) >= 2 else host.lower()

        except Exception as e:
            if self.verbose:
                print(f"⚠️ Domain parse failed for: {publisher} → {e}")
            return "unknown"

    def analyze(
        self, df: pd.DataFrame, top_n: int = 10, return_freq: bool = False
    ) -> pd.DataFrame:
        """
        Parses domains and computes article frequency per domain.
        Also computes frequency stats and stores them for optional plotting.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataset containing publisher column.
        top_n : int
            Number of top domains to display during diagnostics (default: 10).

        Returns:
        --------
        pd.DataFrame : Table of domain → article count.

        Raises:
        -------
        KeyError : If specified column is missing from the input DataFrame.
        """
        if self.publisher_col not in df.columns:
            raise KeyError(
                f"🛑 Column '{self.publisher_col}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

        # Step 1: Fill missing publishers with placeholder
        publishers = df[self.publisher_col].fillna("unknown")

        # Step 2: Apply domain extraction to each entry
        df[self.domain_col] = publishers.apply(self._extract_domain)

        # Step 3: Compute domain frequency table
        freq = (
            df[self.domain_col]
            .value_counts()
            .reset_index()
            .rename(
                columns={"index": self.domain_col, self.domain_col: "article_count"}
            )
        )

        # 🔐 Forcefully set correct columns just in case
        freq.columns = [self.domain_col, "article_count"]
        self._domain_freq_table = freq  # Cache result

        # Step 4: Optional diagnostics
        if self.verbose:
            print(f"✅ Domain analysis complete on column '{self.publisher_col}'")
            print(f"🔢 {freq.shape[0]:,} unique domains found.")
            print(f"📈 Top {top_n} domains:")
            try:
                from IPython.display import display

                display(freq.head(top_n))
            except ImportError:
                print(freq.head(top_n))

        return (df, self._domain_freq_table) if return_freq else df

    def plot_top_domains(self, top_n: int = 10, figsize: tuple = (10, 5)) -> None:
        """
        Renders a horizontal bar chart of the most common publisher domains.

        Parameters:
        -----------
        top_n : int
            Number of domains to display.
        figsize : tuple
            Size of the output plot.

        Raises:
        -------
        ValueError : If analyze() hasn’t been run yet.
        """
        if self._domain_freq_table is None:
            raise ValueError(
                "🛑 Domain frequency table not computed yet. Run `.analyze(df)` first."
            )

        top_data = self._domain_freq_table.head(top_n)

        # Plot configuration
        plt.figure(figsize=figsize)

        if self.verbose:
            print("📋 Columns in top_data:", top_data.columns.tolist())
            print(top_data.head())

        sns.barplot(
            x="article_count", y=self.domain_col, data=top_data, palette="Blues_r"
        )
        plt.title(f"🌐 Top {top_n} Publisher Domains by Article Volume")
        plt.xlabel("Number of Articles")
        plt.ylabel(self.domain_col.replace("_", " ").title())
        plt.tight_layout()
        plt.show()

        if self.verbose:
            print(f"📊 Rendered bar chart for top {top_n} domains.")

    def get_domain_frequency_table(self) -> pd.DataFrame:
        """
        Access the internal frequency table from the last analysis.

        Returns:
        --------
        pd.DataFrame : Copy of the cached frequency table.

        Raises:
        -------
        ValueError : If analyze() has not yet been executed.
        """
        if self._domain_freq_table is None:
            raise ValueError("🛑 No domain frequency table available.")
        return self._domain_freq_table.copy()

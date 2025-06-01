"""
text_timestamp_cleaner.py – Robust Timestamp Cleaning Module with Preprocessing
-------------------------------------------------------------------------------

Designed for financial news datasets where timestamps might contain hidden
invisible characters, whitespace issues, or subtle encoding quirks that break
naive pandas parsing.

Features:
- Null-safe string coercion and trimming
- Invisible Unicode character detection (e.g., zero-width spaces)
- Parsing with error coercion (try format='ISO8601', fallback to auto-infer)
- Explicit localization of naive timestamps to UTC
- Timezone conversion and optional tz removal
- Date normalization (strip time portion, keep date only)
- Fallback normalization if failure rate > 10%
- Parsing success/failure stats and example logging
- Fully object-oriented, modular, and verbose for debugging

Author: Nabil Mohamed
"""

from typing import Optional, List  # For potential type hints and future-proofing
import pandas as pd  # Pandas for dataframe and datetime operations
import unicodedata  # Unicode character properties for invisible char detection
import sys  # To detect terminal output capabilities for colored prints


class TimestampCleaner:
    # ANSI color codes for terminal output formatting (green, yellow, red, reset)
    ANSI_GREEN = "\033[92m"  # Green text for success messages
    ANSI_YELLOW = "\033[93m"  # Yellow text for warnings
    ANSI_RED = "\033[91m"  # Red text for errors/failures
    ANSI_RESET = "\033[0m"  # Reset ANSI color to default

    def __init__(
        self,
        timestamp_col: str = "date",  # Column in input DataFrame containing raw timestamps
        cleaned_col: str = "cleaned_date",  # Column to output cleaned datetime values
        target_tz: str = "America/New_York",  # Timezone string to localize timestamps
        verbose: bool = True,  # Enable/disable verbose debug printing
    ):
        # Store constructor arguments as instance variables for use throughout the class
        self.timestamp_col = timestamp_col  # Raw timestamp column name
        self.cleaned_col = cleaned_col  # Cleaned timestamp output column name
        self.target_tz = target_tz  # Target timezone for parsed timestamps
        self.verbose = verbose  # Verbosity flag for detailed output

        # Initialize a dictionary to keep statistics and diagnostics during processing
        self.stats = {
            "total": 0,  # Total number of rows processed
            "parsed": 0,  # Number of successfully parsed timestamps
            "failed": 0,  # Number of timestamps that failed to parse (NaT)
            "failed_examples": [],  # Sample failed cleaned string values
            "failed_examples_repr": [],  # repr() formatted failed cleaned strings
            "failed_raw_examples": [],  # Sample of original raw inputs that failed
            "invisible_char_examples": [],  # Samples with invisible unicode chars
            "source_type_distribution": {},  # Distribution of raw input value types
            "individual_parse_test": None,  # Result of testing first failed parse individually
            "fallback_activated": False,  # Whether fallback normalization was used
        }

    def _has_invisible_chars(self, s: str) -> bool:
        """
        Check if the input string contains invisible Unicode characters that
        may cause parsing issues. These include format, surrogate, or private
        use characters categorized as Cf, Cs, or Co in Unicode.

        Args:
            s (str): The string to inspect.

        Returns:
            bool: True if any invisible chars are found, False otherwise.
        """
        # Iterate through each character in the string
        for c in s:
            # If character's Unicode category is one of Cf, Cs, Co (invisible/control chars)
            if unicodedata.category(c) in {"Cf", "Cs", "Co"}:
                return True  # Invisible char found
        return False  # No invisible chars detected

    def _remove_control_chars(self, s: str) -> str:
        """
        Remove all Unicode control/non-printable characters from a string.
        These characters have Unicode categories starting with 'C'.

        Args:
            s (str): Input string.

        Returns:
            str: String with all control chars removed.
        """
        # Build a new string omitting any character whose Unicode category starts with 'C'
        return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")

    def _preclean_series(self, series: pd.Series) -> pd.Series:
        """
        Preprocess the timestamp column series by:
        - Filling nulls with empty strings
        - Casting all to strings
        - Removing control characters
        - Stripping leading/trailing whitespace
        - Optionally logging presence of invisible characters for diagnostics

        Args:
            series (pd.Series): Raw timestamp series to clean.

        Returns:
            pd.Series: Cleaned string series ready for datetime parsing.
        """
        # Fill NA values with empty strings, convert to string, remove control chars, strip spaces
        cleaned = (
            series.fillna("").astype(str).apply(self._remove_control_chars).str.strip()
        )

        # If verbosity enabled, find and log samples containing invisible Unicode chars
        if self.verbose:
            invis_mask = cleaned.apply(self._has_invisible_chars)  # Boolean mask
            self.stats["invisible_char_examples"] = (
                cleaned[invis_mask]
                .unique()[:5]
                .tolist()  # Take first 5 unique examples
            )
            # Print warning if any invisible chars are found
            if invis_mask.any():
                print(
                    f"{self.ANSI_YELLOW}⚠️ Found {invis_mask.sum():,} rows with invisible Unicode chars.{self.ANSI_RESET}"
                )
                print(
                    f"Sample invisible-char values: {self.stats['invisible_char_examples']}"
                )

        return cleaned  # Return the cleaned series

    def _apply_fallback_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fallback fix for stubborn parsing failures:
        Round-trip datetime column through string formatting and back to datetime.
        This can normalize subtle format quirks causing parse errors.

        Args:
            df (pd.DataFrame): DataFrame with cleaned datetime column.

        Returns:
            pd.DataFrame: DataFrame with fallback normalized datetime column.
        """
        # Verbose notification about fallback activation
        if self.verbose:
            print(
                f"{self.ANSI_YELLOW}⚠️ Applying fallback normalization via round-trip string conversion.{self.ANSI_RESET}"
            )

        # Format datetime column to uniform string format without timezone
        formatted = df[self.cleaned_col].dt.strftime("%Y-%m-%d %H:%M:%S")

        # Parse the formatted strings back to naive datetime objects
        parsed_series = pd.to_datetime(formatted, errors="coerce")

        # Defensive check: if fallback parse fails to produce valid series
        if parsed_series is None or parsed_series.empty:
            if self.verbose:
                print(
                    f"{self.ANSI_RED}❌ Fallback normalization failed to produce a valid datetime Series.{self.ANSI_RESET}"
                )
            # Fill entire column with NaT as fallback failed
            df[self.cleaned_col] = pd.Series([pd.NaT] * len(df), index=df.index)
            return df  # Return early with all NaT column

        # Overwrite cleaned datetime column with normalized fallback timestamps
        df[self.cleaned_col] = parsed_series

        # Localize naive timestamps (tzinfo=None) explicitly to UTC timezone
        if pd.api.types.is_datetime64_any_dtype(df[self.cleaned_col]):
            naive_mask = (
                df[self.cleaned_col].notna() & df[self.cleaned_col].dt.tz.isnull()
            )
            # Localize all naive timestamps to UTC
            df.loc[naive_mask, self.cleaned_col] = df.loc[
                naive_mask, self.cleaned_col
            ].dt.tz_localize("UTC")

            # Convert all timestamps to target timezone
            parsed_mask = df[self.cleaned_col].notna()
            df.loc[parsed_mask, self.cleaned_col] = df.loc[
                parsed_mask, self.cleaned_col
            ].dt.tz_convert(self.target_tz)

            # Remove timezone info for consistent naive downstream timestamps
            df[self.cleaned_col] = df[self.cleaned_col].dt.tz_localize(None)
        else:
            # If dtype not datetime after fallback, print verbose warning
            if self.verbose:
                print(
                    f"{self.ANSI_RED}❌ After fallback parsing, column is not datetime dtype. Skipping tz localization.{self.ANSI_RESET}"
                )

        # Mark that fallback was applied for diagnostics
        self.stats["fallback_activated"] = True

        return df  # Return DataFrame with fallback-normalized timestamps

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main method to clean and parse timestamps robustly with diagnostics and fallback.

        Steps:
        - Verify input column exists.
        - Log type distribution of raw inputs.
        - Handle numeric timestamps converting them to ISO strings.
        - Pre-clean inputs for invisible/control chars and whitespace.
        - Parse with strict ISO8601 format; fallback to inferred format.
        - Localize naive timestamps to UTC; convert to target timezone.
        - Normalize timestamps to date only.
        - Collect failure diagnostics and apply fallback if >10% fails.
        - Verbosely print parsing diagnostics and summary.

        Args:
            df (pd.DataFrame): Input DataFrame with raw timestamps.

        Returns:
            pd.DataFrame: Output DataFrame with cleaned timestamps.
        """

        # Ensure input DataFrame contains the required timestamp column
        if self.timestamp_col not in df.columns:
            raise KeyError(f"🛑 Column '{self.timestamp_col}' not found in DataFrame.")

        self.stats["total"] = len(df)  # Track total rows

        raw_series = df[
            self.timestamp_col
        ]  # Extract raw timestamp column for diagnostics

        # Log input types (str, float, etc.) distribution for troubleshooting
        self.stats["source_type_distribution"] = (
            raw_series.apply(lambda x: type(x).__name__).value_counts().to_dict()
        )

        # Check if raw data is numeric (e.g., Excel float or UNIX seconds)
        if pd.api.types.is_numeric_dtype(raw_series):
            # Convert numeric timestamps to string representation of ISO datetime for parsing
            df[self.timestamp_col] = pd.to_datetime(
                raw_series, errors="coerce", unit="s"
            ).astype(str)

        # Pre-clean the timestamp strings to remove control chars and whitespace
        cleaned_strings = self._preclean_series(df[self.timestamp_col])

        # Attempt parsing with strict ISO8601 format to maximize correctness and speed
        try:
            df[self.cleaned_col] = pd.to_datetime(
                cleaned_strings, errors="coerce", format="ISO8601"
            )
        except Exception as e:
            # On failure, fallback to pandas inference parsing with errors coerced
            if self.verbose:
                print(
                    f"{self.ANSI_YELLOW}⚠️ Warning: ISO8601 parsing failed: {e}{self.ANSI_RESET}"
                )
            df[self.cleaned_col] = pd.to_datetime(
                cleaned_strings, errors="coerce", infer_datetime_format=True
            )

        # Calculate parse failures and successes
        self.stats["failed"] = df[self.cleaned_col].isna().sum()  # Count NaT rows
        self.stats["parsed"] = (
            self.stats["total"] - self.stats["failed"]
        )  # Success count

        if self.stats["parsed"] > 0:
            # Create mask of successfully parsed rows
            parsed_mask = df[self.cleaned_col].notna()
            timestamps = df.loc[parsed_mask, self.cleaned_col]

            # Detect naive timestamps that lack timezone info
            naive_mask = timestamps.apply(lambda x: x.tzinfo is None)

            # Localize naive timestamps explicitly to UTC timezone
            to_localize = timestamps[naive_mask]
            if not to_localize.empty:
                df.loc[to_localize.index, self.cleaned_col] = (
                    to_localize.dt.tz_localize("UTC")
                )

            # Convert all timezone-aware timestamps to the target timezone
            df.loc[parsed_mask, self.cleaned_col] = df.loc[
                parsed_mask, self.cleaned_col
            ].dt.tz_convert(self.target_tz)

            # Remove timezone info to keep timestamps naive downstream
            df[self.cleaned_col] = df[self.cleaned_col].dt.tz_localize(None)

        # Normalize the timestamps to date only (strip time part to midnight)
        df[self.cleaned_col] = df[self.cleaned_col].dt.normalize()

        # Gather sample failed values for diagnostics
        failed_idx = df.index[df[self.cleaned_col].isna()]  # Indices of NaT rows
        self.stats["failed_raw_examples"] = (
            raw_series.loc[failed_idx].head(3).tolist()
        )  # First 3 raw fails
        failed_vals = (
            cleaned_strings.loc[failed_idx].dropna().unique()
        )  # Unique failed cleaned strings
        self.stats["failed_examples"] = failed_vals[
            :5
        ].tolist()  # Take first 5 unique fails
        self.stats["failed_examples_repr"] = [
            repr(v) for v in self.stats["failed_examples"]
        ]  # repr for diagnostics

        # Run individual parsing test on first failed example to debug
        if self.stats["failed_examples"]:
            test_val = self.stats["failed_examples"][0]  # Pick first failed example
            try:
                pd.to_datetime(
                    test_val, errors="raise"
                )  # Attempt parse (raises error if fails)
                self.stats["individual_parse_test"] = (
                    "SUCCESS"  # Mark success if no error
                )
            except Exception as e:
                self.stats["individual_parse_test"] = (
                    f"FAIL: {str(e)}"  # Record failure message
                )

        # If failure rate is high, trigger fallback normalization to rescue timestamps
        failure_rate = self.stats["failed"] / max(
            self.stats["total"], 1
        )  # Protect divide by zero
        if failure_rate > 0.1:
            # Notify user about fallback application
            if self.verbose:
                print(
                    f"\n{self.ANSI_YELLOW}⚠️ Failure rate {failure_rate:.2%} exceeds 10%, applying fallback normalization.{self.ANSI_RESET}"
                )
            df = self._apply_fallback_normalization(df)  # Perform fallback fix

            # Recalculate failure and success counts post-fallback
            self.stats["failed"] = df[self.cleaned_col].isna().sum()
            self.stats["parsed"] = self.stats["total"] - self.stats["failed"]

            # Normalize timestamps to date only after fallback too
            df[self.cleaned_col] = df[self.cleaned_col].dt.normalize()

        # Output detailed diagnostic summary if verbose
        if self.verbose:

            # Helper function to apply ANSI color codes conditionally for terminal output
            def color_text(text: str, color: str) -> str:
                return (
                    f"{color}{text}{self.ANSI_RESET}" if sys.stdout.isatty() else text
                )

            # Print section divider and heading
            print("\n" + "=" * 40)
            print(color_text("TimestampCleaner Summary:", self.ANSI_GREEN))
            print("=" * 40)

            # Print summary of total processed rows
            print(f"Total rows processed:    {self.stats['total']:,}")

            # Print count of successfully parsed timestamps with green coloring
            print(
                f"Successfully parsed:     {color_text(str(self.stats['parsed']), self.ANSI_GREEN)}"
            )

            # Print count of failed parses with red coloring
            print(
                f"Failed parses (NaT):     {color_text(str(self.stats['failed']), self.ANSI_RED)}"
            )

            # Print distribution of raw input data types (for troubleshooting)
            print("\nSource value types:")
            for t, c in self.stats["source_type_distribution"].items():
                print(f"  - {t}: {c}")

            # Notify if fallback normalization was applied
            if self.stats["fallback_activated"]:
                print(
                    f"\n{self.ANSI_YELLOW}Fallback normalization was applied due to high failure rate.{self.ANSI_RESET}"
                )

            # Show samples of failed cleaned strings and their repr() for hidden chars
            if self.stats["failed"] > 0:
                print("\nSample failed values:")
                for i, v in enumerate(self.stats["failed_examples"]):
                    print(f"  - {color_text(v, self.ANSI_RED)}")
                    print(
                        f"    repr: {color_text(self.stats['failed_examples_repr'][i], self.ANSI_RED)}"
                    )

                # Show the first 3 raw failed values before cleaning
                print("\nFirst 3 raw failed values (pre-preprocessing):")
                for v in self.stats["failed_raw_examples"]:
                    print(f"  - {color_text(repr(v), self.ANSI_YELLOW)}")

                # Display result of individual parsing test on first failed value
                print(
                    f"\nIndividual parse test: {self.stats.get('individual_parse_test', 'N/A')}"
                )

            # Show sample values that contained invisible unicode characters
            if self.stats["invisible_char_examples"]:
                print("\nSample values with invisible chars:")
                for v in self.stats["invisible_char_examples"]:
                    print(f"  - {color_text(v, self.ANSI_YELLOW)}")

            # Print closing section divider
            print("=" * 40 + "\n")

        # Return the processed DataFrame with a new 'cleaned_date' column
        return df

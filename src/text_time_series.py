"""
text_time_series.py – Headline Timestamp Transformation & Visualization
------------------------------------------------------------------------

Modular, production-grade system for:
- Extracting time-based features from headline publication timestamps
- Visualizing headline frequency across hours, weekdays, and dates

Complies with:
✔️ Single Responsibility Principle (SRP)
✔️ UTC-4 awareness for timestamp parsing
✔️ Null-safe, verbose diagnostics
✔️ Matplotlib/Seaborn integration for time series visuals

Author: Nabil Mohamed
"""

import pandas as pd  # DataFrame handling
import matplotlib.pyplot as plt  # Plotting library
import seaborn as sns  # Statistical visualization
from typing import Optional, Union  # Type hinting for flexibility

# ------------------------------------------------------------------------------
# 🧠 HeadlineTimeTransformer – Extracts time-based features from datetime
# ------------------------------------------------------------------------------


class HeadlineTimeTransformer:
    """
    Extracts interpretable time components from a datetime column.

    Features extracted:
    - date_only (YYYY-MM-DD)
    - hour (0–23)
    - day_of_week (Mon–Sun)
    - is_weekend (True/False)

    Parameters:
    -----------
    timestamp_col : str
        Column name containing datetime strings (default: "date").
    timezone : str
        Timezone to localize datetime to (default: "America/New_York").
    verbose : bool
        Whether to print transformation diagnostics (default: True).
    """

    def __init__(  # Function to initialize the transformer
        self,  # Parameters for the transformer
        timestamp_col: str = "date",  # Column name for datetime strings
        timezone: str = "America/New_York",  # Timezone for localization
        verbose: bool = True,  # Whether to print diagnostics
    ):
        self.timestamp_col = timestamp_col  # Column name for datetime strings
        self.timezone = timezone  # Timezone for localization
        self.verbose = verbose  # Whether to print diagnostics

    def transform(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:  # Function to transform DataFrame
        """
        Adds time-based columns derived from a timestamp.

        Returns:
        --------
        pd.DataFrame : Transformed DataFrame with new datetime features.

        Raises:
        -------
        KeyError : If the specified timestamp column is missing.
        ValueError : If date parsing fails for all values.
        """

        if self.timestamp_col not in df.columns:  # Check if the timestamp column exists
            raise KeyError(f"🛑 Column '{self.timestamp_col}' not found in DataFrame.")

        # Step 1: Parse and localize datetime
        try:  # Attempt to parse the timestamp column
            df["_parsed_ts"] = pd.to_datetime(
                df[self.timestamp_col], errors="coerce", utc=True
            )  # Parse with UTC
            df["_parsed_ts"] = df["_parsed_ts"].dt.tz_convert(
                self.timezone
            )  # Convert to specified timezone
        except Exception as e:  # Catch any parsing errors
            raise ValueError(
                f"❌ Timestamp parsing failed: {e}"
            )  # Raise error if parsing fails

        # Step 2: Null check
        missing_ts = df["_parsed_ts"].isna().sum()  # Count unparseable timestamps
        if missing_ts > 0:  # If there are unparseable timestamps
            print(
                f"⚠️ {missing_ts} rows had unparseable timestamps and will be NaN."
            )  # Print warning

        # Step 3: Extract features
        df["date_only"] = df["_parsed_ts"].dt.date  # Extract date part (YYYY-MM-DD)
        df["hour"] = df["_parsed_ts"].dt.hour  # Extract hour (0–23)
        df["day_of_week"] = df[
            "_parsed_ts"
        ].dt.day_name()  # Extract day of week (Mon–Sun)
        df["is_weekend"] = df["day_of_week"].isin(
            ["Saturday", "Sunday"]
        )  # Check if day is weekend

        # Drop temporary column
        df.drop(
            columns=["_parsed_ts"], inplace=True
        )  # Remove the parsed timestamp column

        if self.verbose:  # If verbose mode is enabled
            print(
                "✅ Time features added: ['date_only', 'hour', 'day_of_week', 'is_weekend']"
            )  # Print confirmation
            print("📅 Unique dates:", df["date_only"].nunique())
            print("🕒 Publishing hours:", sorted(df["hour"].dropna().unique().tolist()))

        return df


# ------------------------------------------------------------------------------
# 📊 HeadlineTimeVisualizer – Plots time-based publication patterns
# ------------------------------------------------------------------------------


class HeadlineTimeVisualizer:  # Class for visualizing headline publishing trends
    """
    Visualizes headline publishing trends over time.

    Methods:
    --------
    plot_daily_counts(df)
    plot_hourly_distribution(df)
    plot_weekday_distribution(df)
    """

    def __init__(
        self, style: str = "seaborn-v0_8-muted"
    ):  # Function to initialize the visualizer
        self.style = style  # Style for plots
        plt.style.use(self.style)  # Set the plotting style

    def plot_daily_counts(
        self, df: pd.DataFrame, date_col: str = "date_only"
    ):  # Function to plot daily article counts
        """
        Line chart showing daily article counts.

        Parameters:
        -----------
        df : pd.DataFrame
        date_col : str : Column with date values
        """
        if date_col not in df.columns:  # Check if the date column exists
            raise KeyError(f"🛑 Missing column: {date_col}")

        daily_counts = (
            df[date_col].value_counts().sort_index()
        )  # Count articles per day
        plt.figure(figsize=(12, 4))  # Create figure for daily counts
        sns.lineplot(
            x=daily_counts.index, y=daily_counts.values
        )  # Plot line chart of daily counts
        plt.title("🗓️ Headlines Published per Day")  # Sets plot title
        plt.xlabel("Date")  # Sets x-axis label
        plt.ylabel("Count")  # Sets y-axis label
        plt.xticks(rotation=45)  # Rotate x-axis labels for readability
        plt.tight_layout()  # Adjust layout to prevent clipping
        plt.show()  # Display the plot

    def plot_hourly_distribution(
        self, df: pd.DataFrame, hour_col: str = "hour"
    ):  # Function to plot hourly distribution of headlines
        """
        Histogram of headline publishing hours.

        Parameters:
        -----------
        df : pd.DataFrame
        hour_col : str : Column with hour values
        """
        if hour_col not in df.columns:  # Check if the hour column exists
            raise KeyError(f"🛑 Missing column: {hour_col}")

        plt.figure(figsize=(8, 4))  # Create figure for hourly distribution
        sns.histplot(
            df[hour_col], bins=24, kde=False, color="steelblue"
        )  # Plot histogram of hours
        plt.title("🕒 Headlines by Hour of Day")  # Set plot title
        plt.xlabel("Hour (0–23)")  # Set x-axis label
        plt.ylabel("Frequency")  # Set y-axis label
        plt.tight_layout()  # Adjust layout to prevent clipping
        plt.show()  # Display the plot

    def plot_weekday_distribution(
        self, df: pd.DataFrame, weekday_col: str = "day_of_week"
    ):  # Function to plot weekday distribution of headlines
        """
        Bar chart of headline frequency by weekday.

        Parameters:
        -----------
        df : pd.DataFrame
        weekday_col : str : Column with weekday names
        """
        if weekday_col not in df.columns:  # Check if the weekday column exists
            raise KeyError(f"🛑 Missing column: {weekday_col}")

        order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]  # Define order of weekdays
        plt.figure(figsize=(8, 4))  # Set figure size for weekday distribution
        sns.countplot(
            x=weekday_col, data=df, order=order, palette="crest"
        )  # Plot bar chart of weekdays
        plt.title("📆 Headlines by Day of Week")  # Set plot title
        plt.xlabel("Weekday")  # Set x-axis label
        plt.ylabel("Frequency")  #
        plt.tight_layout()  # Adjust layout to prevent clipping
        plt.show()  # Display the plot


# ------------------------------------------------------------------------------
# 🧩 TimeSeriesTextAnalyzer – Unified Wrapper for Transform + Plot
# ------------------------------------------------------------------------------


class TimeSeriesTextAnalyzer:  # Unified wrapper for time series text analysis
    """
    Unified wrapper for transforming and visualizing time-based headline data.

    Combines:
    - HeadlineTimeTransformer for timestamp decomposition
    - HeadlineTimeVisualizer for plotting trends

    Methods:
    --------
    transform(df): Adds time-based features.
    plot(df): Plots daily, hourly, and weekday distributions.
    """

    def __init__(  # Function to initialize the analyzer
        self,  # Parameters for the analyzer
        date_col: str = "date",  # Column name for datetime strings
        timezone: str = "America/New_York",  # Timezone for localization
        verbose: bool = True,  # Whether to print diagnostics
    ):
        self.transformer = HeadlineTimeTransformer(
            timestamp_col=date_col, timezone=timezone, verbose=verbose
        )  # Initialize the transformer
        self.visualizer = HeadlineTimeVisualizer()  # Initialize the visualizer

    def transform(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:  # Function to transform DataFrame
        return self.transformer.transform(
            df
        )  # Transform the DataFrame using the transformer

    def plot(self, df: pd.DataFrame) -> None:  # Function to plot visualizations
        self.visualizer.plot_daily_counts(df)  # Plot daily article counts
        self.visualizer.plot_hourly_distribution(
            df
        )  # Plot hourly distribution of headlines
        self.visualizer.plot_weekday_distribution(
            df
        )  # Plot weekday distribution of headlines

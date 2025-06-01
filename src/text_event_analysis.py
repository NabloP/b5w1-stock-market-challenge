"""
text_event_analysis.py – Financial Event & Entity Extraction Engine
-------------------------------------------------------------------

A modular NLP system for identifying key financial event phrases and named entities
from cleaned news headlines using spaCy-based noun chunking and NER. Includes advanced
relation-based extraction with REBEL transformer model and timeline analysis for anomaly detection.

Features:
- Extracts named entities like ORG, GPE, DATE, MONEY, EVENT from headlines
- Captures multi-word financial noun phrases (e.g., 'interest rate hike')
- Filters for key event keywords (e.g., earnings, merger, acquisition, fine)
- Computes frequency distribution of extracted event phrases
- Visualizes top N events using a horizontal bar chart
- Extracts structured event triplets using REBEL transformer (subject-relation-object)
- Combines REBEL and fallback methods for scalable analysis of large corpora
- Analyzes temporal trends via cleaned_date column (with graceful fallback)
- Strong error handling and verbose diagnostics
- Fully encapsulated OOP architecture

Author: Nabil Mohamed
"""

from typing import List, Optional
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import spacy
import re
import random
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM


class EventExtractor:
    def __init__(self, headlines: List[str], model: str = "en_core_web_sm", verbose: bool = True):
        self.headlines = headlines
        self.verbose = verbose
        if not headlines:
            raise ValueError("Headlines list is empty. Provide at least one headline.")
        try:
            self.nlp = spacy.load(model)
        except OSError:
            raise RuntimeError(f"spaCy model '{model}' not installed. Run: python -m spacy download {model}")
        try:
            self.doc_objects = list(self.nlp.pipe(headlines, disable=["parser"]))
        except Exception as e:
            raise RuntimeError(f"Failed to parse headlines with spaCy: {e}")
        if self.verbose:
            print(f"✅ Loaded {len(self.headlines)} headlines and parsed with model '{model}'.")

    def extract_named_entities(self, allowed_labels: Optional[List[str]] = None) -> pd.DataFrame:
        allowed_labels = allowed_labels or ["ORG", "GPE", "MONEY", "DATE", "EVENT"]
        records = []
        for doc, headline in zip(self.doc_objects, self.headlines):
            for ent in doc.ents:
                if ent.label_ in allowed_labels:
                    records.append({"headline": headline, "entity_text": ent.text, "label": ent.label_})
        df = pd.DataFrame(records)
        if self.verbose:
            print(f"🔍 Extracted {len(df)} named entities from headlines.")
        return df

    def extract_noun_phrases(self, min_len: int = 2) -> pd.DataFrame:
        results = []
        for doc, headline in zip(self.doc_objects, self.headlines):
            for chunk in doc.noun_chunks:
                chunk_text = chunk.text.lower()
                if len(chunk_text.split()) >= min_len:
                    results.append({"headline": headline, "noun_phrase": chunk_text})
        df = pd.DataFrame(results)
        if self.verbose:
            print(f"🧠 Extracted {len(df)} noun phrases of length ≥ {min_len}.")
        return df

    def extract_event_phrases(self) -> pd.DataFrame:
        financial_keywords = [
            "merger", "acquisition", "hike", "cut", "lawsuit", "dividend", "split",
            "earnings", "ipo", "bankruptcy", "investigation", "fine", "forecast",
            "downgrade", "upgrade", "guidance", "expansion"
        ]
        results = []
        for doc, headline in zip(self.doc_objects, self.headlines):
            for chunk in doc.noun_chunks:
                phrase = chunk.text.lower()
                if any(k in phrase for k in financial_keywords):
                    results.append({
                        "headline": headline,
                        "event_phrase": phrase,
                        "extraction_method": "KeywordBased"
                    })
        df = pd.DataFrame(results)
        if self.verbose:
            print(f"📌 Identified {len(df)} candidate financial event phrases.")
        return df

    def compute_event_frequencies(self) -> pd.DataFrame:
        event_df = self.extract_event_phrases()
        if event_df.empty:
            if self.verbose:
                print("⚠️ No event phrases extracted. Skipping frequency computation.")
            return pd.DataFrame(columns=["event_phrase", "frequency"])
        freq_counter = Counter(event_df["event_phrase"])
        freq_df = pd.DataFrame(freq_counter.items(), columns=["event_phrase", "frequency"])
        freq_df = freq_df.sort_values(by="frequency", ascending=False)
        if self.verbose:
            print(f"📊 Computed frequencies for {len(freq_df)} unique event phrases.")
        return freq_df

    def visualize_top_events(self, top_n: int = 15):
        freq_df = self.compute_event_frequencies().head(top_n)
        if freq_df.empty:
            print("⚠️ No event data available for visualization.")
            return
        plt.figure(figsize=(10, 6))
        plt.barh(freq_df["event_phrase"], freq_df["frequency"], color="darkred")
        plt.xlabel("Frequency")
        plt.title(f"Top {top_n} Financial Event Phrases")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        if self.verbose:
            print(f"📈 Plotted top {top_n} financial event phrases.")

    def extract_combined_events(self, rebel_model: "EventExtractorREBEL", sample_size: int = 1000) -> pd.DataFrame:
        """Uses REBEL for sampled subset and keyword-based extraction for remaining."""
        if len(self.headlines) <= sample_size:
            sample_headlines = self.headlines
            fallback_headlines = []
        else:
            sample_headlines = random.sample(self.headlines, sample_size)
            fallback_headlines = list(set(self.headlines) - set(sample_headlines))

        rebel_df = rebel_model.extract_triplets(sample_headlines)
        rebel_df["extraction_method"] = "REBEL"

        fallback_events = EventExtractor(fallback_headlines, verbose=False).extract_event_phrases()

        combined = pd.concat([rebel_df, fallback_events], ignore_index=True, sort=False)
        if self.verbose:
            print(f"🧩 Combined REBEL triplets ({len(rebel_df)}) and keyword phrases ({len(fallback_events)}).")
        return combined


class EventTimelineAnalyzer:
    def __init__(self, event_df: pd.DataFrame, verbose: bool = True):
        self.verbose = verbose
        self.df = event_df.copy()
        required_cols = {"event_phrase", "headline"}
        missing_cols = required_cols - set(self.df.columns)
        if missing_cols:
            raise KeyError(f"Missing required columns in event_df: {missing_cols}")
        if "cleaned_date" not in self.df.columns:
            self.df["cleaned_date"] = pd.NaT
            if self.verbose:
                print("⚠️ 'cleaned_date' column missing. Initialized with NaT.")
        else:
            self.df["cleaned_date"] = pd.to_datetime(self.df["cleaned_date"], errors="coerce")
            if self.df["cleaned_date"].isna().all():
                print("⚠️ All 'cleaned_date' values are NaT.")
            elif self.df["cleaned_date"].isna().any():
                print(f"⚠️ {self.df['cleaned_date'].isna().sum()} rows have unparseable dates.")
        if self.verbose:
            print(f"✅ EventTimelineAnalyzer initialized with {len(self.df)} rows.")

    def get_event_counts_by_day(self) -> pd.DataFrame:
        if self.df["cleaned_date"].isna().all():
            return pd.DataFrame(columns=["cleaned_date", "event_count"])
        counts = (
            self.df.dropna(subset=["cleaned_date"])
            .groupby("cleaned_date")["event_phrase"]
            .count()
            .reset_index()
            .rename(columns={"event_phrase": "event_count"})
        )
        if self.verbose:
            print(f"🗓️ Computed event counts across {len(counts)} dates.")
        return counts

    def plot_event_timeline(self, rolling_window: int = 3):
        counts_df = self.get_event_counts_by_day().sort_values("cleaned_date")
        if counts_df.empty:
            print("⚠️ Skipping plot. No timeline data available.")
            return
        plt.figure(figsize=(12, 5))
        plt.plot(counts_df["cleaned_date"], counts_df["event_count"], label="Daily Events", marker="o", linewidth=2)
        if rolling_window > 1:
            counts_df["rolling_avg"] = counts_df["event_count"].rolling(rolling_window).mean()
            plt.plot(counts_df["cleaned_date"], counts_df["rolling_avg"], label=f"{rolling_window}-Day Avg", linestyle="--")
        plt.title("📈 Financial Event Count Over Time")
        plt.xlabel("Date")
        plt.ylabel("Event Count")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.show()
        if self.verbose:
            print("📊 Event timeline plot generated.")


class EventExtractorREBEL:
    def __init__(self, model_name: str = "Babelscape/rebel-large", verbose: bool = True):
        self.verbose = verbose
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.generator = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer)
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load REBEL model '{model_name}': {e}")
        if self.verbose:
            print(f"✅ REBEL model '{model_name}' loaded.")

    def _parse_triplets(self, generated_text: str):
        triplet_regex = r"<triplet> (.*?) <relation> (.*?) <object> (.*?)(?=<triplet>|$)"
        return re.findall(triplet_regex, generated_text)

    def extract_triplets(self, headlines: List[str]) -> pd.DataFrame:
        results = []
        if self.verbose:
            print(f"🔍 Extracting REBEL triplets from {len(headlines)} headlines...")
        for headline in headlines:
            try:
                output = self.generator(f"<triplet> {headline}", max_length=256)[0]["generated_text"]
                triplets = self._parse_triplets(output)
                for subj, rel, obj in triplets:
                    results.append({
                        "headline": headline,
                        "subject": subj.strip(),
                        "relation": rel.strip(),
                        "object": obj.strip(),
                    })
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Skipped headline due to extraction error:\n   {headline}\n   {e}")
        df = pd.DataFrame(results)
        if self.verbose:
            print(f"📊 Extracted {len(df)} triplets.")
        return df

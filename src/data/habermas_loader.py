"""Loader for Habermas Machine dataset from Google DeepMind."""

import io
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

from .schemas import Issue, Statement, PreferenceRecord, HabermasBank


# Google Cloud Storage URLs for Habermas Machine data
HABERMAS_DATA_URLS = {
    "candidate_comparisons": "https://storage.googleapis.com/habermas_machine/datasets/hm_all_candidate_comparisons.parquet",
    "final_rankings": "https://storage.googleapis.com/habermas_machine/datasets/hm_all_final_preference_rankings.parquet",
    "position_ratings": "https://storage.googleapis.com/habermas_machine/datasets/hm_all_position_statement_ratings.parquet",
    "round_surveys": "https://storage.googleapis.com/habermas_machine/datasets/hm_all_round_survey_responses.parquet",
}


class HabermasLoader:
    """Load and parse Habermas Machine data."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize loader.

        Args:
            cache_dir: Directory to cache downloaded files. If None, uses data/raw/habermas/.
        """
        if cache_dir is None:
            # Default to project data directory
            cache_dir = Path(__file__).parent.parent.parent / "data" / "raw" / "habermas"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _download_or_load_cached(self, name: str, url: str) -> pd.DataFrame:
        """Download parquet file or load from cache.

        Args:
            name: Dataset name (for caching).
            url: URL to download from.

        Returns:
            Loaded DataFrame.
        """
        cache_path = self.cache_dir / f"{name}.parquet"

        if cache_path.exists():
            print(f"  Loading cached {name}...")
            return pd.read_parquet(cache_path)

        print(f"  Downloading {name}...")
        response = requests.get(url)
        response.raise_for_status()

        # Save to cache
        with open(cache_path, "wb") as f:
            f.write(response.content)

        # Load and return
        with io.BytesIO(response.content) as f:
            return pd.read_parquet(f)

    def load_candidate_comparisons(self) -> pd.DataFrame:
        """Load the main candidate comparisons dataset."""
        return self._download_or_load_cached(
            "candidate_comparisons",
            HABERMAS_DATA_URLS["candidate_comparisons"]
        )

    def extract_issues(self, df: pd.DataFrame) -> Dict[str, Issue]:
        """Extract unique issues from the dataset.

        Args:
            df: Candidate comparisons DataFrame.

        Returns:
            Dict mapping issue_id to Issue objects.
        """
        issues = {}

        # Get unique questions
        question_cols = ["question.id", "question.text", "question.topic",
                        "question.affirming_statement", "question.negating_statement",
                        "question.split"]
        questions_df = df[question_cols].drop_duplicates("question.id")

        for _, row in questions_df.iterrows():
            issue_id = row["question.id"]
            issues[issue_id] = Issue(
                issue_id=issue_id,
                issue_text=row["question.text"],
                topic_id=int(row["question.topic"]) if pd.notna(row["question.topic"]) else None,
                affirming_statement=row["question.affirming_statement"],
                negating_statement=row["question.negating_statement"],
                split=row["question.split"],
            )

        return issues

    def extract_statements(self, df: pd.DataFrame) -> Dict[str, Statement]:
        """Extract unique statements from the dataset.

        Args:
            df: Candidate comparisons DataFrame.

        Returns:
            Dict mapping statement_id to Statement objects.
        """
        statements = {}

        for _, row in df.iterrows():
            issue_id = row["question.id"]
            candidate_ids = row["candidates.metadata.id"]
            candidate_texts = row["candidates.text"]
            candidate_labels = row["candidates.display_label"]
            candidate_provenances = row["candidates.metadata.provenance"]
            parent_ids = row["candidates.parent_statement_ids"]

            if candidate_ids is None:
                continue

            for i, stmt_id in enumerate(candidate_ids):
                if stmt_id in statements:
                    continue

                statements[stmt_id] = Statement(
                    statement_id=stmt_id,
                    issue_id=issue_id,
                    text=candidate_texts[i] if candidate_texts is not None else "",
                    display_label=candidate_labels[i] if candidate_labels is not None else None,
                    provenance=candidate_provenances[i] if candidate_provenances is not None else None,
                    parent_statement_ids=list(parent_ids[i]) if parent_ids is not None and parent_ids[i] is not None else None,
                )

        return statements

    def extract_preferences(self, df: pd.DataFrame) -> List[PreferenceRecord]:
        """Extract preference records from the dataset.

        Args:
            df: Candidate comparisons DataFrame.

        Returns:
            List of PreferenceRecord objects.
        """
        preferences = []

        for _, row in df.iterrows():
            pref_id = row["metadata.id"]
            participant_id = row["metadata.participant_id"]
            issue_id = row["question.id"]

            # Skip if missing key data
            if pd.isna(pref_id) or pd.isna(participant_id):
                continue

            candidate_ids = row["candidates.metadata.id"]
            rankings = row["rankings.numerical_ranks"]
            agreements = row["ratings.agreement"]
            quality = row["ratings.quality"]
            own_opinion = row["own_opinion.text"]

            if candidate_ids is None or rankings is None:
                continue

            preferences.append(PreferenceRecord(
                preference_id=pref_id,
                participant_id=participant_id,
                issue_id=issue_id,
                statement_ids=list(candidate_ids),
                rankings=list(rankings),
                agreements=list(agreements) if agreements is not None else None,
                quality_ratings=list(quality) if quality is not None else None,
                own_opinion_text=own_opinion if pd.notna(own_opinion) else None,
            ))

        return preferences

    def load(self) -> HabermasBank:
        """Load complete Habermas Machine data.

        Returns:
            HabermasBank containing all issues, statements, and preferences.
        """
        print("Loading Habermas Machine data...")

        # Load main dataset
        df = self.load_candidate_comparisons()
        print(f"  Loaded {len(df)} records")

        # Extract components
        print("  Extracting issues...")
        issues = self.extract_issues(df)
        print(f"    Found {len(issues)} unique issues")

        print("  Extracting statements...")
        statements = self.extract_statements(df)
        print(f"    Found {len(statements)} unique statements")

        print("  Extracting preferences...")
        preferences = self.extract_preferences(df)
        print(f"    Found {len(preferences)} preference records")

        return HabermasBank(
            issues=issues,
            statements=statements,
            preferences=preferences,
        )


def load_habermas_bank(cache_dir: Optional[Path] = None) -> HabermasBank:
    """Convenience function to load Habermas data.

    Args:
        cache_dir: Optional directory to cache downloaded files.

    Returns:
        HabermasBank with all data.
    """
    loader = HabermasLoader(cache_dir=cache_dir)
    return loader.load()

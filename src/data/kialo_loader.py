"""Loader for Kialo debate topics from HuggingFace."""

from typing import Dict

from datasets import load_dataset, concatenate_datasets

from .schemas import Issue, HabermasBank


class KialoLoader:
    """Load Kialo debate topics from HuggingFace dataset.

    The Kialo dataset contains debate topics with pro/con perspectives,
    suitable for generating preference triplets.

    Dataset: timchen0618/Kialo
    - 1,032 debate topics (774 test + 258 validation)
    - Each topic has: question, perspectives (pro/con), type, id
    """

    def __init__(self):
        """Initialize the Kialo loader."""
        pass

    def load(self) -> HabermasBank:
        """Load Kialo issues, return in HabermasBank format for compatibility.

        Returns:
            HabermasBank containing Kialo issues with empty statements/preferences.
        """
        print("Loading Kialo dataset from HuggingFace...")

        # Load test and validation splits
        ds_test = load_dataset("timchen0618/Kialo", split="test")
        ds_val = load_dataset("timchen0618/Kialo", split="validation")
        ds = concatenate_datasets([ds_test, ds_val])

        print(f"  Loaded {len(ds)} debate topics ({len(ds_test)} test + {len(ds_val)} validation)")

        issues: Dict[str, Issue] = {}
        for ex in ds:
            issue_id = f"kialo_{ex['id']}"
            perspectives = ex.get("perspectives", [])

            issues[issue_id] = Issue(
                issue_id=issue_id,
                issue_text=ex["question"],
                affirming_statement=perspectives[0] if len(perspectives) > 0 else None,
                negating_statement=perspectives[1] if len(perspectives) > 1 else None,
            )

        print(f"  Extracted {len(issues)} issues")

        return HabermasBank(issues=issues, statements={}, preferences=[])


def load_kialo_bank() -> HabermasBank:
    """Convenience function to load Kialo data.

    Returns:
        HabermasBank with Kialo issues.
    """
    loader = KialoLoader()
    return loader.load()

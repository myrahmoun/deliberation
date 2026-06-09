"""Dataset classes for triplet-based embedding training."""

import json
from pathlib import Path
from typing import List, Union

from datasets import Dataset as HFDataset
from sentence_transformers import InputExample
from torch.utils.data import Dataset

from src.generation.schemas import Triplet


class TripletDataset(Dataset):
    """Dataset for loading triplets for contrastive learning.

    Uses (anchor, positive) pairs for MultipleNegativesRankingLoss,
    which uses in-batch negatives for efficient training.
    """

    def __init__(self, triplets: List[Triplet]):
        """Initialize dataset with triplets.

        Args:
            triplets: List of Triplet objects
        """
        self.triplets = triplets

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> InputExample:
        """Get a training example.

        Returns full (anchor, positive, negative) triplet for TripletLoss.
        This directly optimizes: sim(anchor, pos) > sim(anchor, neg)

        Args:
            idx: Index of the triplet

        Returns:
            InputExample with (anchor_text, pos_text, neg_text) triplet
        """
        t = self.triplets[idx]
        return InputExample(texts=[t.anchor_text, t.pos_text, t.neg_text])

    @classmethod
    def from_jsonl(cls, path: Union[str, Path]) -> "TripletDataset":
        """Load dataset from JSONL file.

        Args:
            path: Path to JSONL file containing triplets

        Returns:
            TripletDataset instance
        """
        path = Path(path)
        triplets = []

        with open(path, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                triplet = Triplet(**data)
                triplets.append(triplet)

        return cls(triplets)

    def split(
        self, train_ratio: float = 0.9, seed: int = 42
    ) -> tuple["TripletDataset", "TripletDataset"]:
        """Split dataset into train and validation sets.

        Args:
            train_ratio: Fraction of data for training
            seed: Random seed for reproducibility

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        import random

        random.seed(seed)
        indices = list(range(len(self.triplets)))
        random.shuffle(indices)

        split_idx = int(len(indices) * train_ratio)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        train_triplets = [self.triplets[i] for i in train_indices]
        val_triplets = [self.triplets[i] for i in val_indices]

        return TripletDataset(train_triplets), TripletDataset(val_triplets)

    def to_hf_dataset(self) -> HFDataset:
        """Convert to HuggingFace Dataset for SentenceTransformerTrainer.

        Returns:
            HuggingFace Dataset with anchor, positive, negative columns
        """
        data = {
            "anchor": [t.anchor_text for t in self.triplets],
            "positive": [t.pos_text for t in self.triplets],
            "negative": [t.neg_text for t in self.triplets],
        }
        return HFDataset.from_dict(data)


class InBatchNegativesDataset(Dataset):
    """Dataset for classic contrastive learning baseline.

    Uses (anchor, positive) pairs with MultipleNegativesRankingLoss.
    Negatives are sampled randomly from other examples in the batch.

    This tests: does explicit negative supervision from preference
    rankings help vs random in-batch negatives?
    """

    def __init__(self, triplets: List[Triplet]):
        """Initialize dataset with triplets.

        Args:
            triplets: List of Triplet objects
        """
        self.triplets = triplets

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> InputExample:
        """Get a training example as (anchor, positive) pair.

        Negatives come from other samples in the batch during training.

        Args:
            idx: Index

        Returns:
            InputExample with (anchor_text, pos_text) pair
        """
        t = self.triplets[idx]
        return InputExample(texts=[t.anchor_text, t.pos_text])

    @classmethod
    def from_jsonl(cls, path: Union[str, Path]) -> "InBatchNegativesDataset":
        """Load dataset from JSONL file.

        Args:
            path: Path to JSONL file containing triplets

        Returns:
            InBatchNegativesDataset instance
        """
        path = Path(path)
        triplets = []

        with open(path, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                triplet = Triplet(**data)
                triplets.append(triplet)

        return cls(triplets)

    def to_hf_dataset(self) -> HFDataset:
        """Convert to HuggingFace Dataset for SentenceTransformerTrainer.

        For in-batch negatives, we only need anchor and positive.

        Returns:
            HuggingFace Dataset with anchor, positive columns
        """
        data = {
            "anchor": [t.anchor_text for t in self.triplets],
            "positive": [t.pos_text for t in self.triplets],
        }
        return HFDataset.from_dict(data)

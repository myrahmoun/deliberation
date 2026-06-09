"""Embedding module for preference-aware sentence embeddings."""

from .dataset import TripletDataset
from .model import load_model, save_model
from .trainer import EmbeddingTrainer, TrainingConfig

__all__ = [
    "TripletDataset",
    "EmbeddingTrainer",
    "TrainingConfig",
    "load_model",
    "save_model",
]

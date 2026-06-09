"""Learned scoring functions (probes) on top of frozen embeddings.

Trains lightweight models to predict preference from embedding pairs,
comparing against raw cosine similarity.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class LinearProbe(nn.Module):
    """Linear scoring: score = w^T (emb_a * emb_b) + b"""

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, 1)

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        return self.linear(emb_a * emb_b).squeeze(-1)


class BilinearProbe(nn.Module):
    """Bilinear scoring: score = emb_a^T W emb_b"""

    def __init__(self, dim: int):
        super().__init__()
        self.bilinear = nn.Bilinear(dim, dim, 1)

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        return self.bilinear(emb_a, emb_b).squeeze(-1)


class PSDProbe(nn.Module):
    """Positive semi-definite kernel: score = emb_a^T (L L^T) emb_b

    Guarantees the learned kernel is PSD by parameterizing W = L L^T.
    """

    def __init__(self, dim: int, rank: int = 64):
        super().__init__()
        self.L = nn.Parameter(torch.randn(dim, rank) * 0.01)

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        a_proj = emb_a @ self.L  # (batch, rank)
        b_proj = emb_b @ self.L  # (batch, rank)
        return (a_proj * b_proj).sum(dim=-1)


class MLPProbe(nn.Module):
    """2-layer MLP on concatenated/combined embeddings."""

    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([emb_a, emb_b, emb_a * emb_b], dim=-1)
        return self.net(combined).squeeze(-1)


class TwoTowerMLPProbe(nn.Module):
    """Two-tower MLP: apply the same MLP to each text independently, then dot product.

    Produces a proper embedding space (unlike MLPProbe which fuses (a,b) into a
    scalar score). Tests whether intra-text nonlinearity beyond bilinear matters:
      e(x) = MLP(psi(x))
      score(a, b) = e(a) . e(b)
    """

    def __init__(self, dim: int, hidden: int = 256, output: int = 20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, output),
        )

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        e_a = self.net(emb_a)
        e_b = self.net(emb_b)
        return (e_a * e_b).sum(dim=-1)


PROBE_REGISTRY = {
    "linear": LinearProbe,
    "bilinear": BilinearProbe,
    "psd": PSDProbe,
    "mlp": MLPProbe,
    "two_tower": TwoTowerMLPProbe,
}


def _resolve_probe(probe_type: str, embedding_dim: int):
    """Build a probe from a type string.

    Supports parametric names like:
      - "psd_r20"           → PSDProbe(dim, rank=20)
      - "psd_r128"          → PSDProbe(dim, rank=128)
      - "mlp_h256"          → MLPProbe(dim, hidden=256)
      - "two_tower_h256_o20"→ TwoTowerMLPProbe(dim, hidden=256, output=20)
      - "two_tower_h256"    → TwoTowerMLPProbe(dim, hidden=256, output=20)  (default output)
    """
    if probe_type.startswith("psd_r"):
        rank = int(probe_type.split("_r", 1)[1])
        return PSDProbe(embedding_dim, rank=rank)
    if probe_type.startswith("two_tower"):
        parts = probe_type.split("_")
        hidden = 256
        output = 20
        for p in parts:
            if p.startswith("h"):
                try: hidden = int(p[1:])
                except ValueError: pass
            elif p.startswith("o"):
                try: output = int(p[1:])
                except ValueError: pass
        return TwoTowerMLPProbe(embedding_dim, hidden=hidden, output=output)
    if probe_type.startswith("mlp_h"):
        hidden = int(probe_type.split("_h", 1)[1])
        return MLPProbe(embedding_dim, hidden=hidden)
    if probe_type not in PROBE_REGISTRY:
        raise ValueError(f"Unknown probe type: {probe_type}")
    return PROBE_REGISTRY[probe_type](embedding_dim)


class ProbeTrainer:
    """Train and evaluate probes on frozen embeddings."""

    def __init__(
        self,
        probe_type: str,
        embedding_dim: int,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 256,
        device: str = "cpu",
        weight_decay: float = 0.0,
    ):
        self.probe = _resolve_probe(probe_type, embedding_dim).to(device)
        self.weight_decay = weight_decay
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

    def train(
        self,
        anchor_embs: np.ndarray,
        pos_embs: np.ndarray,
        neg_embs: np.ndarray,
        val_anchor: np.ndarray = None,
        val_pos: np.ndarray = None,
        val_neg: np.ndarray = None,
    ) -> Dict[str, float]:
        """Train probe on triplets using Bradley-Terry loss.

        Returns dict with final train/val accuracy.
        """
        anchor_t = torch.tensor(anchor_embs, dtype=torch.float32, device=self.device)
        pos_t = torch.tensor(pos_embs, dtype=torch.float32, device=self.device)
        neg_t = torch.tensor(neg_embs, dtype=torch.float32, device=self.device)

        dataset = TensorDataset(anchor_t, pos_t, neg_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.probe.parameters(), lr=self.lr,
                                      weight_decay=self.weight_decay)

        best_val_acc = 0.0
        for epoch in range(self.epochs):
            self.probe.train()
            total_loss = 0
            for a, p, n in loader:
                pos_score = self.probe(a, p)
                neg_score = self.probe(a, n)
                loss = -F.logsigmoid(pos_score - neg_score).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if val_anchor is not None and (epoch + 1) % 10 == 0:
                val_acc = self.triplet_accuracy(val_anchor, val_pos, val_neg)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc

        train_acc = self.triplet_accuracy(anchor_embs, pos_embs, neg_embs)
        results = {"train_accuracy": train_acc}

        if val_anchor is not None:
            val_acc = self.triplet_accuracy(val_anchor, val_pos, val_neg)
            results["val_accuracy"] = val_acc
            results["best_val_accuracy"] = max(best_val_acc, val_acc)

        return results

    def triplet_accuracy(
        self, anchor_embs: np.ndarray, pos_embs: np.ndarray, neg_embs: np.ndarray,
    ) -> float:
        """Fraction of triplets where score(anchor, pos) > score(anchor, neg)."""
        self.probe.eval()
        with torch.no_grad():
            a = torch.tensor(anchor_embs, dtype=torch.float32, device=self.device)
            p = torch.tensor(pos_embs, dtype=torch.float32, device=self.device)
            n = torch.tensor(neg_embs, dtype=torch.float32, device=self.device)
            pos_score = self.probe(a, p)
            neg_score = self.probe(a, n)
            correct = (pos_score > neg_score).float().mean().item()
        return correct

    def score(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """Score a single pair of embeddings."""
        self.probe.eval()
        with torch.no_grad():
            a = torch.tensor(emb_a, dtype=torch.float32, device=self.device).unsqueeze(0)
            b = torch.tensor(emb_b, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.probe(a, b).item()

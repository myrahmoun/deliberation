"""Unified evaluation on precomputed eval triplets.

Reads flat eval triplets from data/processed/eval/{dataset}.jsonl.
Each line:
    {"anchor_texts": [...], "preferred": "...", "dispreferred": "...", "dataset": "...", "participant_id": "..."}

Evaluation: encode anchor texts (mean-pool), encode preferred/dispreferred,
check if cosine(anchor, preferred) > cosine(anchor, dispreferred).
"""

import json
import logging
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)

SPLIT_SEED = 0
VAL_RATIO = 0.7

EVAL_DATASETS = [
    "gsc_abortion_gen",
    "gsc_abortion_val",
    "gsc_chatbot_gen",
    "remesh_campus_protests",
    "remesh_foreign_intervention",
    "remesh_right_to_assemble",
    "polis_15_per_hour_seattle",
    "polis_american_assembly_bowling_green",
    "polis_brexit_consensus",
    "polis_canadian_electoral_reform",
    "polis_scoop_hivemind_ubi",
]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def split_participants(
    participant_ids: List, split: str, seed: int = SPLIT_SEED, val_ratio: float = VAL_RATIO,
) -> Set:
    """Deterministic val/test split. Returns set of selected IDs."""
    if split == "all":
        return set(participant_ids)
    ids = sorted(set(participant_ids))
    rng = random.Random(seed)
    rng.shuffle(ids)
    cutoff = int(len(ids) * val_ratio)
    if split == "val":
        return set(ids[:cutoff])
    else:
        return set(ids[cutoff:])


def compute_gains(
    base_metrics: Dict[str, float],
    tuned_metrics: Dict[str, float],
    exclude: Optional[Set[str]] = None,
) -> Dict[str, float]:
    exclude = exclude or set()
    gains = {}
    for k in base_metrics:
        if k in exclude or k not in tuned_metrics:
            continue
        gains[k] = tuned_metrics[k] - base_metrics[k]
    if not gains:
        return {"mean_gain": 0.0, "min_gain": 0.0, "n_improved": 0, "n_total": 0}
    return {
        "per_dataset": gains,
        "mean_gain": sum(gains.values()) / len(gains),
        "min_gain": min(gains.values()),
        "n_improved": sum(1 for g in gains.values() if g > 0),
        "n_total": len(gains),
    }


def compute_objective(base_metrics: Dict[str, float], tuned_metrics: Dict[str, float]) -> float:
    """mean_gain + 0.5 * min_gain. For hyperparameter selection on val split."""
    g = compute_gains(base_metrics, tuned_metrics)
    return g["mean_gain"] + 0.5 * g["min_gain"]


class EmbeddingEvaluator:
    """Evaluate an embedding model on precomputed eval triplets."""

    def __init__(
        self,
        eval_dir: Path,
        split_seed: int = SPLIT_SEED,
        val_ratio: float = VAL_RATIO,
        score_fn: Callable = cosine_similarity,
    ):
        self.eval_dir = Path(eval_dir)
        self.split_seed = split_seed
        self.val_ratio = val_ratio
        self.score_fn = score_fn

    def evaluate_dataset(self, model, dataset_name: str, split: str = "val",
                          formatter=None) -> Optional[float]:
        """Evaluate on a single dataset's eval triplets.

        Returns fraction of triplets where score(anchor, preferred) > score(anchor, dispreferred).

        If ``formatter`` is provided (a src.evaluation.formatters.* instance),
        anchors are encoded via formatter.encode_queries and items via
        formatter.encode_passages — lets retrieval-trained encoders use
        their native query/passage asymmetry. Default (None) falls back to
        plain symmetric `model.encode(text)` for backward compatibility.
        """
        data_path = self.eval_dir / f"{dataset_name}.jsonl"
        if not data_path.exists():
            logger.warning(f"Dataset not found: {data_path}")
            return None

        # Load triplets
        triplets = []
        with open(data_path) as f:
            for line in f:
                triplets.append(json.loads(line))

        if not triplets:
            return None

        # Split by participant
        all_pids = [t["participant_id"] for t in triplets]
        selected_pids = split_participants(all_pids, split, self.split_seed, self.val_ratio)
        triplets = [t for t in triplets if t["participant_id"] in selected_pids]

        if len(triplets) < 10:
            return None

        # Collect anchor and item texts separately so we can encode each
        # side with the appropriate prefix when a formatter is given.
        anchor_texts = sorted({a for t in triplets for a in t["anchor_texts"]})
        item_texts = sorted({t["preferred"] for t in triplets} |
                            {t["dispreferred"] for t in triplets})

        if formatter is None:
            all_texts = list(set(anchor_texts) | set(item_texts))
            # Modest batch size: ST5-XL on a 20-GiB MIG slice OOMs at the
            # SentenceTransformer default (32) when the model still holds
            # post-training memory.
            embs = model.encode(all_texts, convert_to_numpy=True,
                                 show_progress_bar=False, batch_size=16)
            text_to_emb = dict(zip(all_texts, embs))
            anchor_to_emb = text_to_emb
            item_to_emb = text_to_emb
        else:
            a_emb = formatter.encode_queries(model, anchor_texts)
            i_emb = formatter.encode_passages(model, item_texts)
            anchor_to_emb = dict(zip(anchor_texts, a_emb))
            item_to_emb = dict(zip(item_texts, i_emb))

        # Cache mean-pooled anchor embeddings per participant
        anchor_cache = {}
        correct = 0
        total = 0

        for t in triplets:
            pid = t["participant_id"]
            if pid not in anchor_cache:
                a_embs = [anchor_to_emb[txt] for txt in t["anchor_texts"] if txt in anchor_to_emb]
                if not a_embs:
                    continue
                anchor = np.mean(a_embs, axis=0)
                if np.linalg.norm(anchor) == 0:
                    continue
                anchor_cache[pid] = anchor

            anchor = anchor_cache.get(pid)
            if anchor is None:
                continue

            pref_emb = item_to_emb.get(t["preferred"])
            disp_emb = item_to_emb.get(t["dispreferred"])
            if pref_emb is None or disp_emb is None:
                continue
            if np.linalg.norm(pref_emb) == 0 or np.linalg.norm(disp_emb) == 0:
                continue

            pref_score = self.score_fn(anchor, pref_emb)
            disp_score = self.score_fn(anchor, disp_emb)

            total += 1
            if pref_score > disp_score:
                correct += 1
            elif pref_score == disp_score:
                correct += 0.5

        if total < 10:
            return None
        return correct / total

    def evaluate_all(self, model, split: str = "val", formatter=None) -> Dict[str, float]:
        """Evaluate on all 11 datasets."""
        results = {}
        for name in EVAL_DATASETS:
            logger.info(f"  Evaluating {name} ({split})...")
            acc = self.evaluate_dataset(model, name, split=split, formatter=formatter)
            if acc is not None:
                results[name] = acc
        return results

    def get_baselines(self, base_model_name: str, cache_dir: Path) -> Dict[str, Dict[str, float]]:
        """Compute or load cached base model metrics."""
        from src.embedding.model import get_device, load_model

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        baselines_file = cache_dir / "baselines.json"

        if baselines_file.exists():
            try:
                with open(baselines_file) as f:
                    baselines = json.load(f)
                if "val" in baselines and "test" in baselines:
                    logger.info(f"Loaded cached baselines from {baselines_file}")
                    return baselines
            except (json.JSONDecodeError, IOError):
                pass

        logger.info("Computing base model metrics...")
        device = get_device()
        base_model = load_model(base_model_name, device=device)

        baselines = {
            "val": self.evaluate_all(base_model, split="val"),
            "test": self.evaluate_all(base_model, split="test"),
        }

        import torch
        del base_model
        torch.cuda.empty_cache()

        with open(baselines_file, "w") as f:
            json.dump(baselines, f, indent=2)

        return baselines

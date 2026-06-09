"""Generate cross-topic triplets using o4-mini as a calibrated probability judge.

For each triplet:
1. Sample a random anchor opinion from a random issue
2. Sample two candidate opinions from a different issue
3. Ask o4-mini for a calibrated probability that the anchor person prefers candidate A vs B
"""

import json
import logging
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

from openai import OpenAI

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """\
A person wrote the following opinion in a survey:

"{anchor}"

Which of these two opinions on a different topic would this person be more likely to agree with?

A: "{candidate_a}"
B: "{candidate_b}"

Give a calibrated probability that the person would prefer opinion A over opinion B. \
Respond with just a single number between 0 and 1 (e.g., 0.7 means 70% chance they prefer A)."""


class CrossTopicJudge:
    """Generate cross-topic preference triplets using o4-mini."""

    def __init__(self, client: OpenAI = None, model: str = "o4-mini"):
        self.client = client or OpenAI()
        self.model = model

    def judge(self, anchor: str, candidate_a: str, candidate_b: str) -> Dict:
        """Ask for calibrated probability that anchor person prefers A over B.

        Returns dict with prob_a and raw response, or None on failure.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                    anchor=anchor, candidate_a=candidate_a, candidate_b=candidate_b,
                )}],
                max_completion_tokens=1024,
            )

            text = response.choices[0].message.content.strip()

            # Parse probability from response
            # o4-mini might include reasoning, so find the number
            import re
            numbers = re.findall(r'(?:^|\s)(0\.\d+|1\.0|0|1)(?:\s|$|\.)', text)
            if not numbers:
                # Try to find any float in the response
                numbers = re.findall(r'(\d+\.?\d*)', text)

            if numbers:
                prob_a = float(numbers[-1])  # Take the last number (usually the final answer)
                if prob_a > 1:
                    prob_a = prob_a / 100  # Handle percentage responses
                prob_a = max(0, min(1, prob_a))
            else:
                return None

            return {
                "prob_a": prob_a,
                "raw_response": text,
            }

        except Exception as e:
            logger.warning(f"Judge failed: {e}")
            return None

    def generate_triplets(
        self,
        opinions_data: List[dict],
        issue_ids: set,
        n_triplets: int = 3000,
        max_workers: int = 25,
        seed: int = 42,
    ) -> Tuple[List[dict], Dict]:
        """Generate cross-topic triplets.

        Returns (triplets, stats).
        """
        issues = [d for d in opinions_data if d["issue_id"] in issue_ids and len(d["opinions"]) == 5]
        if len(issues) < 2:
            logger.error("Need at least 2 issues")
            return [], {}

        rng = random.Random(seed)

        candidates = []
        for _ in range(n_triplets):
            anchor_idx = rng.randrange(len(issues))
            anchor_pos = rng.randrange(5)

            cand_idx = rng.randrange(len(issues) - 1)
            if cand_idx >= anchor_idx:
                cand_idx += 1

            positions = list(range(5))
            rng.shuffle(positions)
            cand_pos_1, cand_pos_2 = positions[0], positions[1]

            candidates.append((anchor_idx, anchor_pos, cand_idx, cand_pos_1, cand_pos_2))

        logger.info(f"Judging {len(candidates)} cross-topic pairs with {max_workers} workers...")

        triplets = []
        all_probs = []
        all_confidences = []
        write_lock = threading.Lock()
        done = 0
        failed = 0

        def _judge_one(args):
            anchor_idx, anchor_pos, cand_idx, cand_pos_1, cand_pos_2 = args
            anchor_text = issues[anchor_idx]["opinions"][anchor_pos]
            cand_1_text = issues[cand_idx]["opinions"][cand_pos_1]
            cand_2_text = issues[cand_idx]["opinions"][cand_pos_2]

            result = self.judge(anchor_text, cand_1_text, cand_2_text)
            if result is None:
                return None

            prob_a = result["prob_a"]
            confidence = abs(prob_a - 0.5) * 2

            if prob_a >= 0.5:
                preferred_text, dispreferred_text = cand_1_text, cand_2_text
                pref_pos, disp_pos = cand_pos_1, cand_pos_2
                win_prob = prob_a
            else:
                preferred_text, dispreferred_text = cand_2_text, cand_1_text
                pref_pos, disp_pos = cand_pos_2, cand_pos_1
                win_prob = 1 - prob_a

            return {
                "anchor_text": anchor_text,
                "pos_text": preferred_text,
                "neg_text": dispreferred_text,
                "anchor_issue_id": issues[anchor_idx]["issue_id"],
                "pos_issue_id": issues[cand_idx]["issue_id"],
                "neg_issue_id": issues[cand_idx]["issue_id"],
                "triplet_type": "cross_topic",
                "anchor_position": anchor_pos + 1,
                "preferred_position": pref_pos + 1,
                "dispreferred_position": disp_pos + 1,
                "judge_prob": win_prob,
                "judge_confidence": confidence,
            }

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_judge_one, c): c for c in candidates}
            for future in as_completed(futures):
                result = future.result()
                done += 1
                if result is not None:
                    with write_lock:
                        triplets.append(result)
                        all_probs.append(result["judge_prob"])
                        all_confidences.append(result["judge_confidence"])
                else:
                    failed += 1
                if done % 500 == 0:
                    logger.info(f"  {done}/{len(candidates)} judged ({len(triplets)} triplets, {failed} failed)")

        import numpy as np
        stats = {
            "n_triplets": len(triplets),
            "n_failed": failed,
            "mean_judge_prob": float(np.mean(all_probs)) if all_probs else 0,
            "mean_confidence": float(np.mean(all_confidences)) if all_confidences else 0,
            "median_confidence": float(np.median(all_confidences)) if all_confidences else 0,
            "pct_confident": float(np.mean(np.array(all_confidences) > 0.5)) if all_confidences else 0,
            "pct_uncertain": float(np.mean(np.array(all_confidences) < 0.2)) if all_confidences else 0,
        }

        logger.info(f"Generated {len(triplets)} cross-topic triplets ({failed} failed)")
        logger.info(f"  Judge prob (winner): mean={stats['mean_judge_prob']:.3f}")
        logger.info(f"  Confidence: mean={stats['mean_confidence']:.3f}, median={stats['median_confidence']:.3f}")
        logger.info(f"  Confident (>0.5): {stats['pct_confident']:.1%}, Uncertain (<0.2): {stats['pct_uncertain']:.1%}")

        return triplets, stats

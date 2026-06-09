"""Generate diverse opinions per issue using an LLM.

For each issue, generates 5 opinions spanning the ideological spectrum
from strongly supportive (position 1) to strongly opposed (position 5).
"""

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

logger = logging.getLogger(__name__)

OPINION_PROMPT = """\
Topic: {issue_text}

Write 5 standalone opinions on this topic from 5 different people. These should \
read like things people wrote in a forum --- NOT like they are answering a question.

The 5 opinions should be clearly distinct and evenly spaced across the full opinion \
spectrum:
1. Strongly supportive --- fully committed to this position
2. Moderately supportive --- in favor but with reservations
3. Genuinely ambivalent --- sees valid points on both sides
4. Moderately opposed --- against but acknowledges some merit
5. Strongly opposed --- firmly against

Important:
- Do not start with Yes, No, I agree, I disagree or any response-like framing
- Each opinion should stand alone as a statement of belief
- Make the difference between adjacent positions (e.g., 1 vs 2, 4 vs 5) clear and meaningful
- Keep them short and natural --- average ~25 words, some shorter, some longer

{{"opinions": ["op1", "op2", "op3", "op4", "op5"]}}"""


class OpinionGenerator:
    """Generate diverse opinions per issue."""

    def __init__(self, client, model: str = "claude-sonnet-4-20250514"):
        self.client = client
        self.model = model
        self._is_anthropic = hasattr(client, "messages")

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM and return the text response."""
        if self._is_anthropic:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=1.0,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=1024,
                temperature=1.0,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content

    def generate(self, issue_text: str) -> Optional[List[str]]:
        """Generate 5 opinions for a single issue."""
        try:
            text = self._call_llm(OPINION_PROMPT.format(issue_text=issue_text))

            # Parse JSON from response (handle markdown code blocks)
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]

            data = json.loads(text)
            opinions = data["opinions"]

            if len(opinions) != 5:
                logger.warning(f"Expected 5 opinions, got {len(opinions)}")
                return None

            return opinions

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return None

    def generate_batch(
        self, issues: List[dict], checkpoint_path: str = None, max_workers: int = 50,
    ) -> List[dict]:
        """Generate opinions for a batch of issues with parallel workers.

        Args:
            issues: List of issue dicts with 'issue_id' and 'issue_text'.
            checkpoint_path: If provided, save progress as results come in.
            max_workers: Number of parallel API workers.

        Returns:
            List of dicts with 'issue_id', 'issue_text', 'opinions'.
        """
        results = []
        done_ids = set()
        write_lock = threading.Lock()

        # Resume from checkpoint
        if checkpoint_path:
            try:
                with open(checkpoint_path) as f:
                    for line in f:
                        r = json.loads(line)
                        results.append(r)
                        done_ids.add(r["issue_id"])
                logger.info(f"Resumed from checkpoint: {len(results)} already done")
            except FileNotFoundError:
                pass

        # Filter to remaining issues
        remaining = [issue for issue in issues if issue["issue_id"] not in done_ids]
        if not remaining:
            logger.info("All issues already done")
            return results

        logger.info(f"Generating opinions for {len(remaining)} issues with {max_workers} workers...")

        def _process(issue):
            opinions = self.generate(issue["issue_text"])
            if opinions is None:
                return None
            return {
                "issue_id": issue["issue_id"],
                "issue_text": issue["issue_text"],
                "source": issue.get("source", "unknown"),
                "opinions": opinions,
            }

        done = 0
        failed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_process, issue): issue for issue in remaining}
            for future in as_completed(futures):
                result = future.result()
                done += 1

                if result is None:
                    failed += 1
                    continue

                with write_lock:
                    results.append(result)
                    if checkpoint_path:
                        with open(checkpoint_path, "a") as f:
                            f.write(json.dumps(result) + "\n")

                if done % 100 == 0:
                    logger.info(f"  {done}/{len(remaining)} done "
                                f"({len(results)} total, {failed} failed)")

        logger.info(f"Done: {len(results)} issues with opinions ({failed} failed)")
        return results

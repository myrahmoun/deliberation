"""Filter issues for training data using GPT-4o-mini.

Three filter conditions for ablation:
- "broad": any genuinely debatable topic
- "political": political, policy, and social issues
- "us_civic": issues relevant to US public deliberation
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

from openai import OpenAI

logger = logging.getLogger(__name__)

FILTER_PROMPTS = {
    "broad": """\
You are deciding whether a debate topic is genuinely debatable — meaning \
reasonable, informed people would hold different positions on it.

ACCEPT if:
- Reasonable people would genuinely disagree on this
- There is a spectrum of defensible positions (not just one correct answer)
- The topic is clear and specific enough to form an opinion on

REJECT if:
- There is a clear factual answer (e.g., "Is the earth round?")
- Almost everyone would agree (e.g., "Should we protect children from abuse?")
- The topic is nonsensical, incoherent, or too vague to have a position on
- It is a pure personal preference with no broader implications (e.g., "Is chocolate better than vanilla?")

Topic: "{issue_text}"

Respond with exactly one word: ACCEPT or REJECT""",

    "political": """\
You are deciding whether a debate topic is a political, policy, or social \
issue suitable for studying public opinion and preference diversity.

ACCEPT if:
- It is a political, policy, economic, or social issue
- People's positions reflect their values, ideology, or worldview (not just personal taste)
- It is the kind of topic debated in legislatures, newspapers, public surveys, or town halls
- There is a clear spectrum of positions (e.g., pro-regulation vs. free-market)

REJECT if:
- It is purely factual or scientific with a clear answer
- It is abstract philosophy with no policy relevance (e.g., "Is free will real?")
- It is a niche technical question that only specialists would engage with
- It is about personal lifestyle preferences rather than public affairs
- It has near-universal consensus
- It is nonsensical or too vague

Topic: "{issue_text}"

Respond with exactly one word: ACCEPT or REJECT""",

    "us_civic": """\
You are deciding whether a debate topic is relevant to US public \
deliberation — the kind of issue discussed by American voters, in US town \
halls, or in US policy debates.

ACCEPT if:
- A typical US voter or citizen would have an opinion on this
- It relates to US domestic policy, American social issues, or topics commonly debated in US public life
- It touches on values that divide Americans (e.g., government role, individual rights, social justice, religious liberty, gun policy, immigration, healthcare)
- Even if not US-specific, the topic is actively debated in the US context

REJECT if:
- It is specific to another country with no US relevance (e.g., "Should the UK rejoin the EU?")
- It is purely factual, scientific, or has a clear correct answer
- It is abstract, academic, or niche with no connection to US civic life
- It has near-universal consensus among Americans
- It is nonsensical or too vague

Topic: "{issue_text}"

Respond with exactly one word: ACCEPT or REJECT""",
    "targeted": """\
You are deciding whether a debate topic falls within one of these specific \
policy/social domains:

- Abortion, reproductive rights, bodily autonomy
- AI, chatbots, technology personalization, digital privacy
- Campus protests, student activism, free speech on campus
- Foreign intervention, military policy, defense spending
- Minimum wage, labor rights, worker protections, cost of living
- Community development, local governance, housing, urban planning
- Brexit, EU membership, trade policy, national sovereignty
- Electoral reform, voting systems, democratic representation
- Universal basic income, welfare, social safety nets

ACCEPT if the topic clearly falls within or is closely related to one of \
these domains.

REJECT if the topic is outside all of these domains, even if it is a \
perfectly good debate topic.

Topic: "{issue_text}"

Respond with exactly one word: ACCEPT or REJECT""",
}


class IssueFilter:
    """Filter issues using GPT-4o-mini with configurable filter condition."""

    def __init__(
        self,
        condition: str = "political",
        client: OpenAI = None,
        model: str = "gpt-4o-mini",
    ):
        if condition not in FILTER_PROMPTS:
            raise ValueError(f"Unknown condition: {condition}. Choose from: {list(FILTER_PROMPTS.keys())}")
        self.condition = condition
        self.prompt_template = FILTER_PROMPTS[condition]
        self.client = client or OpenAI()
        self.model = model

    def is_acceptable(self, issue_text: str) -> bool:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": self.prompt_template.format(issue_text=issue_text)}],
            max_tokens=5,
            temperature=0,
        )
        answer = response.choices[0].message.content.strip().upper()
        return answer == "ACCEPT"

    def filter_batch(
        self, issues: List[dict], max_workers: int = 20,
    ) -> Tuple[List[dict], List[dict]]:
        """Filter a list of issues in parallel.

        Each issue must have 'issue_text' key.
        Returns (accepted, rejected) preserving original order.
        """
        results = [None] * len(issues)

        def _classify(idx_issue):
            idx, issue = idx_issue
            try:
                return idx, self.is_acceptable(issue["issue_text"])
            except Exception as e:
                logger.warning(f"Issue {idx} failed: {e}")
                return idx, False

        done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_classify, (i, issue)): i for i, issue in enumerate(issues)}
            for future in as_completed(futures):
                idx, accepted = future.result()
                results[idx] = accepted
                done += 1
                if done % 200 == 0:
                    n_acc = sum(1 for r in results[:done] if r is True)
                    logger.info(f"[{self.condition}] {done}/{len(issues)} done ({n_acc} accepted so far)")

        accepted = [issues[i] for i, r in enumerate(results) if r is True]
        rejected = [issues[i] for i, r in enumerate(results) if r is not True]

        logger.info(f"[{self.condition}] Done: {len(accepted)} accepted, {len(rejected)} rejected")
        return accepted, rejected

"""Baseline embedding and LLM-based preference prediction methods."""

import logging
import time
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class OpenAIEmbedder:
    """OpenAI embedding API wrapper matching SentenceTransformer .encode() interface."""

    def __init__(self, model: str = "text-embedding-3-large", client=None):
        from openai import OpenAI
        self.client = client or OpenAI()
        self.model = model

    def encode(
        self, texts: List[str], convert_to_numpy: bool = True, show_progress_bar: bool = False,
        batch_size: int = 100, **kwargs,
    ) -> np.ndarray:
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.embeddings.create(input=batch, model=self.model)
            embs = [r.embedding for r in response.data]
            all_embs.extend(embs)
            if i + batch_size < len(texts):
                time.sleep(0.1)
        return np.array(all_embs)


class VoyageEmbedder:
    """Voyage AI embedding API wrapper matching SentenceTransformer .encode() interface."""

    def __init__(self, model: str = "voyage-3", client=None):
        import voyageai
        self.client = client or voyageai.Client()
        self.model = model

    def encode(
        self, texts: List[str], convert_to_numpy: bool = True, show_progress_bar: bool = False,
        batch_size: int = 50, input_type: Optional[str] = None, **kwargs,
    ) -> np.ndarray:
        """If input_type is 'query' or 'document', forwarded to the Voyage API
        for asymmetric embedding (recommended for retrieval-style usage)."""
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            call_kwargs = {"model": self.model}
            if input_type:
                call_kwargs["input_type"] = input_type
            result = self.client.embed(batch, **call_kwargs)
            all_embs.extend(result.embeddings)
            if i + batch_size < len(texts):
                time.sleep(0.2)
        return np.array(all_embs)


class LLMDiscriminativeQuery:
    """LLM-based preference prediction via discriminative queries.

    Given a user's written text and two candidate statements, asks the LLM
    which statement the user would prefer.
    """

    TEXT_ONLY_PROMPT = """\
Based on the following text written by a person, predict which of two statements they would rate higher.

Person's writing:
{user_text}

Statement A: {stmt_a}
Statement B: {stmt_b}

Which statement would this person rate higher, A or B? Respond with just the letter."""

    FEWSHOT_PROMPT = """\
You are predicting a person's preferences based on their writing.

{examples}

Now predict for this person:

Person's writing:
{user_text}

Statement A: {stmt_a}
Statement B: {stmt_b}

Which statement would this person rate higher, A or B? Respond with just the letter."""

    def __init__(self, client=None, model: str = "gpt-4o"):
        from openai import OpenAI
        self.client = client or OpenAI()
        self.model = model

    def predict_preference(
        self, user_text: str, stmt_a: str, stmt_b: str,
        examples: str = None,
    ) -> Optional[str]:
        """Predict whether user prefers statement A or B.

        Returns "A" or "B", or None on failure.
        """
        if examples:
            prompt = self.FEWSHOT_PROMPT.format(
                user_text=user_text, stmt_a=stmt_a, stmt_b=stmt_b, examples=examples,
            )
        else:
            prompt = self.TEXT_ONLY_PROMPT.format(
                user_text=user_text, stmt_a=stmt_a, stmt_b=stmt_b,
            )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0,
            )
            answer = response.choices[0].message.content.strip().upper()
            if answer in ("A", "B"):
                return answer
            return None
        except Exception as e:
            logger.warning(f"LLM query failed: {e}")
            return None

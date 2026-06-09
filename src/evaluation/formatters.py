"""Per-model encoding strategy for baseline evaluation.

Many baselines were trained as asymmetric retrieval encoders, expecting
the *query* and *passage* sides to be formatted differently. Treating
them symmetrically (plain `model.encode(text)`) under-represents what
the model can actually do. For our preference task, the user-authored
anchor is the query and the candidate statements are passages.

A `Formatter` exposes two methods:
  encode_queries(model, texts) -> ndarray
  encode_passages(model, texts) -> ndarray

dispatch_formatter(model_id) returns the right Formatter for a given
HuggingFace / API model id, falling back to PlainFormatter for symmetric
encoders (sentence-T5, MiniLM, mpnet, OpenAI text-embedding-3-*, ...).
"""
from __future__ import annotations

import numpy as np
from typing import List


# Default task instruction for instruction-tuned models (Qwen, Stella).
# Targets within-participant ranking: "find statements the same person
# would endorse." Overridable per-model via Formatter constructor.
DEFAULT_TASK_INSTRUCTION = (
    "Given an opinion statement on a contested social or political issue, "
    "retrieve other statements that share the same stance and underlying values."
)

# BGE family query instruction (BGE, BGE-SparseCL, MxBAI, Arctic-Embed all share this).
BGE_QUERY_INSTRUCTION = (
    "Represent this sentence for searching relevant passages: "
)


class PlainFormatter:
    """Symmetric encoders: no prefixes, no special args."""
    name = "plain"

    def encode_queries(self, model, texts: List[str]) -> np.ndarray:
        return model.encode(texts, convert_to_numpy=True,
                             show_progress_bar=False, batch_size=64)

    def encode_passages(self, model, texts: List[str]) -> np.ndarray:
        return self.encode_queries(model, texts)


class PrefixFormatter:
    """Add a string prefix to queries / passages independently.

    Used by:
      e5         query_prefix='query: '   passage_prefix='passage: '
      BGE / etc. query_prefix=BGE_QUERY_INSTRUCTION   passage_prefix=''
    """
    def __init__(self, query_prefix: str = "", passage_prefix: str = "", batch_size: int = 64):
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.batch_size = batch_size
        self.name = f"prefix(q={query_prefix!r}, p={passage_prefix!r})"

    def encode_queries(self, model, texts):
        return model.encode([self.query_prefix + t for t in texts],
                             convert_to_numpy=True, show_progress_bar=False,
                             batch_size=self.batch_size)

    def encode_passages(self, model, texts):
        return model.encode([self.passage_prefix + t for t in texts],
                             convert_to_numpy=True, show_progress_bar=False,
                             batch_size=self.batch_size)


class QwenInstructFormatter:
    """`Instruct: <task>\\nQuery: <text>` for queries; plain for passages.

    Template matches the prompt the model authors register in
    config_sentence_transformers.json (verified). The cached prompt
    uses task='Given a web search query, retrieve relevant passages
    that answer the query' — a query-to-document retrieval framing
    that does not match our sentence-to-sentence preference task.

    We swap in DEFAULT_TASK_INSTRUCTION (opinion-similarity) by
    default — the analogue of choosing Stella's 's2s_query' over
    's2p_query' (sentence-similarity, not search-retrieval). To use
    the model's registered web-search prompt instead, pass
    task='Given a web search query, retrieve relevant passages that
    answer the query'.
    """
    def __init__(self, task: str = DEFAULT_TASK_INSTRUCTION, batch_size: int = 8):
        self.task = task
        self.batch_size = batch_size
        self.name = f"qwen_instruct(task={task!r})"

    def encode_queries(self, model, texts):
        formatted = [f"Instruct: {self.task}\nQuery: {t}" for t in texts]
        return model.encode(formatted, convert_to_numpy=True,
                             show_progress_bar=False, batch_size=self.batch_size)

    def encode_passages(self, model, texts):
        return model.encode(texts, convert_to_numpy=True,
                             show_progress_bar=False, batch_size=self.batch_size)


class StellaFormatter:
    """Instruction-template encoding for dunzhang/stella_en_1.5B_v5.

    Stella was instruction-tuned with the same `Instruct: <task>\\nQuery:`
    template as Qwen-2-Instruct. Its registered prompts are:
      s2p_query: 'Instruct: Given a web search query, retrieve relevant
                  passages that answer the query.\\nQuery: '
      s2s_query: 'Instruct: Retrieve semantically similar text.\\nQuery: '
    Neither matches our preference-similarity task. We therefore swap
    in DEFAULT_TASK_INSTRUCTION as the task description, mirroring how
    QwenInstructFormatter handles its own registered prompt — both
    instruction-tuned models get the same task-tailored treatment.

    To use a registered prompt verbatim instead, pass
    prompt_name='s2s_query' (or 's2p_query') and the model's own
    template string will be applied.
    """
    def __init__(self, task: str = DEFAULT_TASK_INSTRUCTION,
                 prompt_name: str | None = None, batch_size: int = 8):
        self.task = task
        self.prompt_name = prompt_name
        self.batch_size = batch_size
        self.name = (f"stella(prompt_name={prompt_name})" if prompt_name
                     else f"stella(task={task!r})")

    def encode_queries(self, model, texts):
        if self.prompt_name is not None:
            return model.encode(texts, prompt_name=self.prompt_name,
                                 convert_to_numpy=True, show_progress_bar=False,
                                 batch_size=self.batch_size)
        formatted = [f"Instruct: {self.task}\nQuery: {t}" for t in texts]
        return model.encode(formatted, convert_to_numpy=True,
                             show_progress_bar=False, batch_size=self.batch_size)

    def encode_passages(self, model, texts):
        return model.encode(texts, convert_to_numpy=True,
                             show_progress_bar=False, batch_size=self.batch_size)


class VoyageInputTypeFormatter:
    """Voyage embeddings: input_type='query' vs 'document' API parameter.

    Requires VoyageEmbedder.encode to accept an input_type kwarg
    (we extend it elsewhere).
    """
    def __init__(self, batch_size: int = 50):
        self.batch_size = batch_size
        self.name = "voyage(input_type=query/document)"

    def encode_queries(self, model, texts):
        return model.encode(texts, input_type="query",
                             convert_to_numpy=True, show_progress_bar=False,
                             batch_size=self.batch_size)

    def encode_passages(self, model, texts):
        return model.encode(texts, input_type="document",
                             convert_to_numpy=True, show_progress_bar=False,
                             batch_size=self.batch_size)


# ---- dispatch -----------------------------------------------------------

# (model_id_lowercase_substring, formatter_factory) — first match wins.
# Anything not matched falls back to PlainFormatter. Matches are
# case-insensitive (we lowercase the model_id in dispatch_formatter).
#
# Prefix sourcing — verified against each model's HF card / cached
# config_sentence_transformers.json:
#   e5:         query='query: ', passage='passage: '
#   BGE-large:  query=BGE_QUERY_INSTRUCTION, passage=''
#   MxBAI:      same as BGE (per mxbai blog)
#   Arctic v2:  query='query: ' (E5-style; v2 redesign, NOT the v1 BGE prompt)
#   SparseCL:   plain — the authors' own eval script encodes both sides
#               plain; the BGE-style instruction is wrong here.
#   Qwen-2-Instruct / Stella: separate Formatter classes (templates).
_DISPATCH = [
    # e5 family: 'query: ' / 'passage: '
    ("intfloat/e5-",                    lambda: PrefixFormatter(query_prefix="query: ",   passage_prefix="passage: ")),
    # Snowflake Arctic v2: E5-style 'query: ' prefix (v2 redesign, NOT v1's BGE prompt).
    ("snowflake/snowflake-arctic-embed",lambda: PrefixFormatter(query_prefix="query: ",   passage_prefix="")),
    # BGE family + MxBAI: same query instruction, plain passage.
    ("baai/bge-",                       lambda: PrefixFormatter(query_prefix=BGE_QUERY_INSTRUCTION, passage_prefix="")),
    ("mixedbread-ai/mxbai-",            lambda: PrefixFormatter(query_prefix=BGE_QUERY_INSTRUCTION, passage_prefix="")),
    # SparseCL/BGE-SparseCL: authors' own eval is plain-encoded
    # (https://github.com/xuhaike/SparseCL test_contradiction_faiss_final.py).
    # Note: the released checkpoint has no SBERT modules.json. SentenceTransformer
    # wraps it with default mean pooling, which matches the paper's training-time
    # `--pooler_type avg`. Verified.
    ("sparsecl/bge-",                   lambda: PlainFormatter()),
    # Instruction-tuned 1.5B encoders
    ("alibaba-nlp/gte-qwen2",           lambda: QwenInstructFormatter()),
    ("dunzhang/stella_en_1.5b",         lambda: StellaFormatter()),
    # Voyage API uses input_type, not a prefix
    ("voyage-",                         lambda: VoyageInputTypeFormatter()),
]


def dispatch_formatter(model_id: str):
    """Return the appropriate Formatter for `model_id` (case-insensitive match)."""
    needle = model_id.lower()
    for substr, factory in _DISPATCH:
        if substr in needle:
            return factory()
    return PlainFormatter()

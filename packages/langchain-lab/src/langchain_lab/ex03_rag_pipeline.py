"""Lessons 03 & 24: Concrete RAG pipeline (chunk -> index -> retrieve -> grounded answer)."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_lab.config import build_chat_model

SAMPLE_DOC = """
# Industrial AIGC Systems (Notes)

In Lesson 04, Tyler explains why prototype RAG often breaks in production:
1. Naive fixed-size character chunking can split a table or definition in half.
2. Pure vector search misses exact IDs, error codes, or rare product names.
3. Industrial systems combine inverted index (BM25/ES) with dense vectors (ANN/HA3).

In Lesson 24, Alibaba Havenask (HA3) is used for multi-path recall:
- Inverted index handles exact keyword and metadata filters.
- HNSW / PQ vector index handles semantic similarity at millisecond latency.
- A reranker orders the merged candidates before feeding them to the Prompt Engine.
""".strip()


@dataclass(frozen=True)
class Chunk:
    """A single document slice with its source index."""

    chunk_id: int
    text: str


def chunk_by_paragraphs(text: str, max_chars: int = 280) -> list[Chunk]:
    """Split text on paragraph boundaries so ideas stay intact."""
    chunks: list[Chunk] = []
    current: list[str] = []
    length = 0
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if current and length + len(para) > max_chars:
            chunks.append(Chunk(len(chunks), "\n\n".join(current)))
            current = [para]
            length = len(para)
        else:
            current.append(para)
            length += len(para)
    if current:
        chunks.append(Chunk(len(chunks), "\n\n".join(current)))
    return chunks


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def score_overlap(query: str, chunk_text: str) -> float:
    """Simple bag-of-words cosine score so retrieval works offline without an embedding API."""
    q_counts = Counter(_tokenize(query))
    c_counts = Counter(_tokenize(chunk_text))
    if not q_counts or not c_counts:
        return 0.0
    dot = sum(q_counts[t] * c_counts[t] for t in q_counts)
    norm_q = math.sqrt(sum(v * v for v in q_counts.values()))
    norm_c = math.sqrt(sum(v * v for v in c_counts.values()))
    return dot / (norm_q * norm_c)


def retrieve(query: str, chunks: list[Chunk], top_k: int = 2) -> list[tuple[float, Chunk]]:
    """Return the top-k chunks ranked by similarity score."""
    scored = [(score_overlap(query, c.text), c) for c in chunks]
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:top_k]


def main() -> None:
    chunks = chunk_by_paragraphs(SAMPLE_DOC)
    question = "Why does pure vector search fail in production, and what does HA3 use?"
    hits = retrieve(question, chunks, top_k=2)

    print(f"Indexed {len(chunks)} chunks. Top-2 retrieved for question:\n")
    context_blocks = []
    for score, chunk in hits:
        print(f"--- [chunk {chunk.chunk_id} | score={score:.3f}] ---\n{chunk.text}\n")
        context_blocks.append(f"[{chunk.chunk_id}] {chunk.text}")

    prompt = ChatPromptTemplate.from_template(
        "Answer the question using ONLY the context below. Cite chunk IDs like [0].\n\n"
        "Context:\n{context}\n\nQuestion: {question}"
    )
    chain = prompt | build_chat_model() | StrOutputParser()
    print("--- Grounded LLM Answer ---")
    print(chain.invoke({"context": "\n\n".join(context_blocks), "question": question}))


if __name__ == "__main__":
    main()

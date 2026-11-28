"""Lessons 21-22: Generative Agents AssociativeMemory retrieval and optional reflection."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


@dataclass
class MemoryNode:
    """Single entry in an agent's associative memory stream."""

    text: str
    created_step: int
    last_accessed_step: int
    importance: float  # 1.0 - 10.0
    embedding: tuple[float, ...]


def cosine_similarity(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Compute cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class AssociativeMemory:
    """Lesson 21 retrieval: weighted sum of recency, importance, and relevance."""

    def __init__(self, decay: float = 0.99, weights: tuple[float, float, float] = (0.5, 3.0, 2.0)):
        self.decay = decay
        self.weights = weights
        self.nodes: list[MemoryNode] = []

    def add(self, node: MemoryNode) -> None:
        self.nodes.append(node)

    def retrieve(
        self, query_embedding: tuple[float, ...], current_step: int, top_k: int = 3
    ) -> list[tuple[float, MemoryNode]]:
        if not self.nodes:
            return []

        w_rec, w_imp, w_rel = self.weights
        raw_scores = []
        for node in self.nodes:
            recency = self.decay ** max(0, current_step - node.last_accessed_step)
            importance = node.importance / 10.0
            relevance = max(0.0, cosine_similarity(query_embedding, node.embedding))
            score = w_rec * recency + w_imp * importance + w_rel * relevance
            raw_scores.append((score, node))

        raw_scores.sort(key=lambda item: item[0], reverse=True)
        selected = raw_scores[:top_k]
        for _, node in selected:
            node.last_accessed_step = current_step
        return selected


def _demo_memory() -> AssociativeMemory:
    mem = AssociativeMemory()
    mem.add(MemoryNode("Isabella plans a Valentine's Day party at Hobbs Cafe.", 1, 1, 8.0, (0.9, 0.3, 0.1)))
    mem.add(MemoryNode("John Lin watered the garden tomatoes this morning.", 2, 2, 2.0, (0.1, 0.8, 0.2)))
    mem.add(MemoryNode("Maria promised to bring decorations for the cafe party.", 5, 5, 7.0, (0.85, 0.4, 0.1)))
    mem.add(MemoryNode("Sam announced his candidacy for Smallville mayor.", 8, 8, 9.0, (0.2, 0.2, 0.95)))
    return mem


def main() -> None:
    parser = argparse.ArgumentParser(description="Generative Agents AssociativeMemory demo")
    parser.add_argument("--reflect", action="store_true", help="Call LLM to generate insights")
    args = parser.parse_args()

    mem = _demo_memory()
    party_query = (0.9, 0.35, 0.1)
    hits = mem.retrieve(party_query, current_step=10, top_k=2)
    print("Top retrieved memories for 'cafe party':")
    for score, node in hits:
        print(f"  [{score:.3f}] (imp={node.importance}) {node.text}")

    if args.reflect:
        from langchain_lab.config import build_chat_model

        llm = build_chat_model()
        prompt = ChatPromptTemplate.from_template(
            "Given these agent memories:\n{memories}\n\nSynthesize 2 concise high-level insights."
        )
        chain = prompt | llm | StrOutputParser()
        joined = "\n".join(f"- {node.text}" for _, node in hits)
        print("\nReflection insights:\n" + chain.invoke({"memories": joined}))


if __name__ == "__main__":
    main()

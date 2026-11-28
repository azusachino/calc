"""Lessons 19-20: Few-shot example selection + Self-Consistency majority voting."""

from __future__ import annotations

from collections import Counter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

from langchain_lab.config import build_chat_model

EXAMPLES = [
    {
        "input": "A model has 768B total parameters and activates 16B per token. Architecture?",
        "output": "Sparse MoE",
    },
    {
        "input": "A model has 0.8B parameters and runs all 0.8B weights on every token. Architecture?",
        "output": "Dense",
    },
    {
        "input": "A 671B model routes each token to 8 of 256 FFN blocks plus shared experts. Architecture?",
        "output": "Sparse MoE",
    },
]


def majority_vote(answers: list[str]) -> str:
    """Self-Consistency (Lesson 20): pick the most frequent normalized answer."""
    cleaned = [a.strip() for a in answers if a.strip()]
    if not cleaned:
        return ""
    return Counter(cleaned).most_common(1)[0][0]


def main() -> None:
    example_prompt = ChatPromptTemplate.from_messages([("human", "{input}"), ("ai", "{output}")])
    few_shot = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=EXAMPLES,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Classify the architecture in 1-2 words matching the examples."),
            few_shot,
            ("human", "{input}"),
        ]
    )

    llm = build_chat_model(temperature=0.7)
    chain = prompt | llm | StrOutputParser()

    query = "A 32B model uses all 32B parameters for every forward pass. Architecture?"
    samples = [chain.invoke({"input": query}) for _ in range(3)]
    print("Self-Consistency samples:", samples)
    print("Majority vote:", majority_vote(samples))


if __name__ == "__main__":
    main()

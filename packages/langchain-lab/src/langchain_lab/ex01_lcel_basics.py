"""Modernized Lesson 03: LCEL prompt -> model -> parser chain and tool calling."""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from langchain_lab.config import build_chat_model


@tool
def estimate_kv_cache_mb(context_tokens: int, bytes_per_token: int = 890) -> float:
    """Estimate KV cache footprint in MB given context length and bytes per token."""
    return round((context_tokens * bytes_per_token) / (1024 * 1024), 2)


def main() -> None:
    llm = build_chat_model()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer concisely in 2-3 sentences."),
            ("human", "{question}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    print("--- 1. LCEL chain ---")
    print(chain.invoke({"question": "What is the difference between a dense model and MoE?"}))

    print("\n--- 2. Tool binding ---")
    llm_with_tools = llm.bind_tools([estimate_kv_cache_mb])
    msg = llm_with_tools.invoke(
        "How many MB is the KV cache for 100000 tokens at 890 bytes per token?"
    )
    print("tool_calls:", msg.tool_calls)
    if msg.tool_calls:
        call = msg.tool_calls[0]
        print("tool result (MB):", estimate_kv_cache_mb.invoke(call["args"]))


if __name__ == "__main__":
    main()

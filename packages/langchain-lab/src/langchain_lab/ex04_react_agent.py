"""Lessons 02 & 20: Visible ReAct agent loop (Thought -> Tool Call -> Observation -> Answer)."""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from langchain_lab.config import build_chat_model


@tool
def lora_param_count(d_model: int, rank: int, num_layers: int, num_projections: int = 2) -> int:
    """Compute trainable LoRA parameters across transformer layers (e.g., Q and V projections)."""
    per_matrix = 2 * d_model * rank  # A (d x r) + B (r x d)
    return per_matrix * num_projections * num_layers


@tool
def moe_active_ratio(active_experts: int, total_experts: int) -> float:
    """Return the fraction of routed experts activated per token."""
    return round(active_experts / total_experts, 4)


TOOLS = [lora_param_count, moe_active_ratio]
TOOL_MAP = {t.name: t for t in TOOLS}


def execute_tool_call(call: dict) -> str:
    """Run one LangChain tool call dict and return stringified output."""
    fn = TOOL_MAP.get(call["name"])
    if fn is None:
        return f"Unknown tool: {call['name']}"
    return str(fn.invoke(call["args"]))


def run_react_loop(question: str, max_steps: int = 4) -> str:
    """Run a transparent multi-step tool-calling loop so every message is inspectable."""
    llm = build_chat_model().bind_tools(TOOLS)
    messages: list[BaseMessage] = [
        SystemMessage(content="Use the provided tools for math; explain your final answer concisely."),
        HumanMessage(content=question),
    ]

    for step in range(1, max_steps + 1):
        reply: AIMessage = llm.invoke(messages)
        messages.append(reply)

        if not reply.tool_calls:
            print(f"[step {step}] Final answer:\n{reply.content}")
            return str(reply.content)

        for call in reply.tool_calls:
            result = execute_tool_call(call)
            print(f"[step {step}] Tool {call['name']}({call['args']}) -> {result}")
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

    return "Stopped: reached max_steps without a final answer."


def main() -> None:
    run_react_loop(
        "If d_model=4096, LoRA rank=8, 32 layers, and 2 projections (Q,V), "
        "how many trainable LoRA parameters are there? Also what fraction is 8 active out of 256 experts?"
    )


if __name__ == "__main__":
    main()

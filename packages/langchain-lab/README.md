# LangChain Lab (`AI大模型系统实战` companion)

Concrete, runnable lessons pairing `calc` with 《AI大模型系统实战》, using any OpenAI-compatible endpoint (Magpie, OpenRouter free models, or local `llama.cpp` / Ollama).

## Setup

Copy `.env.example` to `vendor/calc/.env.local` (gitignored and loaded automatically by `mise.toml` and `config.py`) or export variables in your shell:

```sh
# Option A: Local / private OpenAI-compatible gateway (put personal URL in untracked .env)
export OPENAI_BASE_URL="http://127.0.0.1:3425/v1"
export MAGPIE_GATEWAY_KEY="sk-..."
export OPENAI_MODEL="openrouter/openrouter/free"

# Option B: OpenRouter free tier
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
export OPENAI_API_KEY="sk-or-v1-..."
export OPENAI_MODEL="meta-llama/llama-3.3-70b-instruct:free"
```

Sync the workspace from `vendor/calc/`:

```sh
uv sync --all-packages
```

## Lessons

1. **Lesson `03` — LCEL + Tool Binding** (`ex01_lcel_basics.py`):
   `prompt | llm | StrOutputParser()` chain plus `@tool` schema binding (`estimate_kv_cache_mb`).
   ```sh
   uv run --package langchain-lab python -m langchain_lab.ex01_lcel_basics
   ```
2. **Lessons `21`–`22` — Generative Agents `AssociativeMemory`** (`ex02_memory_stream.py`):
   Implements `Recency + Importance + Relevance` scoring and `last_accessed_step` updates (runs offline; pass `--reflect` to synthesize insights with the LLM).
   ```sh
   uv run --package langchain-lab python -m langchain_lab.ex02_memory_stream
   ```
3. **Lessons `03` & `24` — Concrete RAG Pipeline** (`ex03_rag_pipeline.py`):
   Paragraph-aware chunking, offline cosine-overlap retrieval, and grounded Q&A with chunk citations.
   ```sh
   uv run --package langchain-lab python -m langchain_lab.ex03_rag_pipeline
   ```
4. **Lessons `02` & `20` — Visible ReAct Agent Loop** (`ex04_react_agent.py`):
   Explicit multi-step `AIMessage -> tool_calls -> ToolMessage -> Final Answer` loop computing LoRA parameters and MoE active ratio.
   ```sh
   uv run --package langchain-lab python -m langchain_lab.ex04_react_agent
   ```
5. **Lessons `19`–`20` — Few-Shot & Self-Consistency** (`ex05_prompt_eval.py`):
   `FewShotChatMessagePromptTemplate` combined with 3-sample majority voting (`Self-Consistency Sampling`).
   ```sh
   uv run --package langchain-lab python -m langchain_lab.ex05_prompt_eval
   ```

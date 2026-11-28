"""Unit tests for langchain-lab config, memory stream, RAG, ReAct tools, and voting."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_lab.config import load_settings
from langchain_lab.ex02_memory_stream import AssociativeMemory, MemoryNode
from langchain_lab.ex03_rag_pipeline import SAMPLE_DOC, chunk_by_paragraphs, retrieve
from langchain_lab.ex04_react_agent import execute_tool_call
from langchain_lab.ex05_prompt_eval import majority_vote


class LangChainLabTests(unittest.TestCase):
    def test_load_settings_reads_dotenv_and_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("OPENAI_API_KEY=sk-test-123\n", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                settings = load_settings(env_path)
                self.assertEqual(settings.api_key, "sk-test-123")
                self.assertEqual(settings.base_url, "https://openrouter.ai/api/v1")

    def test_load_settings_falls_back_to_magpie_gateway_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            with patch.dict(
                os.environ,
                {
                    "MAGPIE_GATEWAY_KEY": "sk-magpie-test",
                    "OPENAI_BASE_URL": "https://gateway.example.test/v1",
                    "OPENAI_MODEL": "openrouter/openrouter/free",
                },
                clear=True,
            ):
                settings = load_settings(env_path)
                self.assertEqual(settings.api_key, "sk-magpie-test")
                self.assertEqual(settings.base_url, "https://gateway.example.test/v1")
                self.assertEqual(settings.model, "openrouter/openrouter/free")

    def test_associative_memory_ranks_and_updates_access_step(self) -> None:
        mem = AssociativeMemory()
        relevant = MemoryNode("party prep", 1, 1, 8.0, (1.0, 0.0))
        irrelevant = MemoryNode("water plants", 9, 9, 2.0, (0.0, 1.0))
        mem.add(relevant)
        mem.add(irrelevant)

        hits = mem.retrieve((1.0, 0.0), current_step=10, top_k=1)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0][1].text, "party prep")
        self.assertEqual(relevant.last_accessed_step, 10)

    def test_rag_chunk_and_retrieve_finds_ha3_section(self) -> None:
        chunks = chunk_by_paragraphs(SAMPLE_DOC, max_chars=260)
        self.assertGreaterEqual(len(chunks), 2)
        hits = retrieve("Havenask HA3 multi-path recall HNSW", chunks, top_k=1)
        self.assertIn("Havenask", hits[0][1].text)

    def test_react_execute_tool_call_computes_lora_params(self) -> None:
        out = execute_tool_call(
            {
                "name": "lora_param_count",
                "args": {"d_model": 4096, "rank": 8, "num_layers": 32, "num_projections": 2},
            }
        )
        self.assertEqual(out, str(2 * 4096 * 8 * 2 * 32))

    def test_majority_vote_picks_most_frequent(self) -> None:
        self.assertEqual(majority_vote(["Dense", "Sparse MoE", "Dense "]), "Dense")


if __name__ == "__main__":
    unittest.main()

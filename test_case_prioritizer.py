# test_case_prioritizer.py
"""
Automated Test‑Case Grouping & Prioritisation (LangChain, agentic)
=================================================================

**2025‑05‑04 hot‑fix:** Updated imports to match LangChain 0.1+ package split.
Requires:
```
pip install -U langchain langchain-community langchain-huggingface
```

This prototype still:
* uses **ONLY free LLMs** (Zephyr‑7B‑β on HuggingFace Hub or a local Ollama model),
* provides **Agent 1** (*ModuleGrouperAgent*) and **Agent 2** (*SmokeFirstSorterAgent*),
* exposes a simple CLI: `python test_case_prioritizer.py <in>.json out.json`.

The only functional change is the `get_llm()` helper which now constructs
`HuggingFaceHub` (community) → wraps it with `ChatHuggingFace` (huggingface).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
#  Updated LangChain imports after package split                            |
# ---------------------------------------------------------------------------
# Free / OSS integrations now live in *langchain‑community*.
# Chat wrappers for HF models live in *langchain‑huggingface*.
# ---------------------------------------------------------------------------
from langchain_community.llms import HuggingFaceHub
from langchain_community.chat_models import ChatOllama
from langchain_huggingface.chat_models import ChatHuggingFace

from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseOutputParser

# ---------------------------------------------------------------------------
#                         LLM backend helper
# ---------------------------------------------------------------------------

def get_llm(backend: str = "ollama"):
    """Return a **chat model** usable by LangChain.

    * ``backend='hf'``     – Zephyr‑7B‑β via HuggingFaceHub (needs free token)
    * ``backend='ollama'`` – any local Ollama chat model (defaults llama2)
    """

    if backend == "hf":
        # 1) Build a *text‑gen LLM* (community package)
        hf_llm = HuggingFaceHub(
            repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            model_kwargs={"temperature": 0.2, "max_new_tokens": 512},
        )
        # 2) Wrap as *chat* model (huggingface package)
        return ChatHuggingFace(llm=hf_llm, verbose=False)

    if backend == "ollama":
        return ChatOllama(model="mistral", temperature=0.2)

    raise ValueError("backend must be 'hf' or 'ollama'")


# ---------------------------------------------------------------------------
#                         Output parser helpers
# ---------------------------------------------------------------------------

class LineParser(BaseOutputParser[str]):
    """Return the first non‑empty line from model output."""

    def parse(self, text: str) -> str:  # noqa: D401
        for line in text.strip().splitlines():
            if line.strip():
                return line.strip()
        raise ValueError("No non‑empty lines in LLM output")


class JsonListParser(BaseOutputParser[List[str]]):
    """Find the first JSON array in the model output and return it."""

    def parse(self, text: str) -> List[str]:
        import re, json

        # look for the first […] block, even if the model wrapped it in prose
        match = re.search(r"\[[\s\S]*?\]", text)
        if not match:
            raise ValueError("No JSON array found in model output")
        return json.loads(match.group(0))



# ---------------------------------------------------------------------------
#                         Agent 1 – grouping by module
# ---------------------------------------------------------------------------

class ModuleGrouperAgent:
    """Assign a *module* label to each test case using an LLMChain."""

    def __init__(self, llm):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert Python tester. Given a test case, output the *module* it primarily belongs to."
                    " Choose a short dot‑path like `auth.login` or `api.cart`. If unsure, use the stem of file_path.",
                ),
                (
                    "human",
                    "Test file path: {file_path}\n"
                    "Test name: {test_name}\n"
                    "Summary: {summary}\n"
                    "Which module is under test? Output only the module name on a single line.",
                ),
            ]
        )
        self.chain = LLMChain(
            llm=llm,
            prompt=prompt,
            output_parser=LineParser(),
            verbose=False,
        )

    def run(self, tc: Dict) -> str:
        return self.chain.run(tc)


# ---------------------------------------------------------------------------
#                         Agent 2 – smoke‑first sorter
# ---------------------------------------------------------------------------

class SmokeFirstSorterAgent:
    """Return test names ordered so smoke tests appear first."""

    def __init__(self, llm):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a senior QA engineer. The user will give you a JSON list of test cases for *one* module."
                    " Return a JSON list of the **test_name** fields, ordered so that the most fundamental *smoke* tests"
                    " (quick, critical‑path sanity checks) appear at the top, followed by broader or edge‑case tests."
                    " Only output the JSON list and nothing but the JSON list.",
                ),
                ("human", "Here are the test cases:\n{cases}\n"),
            ]
        )
        self.chain = LLMChain(
            llm=llm,
            prompt=prompt,
            output_parser=JsonListParser(),
            verbose=False,
        )

    def run(self, cases: List[Dict]) -> List[str]:
        payload = json.dumps(cases, indent=2)
        return self.chain.run({"cases": payload})


# ---------------------------------------------------------------------------
#                         Orchestration util
# ---------------------------------------------------------------------------

def group_and_prioritise(test_cases: List[Dict], backend: str = "hf") -> Dict[str, List[Dict]]:
    """Full pipeline → {module: [ordered test‑case dicts]}"""

    llm = get_llm(backend)

    grouper = ModuleGrouperAgent(llm)
    sorter = SmokeFirstSorterAgent(llm)

    # 1) Group by module ------------------------------------------------------
    buckets: Dict[str, List[Dict]] = defaultdict(list)
    for tc in test_cases:
        module = grouper.run(tc)
        tc["_module"] = module
        buckets[module].append(tc)

    # 2) Sort each bucket -----------------------------------------------------
    final: Dict[str, List[Dict]] = {}
    for module, cases in buckets.items():
        ordering = sorter.run(cases)
        lut = {c["test_name"]: c for c in cases}
        ordered = [lut[name] for name in ordering if name in lut]
        leftovers = [c for c in cases if c["test_name"] not in ordering]
        final[module] = ordered + leftovers
    return final


# ---------------------------------------------------------------------------
#                         CLI entry‑point
# ---------------------------------------------------------------------------

def _cli():  # noqa: D401
    parser = argparse.ArgumentParser(description="Group & prioritise NL test cases with free LLMs.")
    parser.add_argument("input", type=Path, help="Path to JSON/.txt file containing the test cases list")
    parser.add_argument("output", type=Path, help="Where to write the grouped+ordered JSON")
    parser.add_argument("--backend", choices=["hf", "ollama"], default="hf", help="LLM backend to use (default: hf)")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as fh:
        test_cases = json.load(fh)

    result = group_and_prioritise(test_cases, backend=args.backend)

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    print(f"✅ Done! Grouped & prioritised test cases written to {args.output}")


if __name__ == "__main__":
    _cli()

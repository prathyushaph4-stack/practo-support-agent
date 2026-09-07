"""
mock_autogen_client.py
Practo Capstone - Task 14: Mock ChatCompletionClient for Autogen (MOCK_LLM)
-----------------------------------------------------------------------------
Deterministic, rule-based fake model client so the Autogen review team runs
fully offline with zero API keys, mirroring MockLLM's role for CrewAI.
"""

import json
import re
from typing import Any, Mapping, Optional, Sequence

from autogen_core.models import (
    ChatCompletionClient,
    CreateResult,
    LLMMessage,
    ModelInfo,
    RequestUsage,
    SystemMessage,
    UserMessage,
    AssistantMessage,
)

UNGROUNDED_MARKERS = ["guaranteed", "100% cure", "always works", "never fails", "definitely will"]


class MockAutogenClient(ChatCompletionClient):
    def __init__(self):
        self._model_info = ModelInfo(
            vision=False, function_calling=False, json_output=True,
            family="mock", structured_output=True,
        )
        self._total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

    @property
    def model_info(self) -> ModelInfo:
        return self._model_info

    @property
    def capabilities(self):
        return self._model_info

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools=(),
        json_output: Optional[bool] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token=None,
    ) -> CreateResult:
        with open("autogen_debug.log", "a") as f:
            f.write("\n---- AUTOGEN CALL ----\n")
            for m in messages:
                f.write(f"[{type(m).__name__}] {getattr(m, 'content', '')}\n")
            f.write(f"json_output={json_output}\n")
            f.write("---- END ----\n")

        system_text = ""
        body_texts = []
        for m in messages:
            if isinstance(m, SystemMessage):
                system_text += str(m.content) + "\n"
            else:
                body_texts.append(str(getattr(m, "content", "")))
        full_text = system_text + "\n".join(body_texts)

        is_final_editor = "Final-Editor" in system_text or "Final_Editor" in system_text
        is_reviewer = "Policy-Compliance-Reviewer" in system_text or "Reviewer" in system_text

        draft_match = re.search(r"Draft answer:\s*(.+?)(?:\n\n|\Z)", full_text, re.DOTALL)
        draft_text = draft_match.group(1).strip() if draft_match else ""
        has_issue = any(marker in draft_text.lower() for marker in UNGROUNDED_MARKERS)

        usage = RequestUsage(prompt_tokens=len(full_text.split()), completion_tokens=20)
        self._total_usage = RequestUsage(
            prompt_tokens=self._total_usage.prompt_tokens + usage.prompt_tokens,
            completion_tokens=self._total_usage.completion_tokens + usage.completion_tokens,
        )

        if is_final_editor:
            if has_issue:
                # Remove the WHOLE sentence containing an unsupported claim,
                # rather than just the trigger word (avoids grammatically
                # broken leftovers like "This treatment is to work...").
                sentences = re.split(r'(?<=[.!?])\s+', draft_text)
                cleaned_sentences = [
                    s for s in sentences
                    if not any(marker in s.lower() for marker in UNGROUNDED_MARKERS)
                ]
                cleaned = " ".join(cleaned_sentences).strip()
                verdict = {
                    "approved": False,
                    "final_answer": cleaned,
                    "reason": "Removed an unsupported absolute claim not present in the retrieved context.",
                }
            else:
                verdict = {
                    "approved": True,
                    "final_answer": draft_text,
                    "reason": "Draft is fully grounded in the retrieved context; approved unchanged.",
                }
            return CreateResult(finish_reason="stop", content=json.dumps(verdict), usage=usage, cached=False)

        if is_reviewer:
            content = (
                "REVIEW: Unsupported absolute claim detected; recommend revision."
                if has_issue else
                "REVIEW: Draft is fully supported by the retrieved context. No issues found."
            )
            return CreateResult(finish_reason="stop", content=content, usage=usage, cached=False)

        return CreateResult(finish_reason="stop", content="OK", usage=usage, cached=False)

    async def create_stream(self, *args, **kwargs):
        raise NotImplementedError("Streaming not needed under MOCK_LLM")

    def actual_usage(self) -> RequestUsage:
        return self._total_usage

    def total_usage(self) -> RequestUsage:
        return self._total_usage

    def count_tokens(self, messages, *, tools=()) -> int:
        return sum(len(str(getattr(m, "content", ""))) for m in messages) // 4

    def remaining_tokens(self, messages, *, tools=()) -> int:
        return 100000

    async def close(self) -> None:
        pass
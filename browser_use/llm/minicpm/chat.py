# -*- coding: utf-8 -*-
"""
MiniCPM integration for browser_use using ModelManager.
This version is identical to the successful "manual-load" version EXCEPT:
- model/tokenizer/processor are retrieved from ModelManager (shared model)
Everything else (JSON prompt, debug, validation, parser) is kept exactly aligned
with your stable successful implementation.
"""

import json
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any, Optional, TypeVar, overload

import numpy as np
import torch
from pydantic import BaseModel

from transformers import set_seed as hf_set_seed

from browser_use.llm.base import BaseChatModel
from browser_use.llm.exceptions import ModelProviderError
from browser_use.llm.messages import BaseMessage
from browser_use.llm.minicpm.serializer import MiniCPMMessageSerializer
from browser_use.llm.schema import SchemaOptimizer
from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage

T = TypeVar('T', bound=BaseModel)


# ------------------------------------------------------------------------------
# SEED
# ------------------------------------------------------------------------------
def set_seed(seed: int = 17) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    hf_set_seed(seed)


# ------------------------------------------------------------------------------
# ChatMiniCPM with ModelManager
# ------------------------------------------------------------------------------
@dataclass
class ChatMiniCPM(BaseChatModel):
    """
    MiniCPM-o-2_6 integration optimized for strict JSON action formatting.
    Identical to your successful version except model loading (ModelManager).
    """

    # Model config (unused except for compatibility)
    model_path: str = "/uufs/chpc.utah.edu/common/home/u1533682/model/MiniCPM-o-2_6"

    # Params
    temperature: float = 0.0
    max_completion_tokens: int = 4096
    seed: int = 17
    device: str = "cuda"

    # Internal
    _model_instance: Any = field(default=None, init=False, repr=False)
    tokenizer: Any = field(default=None, init=False, repr=False)
    processor: Any = field(default=None, init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)

    # ----------------------------------------------------------------------
    @property
    def provider(self) -> str:
        return "minicpm"

    @property
    def model(self) -> str:
        return "MiniCPM-o-2_6"

    @property
    def name(self) -> str:
        return f"MiniCPM-o-2_6@{self.device}"

    # ----------------------------------------------------------------------
    def __post_init__(self):
        if not self._initialized:
            self._initialize_model()
            self._initialized = True

    # ----------------------------------------------------------------------
    def _initialize_model(self) -> None:
        """
        Use ModelManager to retrieve shared model/tokenizer/processor.
        All other logic stays identical to successful version.
        """
        set_seed(self.seed)

        try:
            from model_manager import ModelManager
            manager = ModelManager.get()

            if manager.model_type != "minicpm":
                raise ModelProviderError(
                    message=f"ModelManager loaded '{manager.model_type}', "
                            f"but ChatMiniCPM requires 'minicpm'.",
                    model=self.name
                )

            # Retrieve shared components
            self._model_instance = manager.model
            self.tokenizer = manager.tokenizer
            self.processor = manager.processor

            print(f"[MiniCPM] Using shared model from ModelManager")
            print(f"[MiniCPM] Model path: {manager.model_path}")

            if self.device == "cuda":
                if torch.cuda.is_available():
                    print(f"[MiniCPM] CUDA available: {torch.cuda.get_device_name(0)}")
                else:
                    raise RuntimeError("CUDA not available, but device='cuda'.")

        except Exception as e:
            raise ModelProviderError(
                message=f"Failed to initialize MiniCPM via ModelManager: {str(e)}",
                model=self.name
            )

    # ------------------------------------------------------------------------------
    # Token usage estimate (same as successful version)
    # ------------------------------------------------------------------------------
    def _estimate_tokens(self, text: str) -> int:
        return max(len(text) // 4, 1)

    def _get_usage(self, messages: list[BaseMessage], completion: str) -> ChatInvokeUsage:
        prompt_texts = []
        for msg in messages:
            if isinstance(msg.content, str):
                prompt_texts.append(msg.content)
            elif msg.content:
                for part in msg.content:
                    if hasattr(part, "type") and part.type == "text":
                        prompt_texts.append(part.text)
                    elif isinstance(part, dict) and part.get("type") == "text":
                        prompt_texts.append(part.get("text", ""))

        prompt_text = " ".join(prompt_texts)

        prompt_tokens = self._estimate_tokens(prompt_text)
        completion_tokens = self._estimate_tokens(completion)

        return ChatInvokeUsage(
            prompt_tokens=prompt_tokens,
            prompt_cached_tokens=None,
            prompt_cache_creation_tokens=None,
            prompt_image_tokens=None,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    # ------------------------------------------------------------------------------
    # ENHANCED JSON INSTRUCTION (copied EXACTLY from successful version)
    # ------------------------------------------------------------------------------
    def _build_enhanced_json_instruction(self, schema: dict) -> str:
        schema_str = json.dumps(schema, indent=2)

        return f"""
CRITICAL JSON OUTPUT REQUIREMENTS
============================================================

YOU MUST output ONLY a valid JSON object.  
NO markdown formatting, NO commentary, NO text outside JSON.

Valid JSON Schema:
{schema_str}

============================================================
ACTION RULES (EXTREMELY IMPORTANT)
============================================================

The "action" field MUST be a non-empty list of action objects.  
Each action object MUST contain EXACTLY ONE valid action type.

After ANY navigate or click action, you MUST immediately add a wait action.

Example:
[
  {{ "click": {{ "index": 261 }} }},
  {{ "wait": {{}} }}
]

Allowed action types:
- click:     {{"index": number}}
- input:     {{"index": number, "text": string}}
- scroll:    {{"down": boolean, "pages": number}}
- navigate:  {{"url": string}}
- go_back:   {{}}
- wait:      {{}}
- screenshot: {{}}
- extract:   {{"data": string}}
- done:      {{"success": boolean, "text": string}}

============================================================
INDEX RULES
============================================================

You MUST use an index ONLY if it appears in <browser_state>.  
NEVER create or guess indexes.

============================================================
FULL RESPONSE FORMAT
============================================================

{{
  "thinking": "...",
  "evaluation_previous_goal": "...",
  "memory": "...",
  "next_goal": "...",
  "action": [
    {{ "click": {{ "index": VALID INDEX }} }}
  ]
}}

Now respond with ONLY the JSON object:
"""

    # ------------------------------------------------------------------------------
    # MAIN INVOKE (identical to successful version except model loading)
    # ------------------------------------------------------------------------------
    @overload
    async def ainvoke(self, messages: list[BaseMessage], output_format: None = None) -> ChatInvokeCompletion[str]: ...

    @overload
    async def ainvoke(self, messages: list[BaseMessage], output_format: type[T]) -> ChatInvokeCompletion[T]: ...

    async def ainvoke(self, messages: list[BaseMessage], output_format=None):
        if not messages:
            raise ModelProviderError("messages cannot be empty", model=self.name)

        try:
            minicpm_msgs, first_image, system_message = MiniCPMMessageSerializer.serialize_messages(messages)

            if first_image is None:
                raise ModelProviderError(
                    message="MiniCPM-o-2_6 requires an image in the first user message.",
                    model=self.name
                )

            # --------------------------
            # Add JSON instruction
            # --------------------------
            if output_format is not None:
                schema = SchemaOptimizer.create_optimized_json_schema(output_format)
                instruction = self._build_enhanced_json_instruction(schema)

                last_msg = minicpm_msgs[-1]
                if isinstance(last_msg["content"], list):
                    last_msg["content"][1] += instruction
                else:
                    last_msg["content"] += instruction

            # Debug
            print(f"[MiniCPM] Processing {len(minicpm_msgs)} messages")
            print(f"[MiniCPM] Image size: {first_image.size}")

            # --------------------------
            # Generate
            # --------------------------
            with torch.no_grad():
                answer = self._model_instance.chat(
                    image=first_image,
                    msgs=minicpm_msgs,
                    tokenizer=self.tokenizer,
                    temperature=self.temperature,
                    max_new_tokens=self.max_completion_tokens,
                    do_sample=False if self.temperature == 0 else True,
                    use_cache=False,
                )

            print(f"[MiniCPM] Response: {len(answer)} chars")
            usage = self._get_usage(messages, answer)

            # --------------------------
            # Plain text
            # --------------------------
            if output_format is None:
                return ChatInvokeCompletion(completion=answer, usage=usage, stop_reason="stop")

            # --------------------------
            # Parse structured JSON
            # --------------------------
            json_str = answer.strip()

            if "```json" in json_str:
                json_str = re.sub(r"```json\s*(.*?)\s*```", r"\1", json_str, flags=re.DOTALL)
            elif "```" in json_str:
                json_str = re.sub(r"```\s*(.*?)\s*```", r"\1", json_str, flags=re.DOTALL)

            start = json_str.find("{")
            end = json_str.rfind("}")
            if start != -1 and end != -1:
                json_str = json_str[start:end + 1]

            print(f"[MiniCPM] Cleaned JSON: {json_str[:200]}...")

            parsed_json = json.loads(json_str)

            # validate actions
            if "action" in parsed_json:
                actions = parsed_json["action"]
                if not isinstance(actions, list) or len(actions) == 0:
                    raise ValueError("'action' must be a non-empty list")

                for i, action in enumerate(actions):
                    if not isinstance(action, dict):
                        raise ValueError(f"action[{i}] must be dict")
                    if len(action) != 1:
                        raise ValueError(f"action[{i}] must have exactly ONE key")

            parsed = output_format.model_validate(parsed_json)

            return ChatInvokeCompletion(completion=parsed, usage=usage, stop_reason="stop")

        except Exception as e:
            raise ModelProviderError(message=f"Error during invocation: {str(e)}", model=self.name)

    # ------------------------------------------------------------------------------
    # Sync wrapper
    # ------------------------------------------------------------------------------
    def invoke(self, messages, output_format=None):
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.ainvoke(messages, output_format))

    # ------------------------------------------------------------------------------
    def __del__(self):
        """No CUDA cache clear because ModelManager owns the model."""
        pass


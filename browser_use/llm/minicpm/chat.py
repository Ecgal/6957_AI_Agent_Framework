# -*- coding: utf-8 -*-
"""
Optimized MiniCPM integration for browser_use.
Addresses action format issues through enhanced JSON instructions.

Key improvements:
1. Detailed JSON schema explanations
2. Action format examples (correct vs incorrect)
3. Additional validation logic
4. Better error handling
5. ModelManager integration to avoid duplicate model loading
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
from transformers import AutoModel, AutoTokenizer
from transformers import set_seed as hf_set_seed

from browser_use.llm.base import BaseChatModel
from browser_use.llm.exceptions import ModelProviderError
from browser_use.llm.messages import BaseMessage
from browser_use.llm.minicpm.serializer import MiniCPMMessageSerializer
from browser_use.llm.schema import SchemaOptimizer
from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage

T = TypeVar('T', bound=BaseModel)


def set_seed(seed: int = 17) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    hf_set_seed(seed)


@dataclass
class ChatMiniCPM(BaseChatModel):
    """
    Optimized MiniCPM-o-2_6 integration focusing on action format correctness.

    This implementation adds extensive JSON formatting instructions to compensate
    for MiniCPM's lack of native JSON schema constraint (unlike Ollama which has
    built-in 'format' parameter support).

    Now uses ModelManager to retrieve pre-loaded model instead of loading separately.
    """

    # Model configuration (kept for compatibility, but actual path comes from ModelManager)
    model_path: str = "/uufs/chpc.utah.edu/common/home/u1533682/SeeAct/SeeAct/seeact_package/seeact/demo_utils/MiniCPM-o-2_6"

    # Model parameters
    temperature: float = 0.0
    max_completion_tokens: int = 4096
    seed: int = 17
    device: str = "cuda"
    torch_dtype: torch.dtype = torch.bfloat16

    # Model instance (internal use only)
    _model_instance: Any = field(default=None, init=False, repr=False)
    tokenizer: Any = field(default=None, init=False, repr=False)
    processor: Any = field(default=None, init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)

    @property
    def provider(self) -> str:
        return 'minicpm'

    @property
    def model(self) -> str:
        return "MiniCPM-o-2_6"

    @property
    def name(self) -> str:
        return f"MiniCPM-o-2_6@{self.device}"

    def __post_init__(self):
        """Initialize model after dataclass creation."""
        if not self._initialized:
            self._initialize_model()
            self._initialized = True

    def _initialize_model(self) -> None:
        """
        Initialize the MiniCPM model by retrieving it from ModelManager.

        This avoids duplicate model loading when multiple ChatMiniCPM instances are created.
        Falls back to direct loading if ModelManager is not available.

        Raises:
            ModelProviderError: If model initialization fails
        """
        set_seed(self.seed)

        try:
            # Try to get model from ModelManager first
            from model_manager import ModelManager

            manager = ModelManager.get()

            # Verify that MiniCPM model is loaded in ModelManager
            if manager.model_type != "minicpm":
                raise ModelProviderError(
                    message=f"ModelManager is configured for '{manager.model_type}', "
                            "but ChatMiniCPM requires 'minicpm' model type. "
                            "Please initialize ModelManager with model_type='minicpm'.",
                    model=self.name,
                )

            # Retrieve shared model, tokenizer, and processor
            self._model_instance = manager.model
            self.tokenizer = manager.tokenizer
            self.processor = manager.processor

            print(f"[MiniCPM] ✅ Retrieved model from ModelManager (shared instance)")
            print(f"[MiniCPM] Model path: {manager.model_path}")

            if self.device == "cuda" and torch.cuda.is_available():
                print(f"[MiniCPM] Using CUDA: {torch.cuda.get_device_name(0)}")

            print(f"[MiniCPM] Model initialization complete")

        except ImportError:
            # Fallback: Load model directly if ModelManager is not available
            print("[MiniCPM] ⚠️ Warning: ModelManager not found, loading model directly")
            print("[MiniCPM] This may cause memory issues with multiple instances")
            self._load_model_directly()

        except Exception as e:
            # If ModelManager fails, try fallback
            print(f"[MiniCPM] ⚠️ Warning: Failed to use ModelManager: {e}")
            print("[MiniCPM] Falling back to direct model loading")
            self._load_model_directly()

    def _load_model_directly(self) -> None:
        """
        Fallback method to load model directly (for backward compatibility).
        This should not be used when ModelManager is available.

        Raises:
            ModelProviderError: If model initialization fails
        """
        model_path = self.model_path
        if not os.path.isabs(model_path):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, model_path)

        if not os.path.exists(model_path):
            raise ModelProviderError(
                message=f"Model path does not exist: {model_path}",
                model=self.name,
            )

        print(f"[MiniCPM] Initializing model from {model_path}")

        try:
            self._model_instance = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True,
                attn_implementation='sdpa',
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True
            )

            if self.device == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is not available. Set device='cpu' or install CUDA.")
                self._model_instance = self._model_instance.eval().cuda()
                print(f"[MiniCPM] Model loaded on CUDA: {torch.cuda.get_device_name(0)}")
            else:
                self._model_instance = self._model_instance.eval()
                print(f"[MiniCPM] Model loaded on CPU")

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True
            )
            print(f"[MiniCPM] Tokenizer loaded")

            self.processor = getattr(self._model_instance, "processor", None)
            if self.processor is None:
                try:
                    from transformers import AutoProcessor
                    self.processor = AutoProcessor.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        local_files_only=True
                    )
                    print(f"[MiniCPM] Processor loaded")
                except Exception:
                    self.processor = None
                    print(f"[MiniCPM] Processor not available")

            print(f"[MiniCPM] Model initialization complete")

        except Exception as e:
            raise ModelProviderError(
                message=f"Failed to initialize MiniCPM model: {str(e)}",
                model=self.name,
            ) from e

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for usage tracking.
        Uses rough approximation of 4 characters per token.
        """
        return max(len(text) // 4, 1)

    def _get_usage(self, messages: list[BaseMessage], completion: str) -> ChatInvokeUsage:
        """
        Calculate usage statistics for the API call.

        Args:
            messages: Input messages
            completion: Model's completion text

        Returns:
            ChatInvokeUsage with estimated token counts
        """
        prompt_texts = []
        for msg in messages:
            if isinstance(msg.content, str):
                prompt_texts.append(msg.content)
            elif msg.content is not None:
                for part in msg.content:
                    if hasattr(part, 'type') and part.type == 'text':
                        prompt_texts.append(part.text)
                    elif isinstance(part, dict) and part.get('type') == 'text':
                        prompt_texts.append(part.get('text', ''))

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

    def _build_enhanced_json_instruction(self, optimized_schema: dict) -> str:
        """
        Build enhanced JSON format instructions with detailed action examples.

        Args:
            optimized_schema: Optimized JSON schema from SchemaOptimizer

        Returns:
            Comprehensive JSON formatting instruction string
        """
        schema_str = json.dumps(optimized_schema, indent=2)

        instruction = f"""

========================================
CRITICAL: JSON OUTPUT REQUIREMENTS
========================================

You MUST respond with ONLY valid JSON that EXACTLY matches this schema:

{schema_str}

CRITICAL ACTION FORMAT RULES:
1. The "action" field MUST be a list of dictionaries
2. Each dictionary MUST contain EXACTLY ONE key-value pair
3. The key is the action type (e.g., "click_element", "input_text")
4. The value is the parameters object for that action

CORRECT ACTION FORMAT EXAMPLES:
✓ {{"action": [{{"click_element": {{"index": 5}}}}]}}
✓ {{"action": [{{"input_text": {{"index": 3, "text": "hello"}}}}]}}
✓ {{"action": [{{"go_to_url": {{"url": "https://example.com"}}}}]}}
✓ {{"action": [{{"scroll": {{"direction": "down"}}}}]}}

INCORRECT ACTION FORMAT (NEVER DO THIS):
✗ {{"action": [{{"action_name": "click_element", "index": 5}}]}}
✗ {{"action": [{{"type": "click_element", "params": {{"index": 5}}}}]}}
✗ {{"action": [{{"click_element": 5}}]}}
✗ {{"action": "click_element"}}

ACTION TYPE REFERENCE:
- click_element: Click on an element by index
  Format: {{"click_element": {{"index": <number>}}}}

- input_text: Type text into an input field
  Format: {{"input_text": {{"index": <number>, "text": "<string>"}}}}

- go_to_url: Navigate to a URL
  Format: {{"go_to_url": {{"url": "<string>"}}}}

- scroll: Scroll the page
  Format: {{"scroll": {{"direction": "up"|"down"}}}}

- go_back: Go back in browser history
  Format: {{"go_back": {{}}}}

- done: Complete the task
  Format: {{"done": {{}}}}

RESPONSE FORMAT:
- Output ONLY the JSON object
- NO markdown code blocks (```json)
- NO explanatory text before or after the JSON
- NO comments inside the JSON
- Ensure all strings are properly escaped
- Ensure all required fields are present
- Follow the exact structure shown in the schema

START YOUR RESPONSE WITH {{ and END WITH }}
"""
        return instruction

    @overload
    async def ainvoke(
            self,
            messages: list[BaseMessage],
            output_format: None = None,
    ) -> ChatInvokeCompletion[str]:
        ...

    @overload
    async def ainvoke(
            self,
            messages: list[BaseMessage],
            output_format: type[T],
    ) -> ChatInvokeCompletion[T]:
        ...

    async def ainvoke(
            self,
            messages: list[BaseMessage],
            output_format: type[T] | None = None,
    ) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
        """
        Improved invoke method with better error handling and format guidance.

        This method adds extensive JSON formatting instructions when structured output
        is requested, compensating for MiniCPM's lack of native schema constraints.

        Args:
            messages: List of chat messages (images in message content)
            output_format: Optional Pydantic model for structured output

        Returns:
            ChatInvokeCompletion with string or structured response

        Raises:
            ModelProviderError: If invocation or parsing fails
        """
        if not messages:
            raise ModelProviderError(
                message="messages cannot be empty",
                model=self.name,
            )

        try:
            # Use serializer to convert messages
            minicpm_messages, first_image, system_message = MiniCPMMessageSerializer.serialize_messages(messages)

            # Validate image presence
            if first_image is None:
                raise ModelProviderError(
                    message="MiniCPM-o-2_6 requires an image in the first user message.",
                    model=self.name,
                )

            # If structured output is requested, add enhanced instructions
            if output_format is not None:
                optimized_schema = SchemaOptimizer.create_optimized_json_schema(output_format)
                json_instruction = self._build_enhanced_json_instruction(optimized_schema)

                # Append instruction to last message
                if minicpm_messages:
                    last_msg = minicpm_messages[-1]
                    if isinstance(last_msg['content'], list):
                        # Message with image: [image, text]
                        last_msg['content'][1] += json_instruction
                    else:
                        # Text-only message
                        last_msg['content'] += json_instruction

            # Debug output
            print(f"[MiniCPM] Processing {len(minicpm_messages)} messages")
            if system_message:
                print(f"[MiniCPM] System message length: {len(system_message)} chars")
            print(f"[MiniCPM] Image size: {first_image.size}")

            if minicpm_messages:
                first_msg = minicpm_messages[0]
                if isinstance(first_msg['content'], list):
                    text_content = first_msg['content'][1]
                    print(f"[MiniCPM] First message text length: {len(text_content)} chars")
                    print(f"[MiniCPM] First message preview: {text_content[:200]}...")
                else:
                    print(f"[MiniCPM] First message: {first_msg['content'][:200]}...")

            # Generate response
            print(f"[MiniCPM] Generating (temp={self.temperature}, max_tokens={self.max_completion_tokens})")

            with torch.no_grad():
                answer = self._model_instance.chat(
                    image=first_image,
                    msgs=minicpm_messages,
                    tokenizer=self.tokenizer,
                    temperature=self.temperature,
                    max_new_tokens=self.max_completion_tokens,
                    do_sample=False if self.temperature == 0 else True,
                    use_cache=False
                )

            print(f"[MiniCPM] Response length: {len(answer)} chars")
            print(f"[MiniCPM] Raw response preview: {answer[:500]}...")

            usage = self._get_usage(messages, answer)

            if output_format is None:
                # Return plain text response
                return ChatInvokeCompletion(
                    completion=answer,
                    usage=usage,
                    stop_reason='stop',
                )
            else:
                # Parse structured output
                try:
                    json_str = answer.strip()

                    # Remove markdown code blocks using regex
                    if '```json' in json_str:
                        match = re.search(r'```json\s*(.*?)\s*```', json_str, re.DOTALL)
                        if match:
                            json_str = match.group(1).strip()
                    elif '```' in json_str:
                        match = re.search(r'```\s*(.*?)\s*```', json_str, re.DOTALL)
                        if match:
                            json_str = match.group(1).strip()

                    # Extract JSON object by finding first { and last }
                    start_idx = json_str.find('{')
                    end_idx = json_str.rfind('}')
                    if start_idx != -1 and end_idx != -1:
                        json_str = json_str[start_idx:end_idx + 1]

                    print(f"[MiniCPM] Cleaned JSON preview: {json_str[:300]}...")

                    # Parse and validate JSON
                    parsed_data = json.loads(json_str)

                    # Additional validation for action field
                    if 'action' in parsed_data:
                        actions = parsed_data['action']
                        if not isinstance(actions, list):
                            raise ValueError("'action' must be a list")
                        if len(actions) == 0:
                            raise ValueError("'action' list cannot be empty")

                        # Check each action is valid
                        for i, action in enumerate(actions):
                            if not isinstance(action, dict):
                                raise ValueError(f"Action {i} must be a dict")
                            if len(action) != 1:
                                raise ValueError(
                                    f"Action {i} must have exactly one action type, "
                                    f"got: {list(action.keys())}"
                                )

                    # Validate with Pydantic model
                    parsed = output_format.model_validate(parsed_data)

                    return ChatInvokeCompletion(
                        completion=parsed,
                        usage=usage,
                        stop_reason='stop',
                    )

                except json.JSONDecodeError as e:
                    print(f"[MiniCPM] JSON parse error: {str(e)}")
                    print(f"[MiniCPM] Failed JSON string: {json_str[:1000]}")
                    raise ModelProviderError(
                        message=f'Failed to parse JSON: {str(e)}\nOutput: {answer[:1000]}',
                        model=self.name,
                    ) from e
                except ValueError as e:
                    print(f"[MiniCPM] Validation error: {str(e)}")
                except Exception as e:
                    print(f"[MiniCPM] Unexpected error: {str(e)}")
                    raise ModelProviderError(
                        message=f'Failed to parse structured output: {str(e)}',
                        model=self.name,
                    ) from e

        except ModelProviderError:
            raise
        except Exception as e:
            raise ModelProviderError(
                message=f"Error during invocation: {str(e)}",
                model=self.name,
            ) from e

    def invoke(
            self,
            messages: list[BaseMessage],
            output_format: type[T] | None = None,
    ) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
        """
        Synchronous version of ainvoke.

        Creates or reuses event loop to run async method synchronously.
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.ainvoke(messages, output_format))

    def __del__(self):
        """
        Cleanup resources on deletion.
        Note: When using ModelManager, we don't clear CUDA cache as the model
        is shared across instances.
        """
        # Don't clear CUDA cache when using shared model from ModelManager
        pass
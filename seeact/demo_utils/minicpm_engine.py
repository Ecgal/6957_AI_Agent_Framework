# -*- coding: utf-8 -*-
# Copyright (c) 2024 OSU Natural Language Processing Group
#
# Licensed under the OpenRAIL-S License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.licenses.ai/ai-pubs-open-rails-vz1
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import random
import numpy as np
import os
from transformers import set_seed as hf_set_seed
import re


def set_seed(seed=17):
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


set_seed(17)


class MiniCPMEngine:
    """
    Engine for running MiniCPM-o-2_6 multimodal model.
    This class retrieves the model from ModelManager singleton to avoid duplicate loading.
    """

    def __init__(self, model_path=None, **kwargs):
        """
        Initialize the MiniCPM engine by retrieving the model from ModelManager.

        Args:
            model_path: Path to the pretrained model directory (unused, kept for compatibility)
            **kwargs: Additional arguments (unused, kept for compatibility)
        """
        # Import here to avoid circular dependency
        try:
            from model_manager import ModelManager

            # Get the singleton ModelManager instance
            manager = ModelManager.get()

            # Verify that MiniCPM model is loaded
            if manager.model_type != "minicpm":
                raise ValueError(
                    f"ModelManager is configured for '{manager.model_type}', "
                    "but MiniCPMEngine requires 'minicpm' model type"
                )

            # Reference the shared model, tokenizer, and processor
            self.model = manager.model
            self.tokenizer = manager.tokenizer
            self.processor = manager.processor
            self.model_path = manager.model_path

            print(f"✅ MiniCPMEngine initialized using shared model from ModelManager")

        except ImportError:
            # Fallback: load model directly (for backward compatibility)
            print("⚠️ Warning: ModelManager not found, loading model directly (not recommended)")
            self._load_model_directly(model_path)

    def _load_model_directly(self, model_path):
        """
        Fallback method to load model directly (for backward compatibility).
        This should not be used in the new architecture.
        """
        model_path = model_path or "MiniCPM-o-2_6"

        if not os.path.isabs(model_path):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, model_path)

        self.model_path = model_path

        print(f"Initializing MiniCPM-o-2_6 model from {model_path}")

        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation='sdpa',
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        self.model = self.model.eval().cuda()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )

        self.processor = getattr(self.model, "processor", None)
        if self.processor is None:
            try:
                from transformers import AutoProcessor
                self.processor = AutoProcessor.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    local_files_only=True
                )
            except Exception:
                self.processor = None

        print("Model loaded successfully on CUDA")

    def load_image(self, image_path):
        """
        Load and convert image to RGB format.

        Args:
            image_path: Path to the image file

        Returns:
            PIL Image object in RGB format
        """
        image = Image.open(image_path).convert('RGB')
        return image

    def generate(self, prompt=None, max_new_tokens=4096, temperature=0,
                 model=None, image_path=None, ouput_0=None, turn_number=0, **kwargs):
        """
        Generate response from the model for single or multi-turn conversation.

        Args:
            prompt: List of prompts [system_prompt, user_message_1, user_message_2]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature for generation
            model: Model name (unused, for compatibility)
            image_path: Path to the input image
            ouput_0: Output from the first turn (required when turn_number=1)
            turn_number: Conversation turn (0 for first turn, 1 for second turn)
            **kwargs: Additional generation arguments

        Returns:
            Generated text response from the model
        """
        if prompt is None or len(prompt) < 3:
            raise ValueError("Prompt must be a list of at least 3 elements: [system, user1, user2]")

        if image_path is None:
            raise ValueError("image_path must be provided for MiniCPM visual model")

        prompt0, prompt1, prompt2 = prompt

        # Load image
        image = self.load_image(image_path)

        # Build conversation messages based on turn number
        if turn_number == 0:
            # First turn: system + user message with image
            msgs = [
                {"role": "system", "content": prompt0},
                {"role": "user", "content": [image, prompt1]}
            ]
        elif turn_number == 1:
            # Second turn: system + first exchange + new user message
            if ouput_0 is None:
                raise ValueError("ouput_0 must be provided for turn_number=1")

            msgs = [
                {"role": "system", "content": prompt0},
                {"role": "user", "content": [image, prompt1]},
                {"role": "assistant", "content": ouput_0},
                {"role": "user", "content": prompt2}
            ]
        else:
            raise ValueError("turn_number must be 0 or 1 for two-turn conversation")

        # Generate response with specified parameters
        with torch.no_grad():
            answer = self.model.chat(
                image=image,
                msgs=msgs,
                tokenizer=self.tokenizer,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                do_sample=False if temperature == 0 else True,
                use_cache=False
            )

        return answer
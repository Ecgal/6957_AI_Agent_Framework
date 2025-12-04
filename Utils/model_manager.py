
"""
This module handles all model loading for the framework, both the main
reasoning models (MiniCPM or GPT-4o) and the auxiliary ranking model used
by SeeAct.

The important idea here is that the entire framework only ever has one
active model instance at a time. Since these models are large and
expensive to initialize (especially MiniCPM on GPU), we use a singleton
pattern so agents can all share the same loaded model without reloading
it or duplicating memory.

What this file is responsible for:
- Loading GPT (API-based) or MiniCPM (local GPU-based) models depending on
  what the framework is configured to use.
- Lazy loading the SeeAct ranking model on demand (only when SeeAct is used).
- Exposing a simple `.get()` method that lets other parts of the system
  retrieve the currently active model.
- Allowing the system to “switch models” by resetting the singleton.

"""



# model_manager.py
# Global singleton for model loading and sharing.
# Manages both main models (MiniCPM/GPT) and auxiliary models (ranking).

import os
import torch
import random
import numpy as np
from transformers import AutoModel, AutoTokenizer
from transformers import set_seed as hf_set_seed
from dotenv import load_dotenv


class ModelManager:
    """
    Singleton manager for model initialization.
    Manages both main model (minicpm/gpt) and auxiliary models (ranking).
    """

    _instance = None

    def __init__(self, model_type: str = "gpt", seed: int = 17):
        """
        Private constructor. Use ModelManager.get() to retrieve the shared instance.

        Args:
            model_type: "minicpm" or "gpt"
            seed: Random seed for reproducibility
        """
        self.model_type = model_type.lower()
        self.seed = seed

        # Main model attributes
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.api_key = None
        self.model_path = None

        # Ranking model attributes (lazy loaded)
        self.ranking_model = None

        if self.model_type == "minicpm":
            self._initialize_seed(seed)
            self._load_minicpm()
        elif self.model_type == "gpt":
            self._load_gpt()
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Use 'minicpm' or 'gpt'.")

    @classmethod
    def get(cls, model_type: str = "gpt", seed: int = 17):
        """
        Retrieve the global singleton instance.

        Args:
            model_type: "minicpm" or "gpt"
            seed: Random seed for reproducibility

        Note:
            This method only accepts parameters on first call (when creating singleton).
            Subsequent calls return the existing instance and ignore new parameters.
        """
        if cls._instance is None:
            cls._instance = cls(model_type, seed)
        return cls._instance

    @classmethod
    def switch_model(cls, model_type: str, seed: int = 17):
        """
        Switch to a different model by reinitializing the singleton.
        Interface for future Dashboard integration.

        Args:
            model_type: "minicpm" or "gpt"
            seed: Random seed for reproducibility
        """
        cls._instance = None
        return cls.get(model_type, seed)

    # --------------------------
    # Ranking model management
    # --------------------------
    def get_ranking_model(self, ranker_path: str):
        """
        Get or lazy-load the ranking model (used by SeeAct).

        Args:
            ranker_path: Path to the ranking model (required)

        Returns:
            Loaded ranking model instance
        """
        if self.ranking_model is not None:
            return self.ranking_model

        print(f"📚 Loading ranking model from {ranker_path}")

        try:
            # Import here to avoid issues if ranking_model module doesn't exist
            import sys

            # Add the SeeAct package path if needed
            seeact_path = os.path.join(os.path.dirname(__file__), "seeact_package")
            if os.path.exists(seeact_path) and seeact_path not in sys.path:
                sys.path.insert(0, seeact_path)

            from Agents.seeact.demo_utils.ranking_model import CrossEncoder

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.ranking_model = CrossEncoder(
                ranker_path,
                device=device,
                num_labels=1,
                max_length=512
            )
            print(f" Ranking model loaded successfully on {device}")

        except Exception as e:
            print(f" Failed to load ranking model: {e}")
            self.ranking_model = None

        return self.ranking_model

    # --------------------------
    # GPT initialization
    # --------------------------
    def _load_gpt(self):
        """
        Load OpenAI API key from environment.
        """
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        print(f" GPT model initialized with API key")

    # --------------------------
    # MiniCPM initialization
    # --------------------------
    def _initialize_seed(self, seed: int):
        """
        Initialize all necessary random seeds for reproducibility.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        hf_set_seed(seed)

        os.environ["PYTHONHASHSEED"] = str(seed)

        # cuDNN deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # PyTorch deterministic ops
        torch.use_deterministic_algorithms(True, warn_only=True)

    def _load_minicpm(self):
        """
        Load the MiniCPM model, tokenizer, and processor into memory.
        Model path is hardcoded here.
        """
        # Hardcoded path for MiniCPM model
        self.model_path = "/uufs/chpc.utah.edu/common/home/u1533682/model/MiniCPM-o-2_6"

        print(f" Loading MiniCPM model from {self.model_path}")

        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        self.model = self.model.eval().cuda()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True
        )

        try:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=True
            )
        except Exception:
            self.processor = None

        print(" MiniCPM model initialized successfully")
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

    def __init__(self, model_type: str = "gpt", model_path: str = "MiniCPM-o-2_6",
                 ranker_path: str = None, seed: int = 17):
        """
        Private constructor. Use ModelManager.get() to retrieve the shared instance.

        Args:
            model_type: "minicpm" or "gpt"
            model_path: Path to main model (only used if model_type == "minicpm")
            ranker_path: Path to ranking model (for SeeAct, lazy loaded)
            seed: Random seed for reproducibility
        """
        self.model_type = model_type.lower()
        self.seed = seed

        # Main model attributes
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.api_key = None

        # Ranking model attributes (lazy loaded)
        self.ranker_path = ranker_path
        self.ranking_model = None

        if self.model_type == "minicpm":
            self._initialize_seed(seed)
            self.model_path = self._resolve(model_path)
            self._load_minicpm()
        elif self.model_type == "gpt":
            self._load_gpt()
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Use 'minicpm' or 'gpt'.")

    @classmethod
    def get(cls, model_type: str = "gpt", model_path: str = "MiniCPM-o-2_6",
            ranker_path: str = None, seed: int = 17):
        """
        Retrieve the global singleton instance.
        """
        if cls._instance is None:
            cls._instance = cls(model_type, model_path, ranker_path, seed)
        return cls._instance

    @classmethod
    def switch_model(cls, model_type: str, model_path: str = "MiniCPM-o-2_6",
                     ranker_path: str = None, seed: int = 17):
        """
        Switch to a different model by reinitializing the singleton.
        Interface for future dashboard integration.
        """
        cls._instance = None
        return cls.get(model_type, model_path, ranker_path, seed)

    # --------------------------
    # Ranking model management
    # --------------------------
    def get_ranking_model(self, ranker_path: str = None):
        """
        Get or lazy-load the ranking model (used by SeeAct).

        Args:
            ranker_path: Path to the ranking model

        Returns:
            Loaded ranking model instance
        """
        if self.ranking_model is not None:
            return self.ranking_model

        # Use provided path or stored path
        path = ranker_path or self.ranker_path
        if path is None:
            raise ValueError("ranker_path must be provided for ranking model")

        # Update stored path if new one is provided
        if ranker_path:
            self.ranker_path = ranker_path

        print(f"📚 Loading ranking model from {path}")

        try:
            # Import here to avoid issues if ranking_model module doesn't exist
            import sys

            # Add the SeeAct package path if needed
            seeact_path = os.path.join(os.path.dirname(__file__), "seeact_package")
            if os.path.exists(seeact_path) and seeact_path not in sys.path:
                sys.path.insert(0, seeact_path)

            from seeact.demo_utils.ranking_model import CrossEncoder

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.ranking_model = CrossEncoder(
                path,
                device=device,
                num_labels=1,
                max_length=512
            )
            print(f"✅ Ranking model loaded successfully on {device}")

        except Exception as e:
            print(f"❌ Failed to load ranking model: {e}")
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
        print(f"✅ GPT model initialized with API key")

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

    def _resolve(self, path: str) -> str:
        """
        Resolve the model path if it is relative.
        """
        if os.path.isabs(path):
            return path
        base = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base, path)

    def _load_minicpm(self):
        """
        Load the MiniCPM model, tokenizer, and processor into memory.
        """
        print(f"📦 Loading MiniCPM model from {self.model_path}")

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

        print("✅ MiniCPM model initialized successfully")
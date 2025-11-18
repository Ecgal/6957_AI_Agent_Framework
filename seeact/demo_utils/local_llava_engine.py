import torch
from PIL import Image
import os
import gc
import random
import numpy as np
from transformers import set_seed as hf_set_seed

# Try to import llava with proper error handling
try:
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

    LLAVA_AVAILABLE = True
except ImportError as e:
    LLAVA_AVAILABLE = False
    LLAVA_IMPORT_ERROR = str(e)
    print(f"Warning: LLaVA dependencies not available: {e}")
    print("To use LLaVA, please install it from: https://github.com/haotian-liu/LLaVA")


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


class LocalLlavaEngine:
    def __init__(self, model_path=None, model_base=None, device="mps", load_8bit=False, load_4bit=False, **kwargs):
        if not LLAVA_AVAILABLE:
            raise ImportError(
                f"LLaVA is not properly installed. Error: {LLAVA_IMPORT_ERROR}\n"
                "Please install LLaVA by running:\n"
                "  pip uninstall llava -y\n"
                "  pip install git+https://github.com/haotian-liu/LLaVA.git"
            )

        disable_torch_init()
        # Hardcode the model path
        hardcoded_model_path = "liuhaotian/llava-v1.5-7b"
        self.model_name = get_model_name_from_path(hardcoded_model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            hardcoded_model_path, model_base, self.model_name, load_8bit, load_4bit, device=device
        )
        self.model.eval()
        # Set conversation mode based on model
        if "v1.6-34b" in self.model_name.lower():
            self.conv_mode = "chatml_direct"
        else:
            self.conv_mode = "llava_v0"
        self.roles = ('user', 'assistant') if "mpt" in self.model_name.lower() else conv_templates[self.conv_mode].roles
        self.conv_template = conv_templates[self.conv_mode].copy()
        self.device = device
        # Set stop string and keywords as instance variables
        sep_style = self.conv_template.sep_style
        self.stop_str = self.conv_template.sep if sep_style != SeparatorStyle.TWO else self.conv_template.sep2
        self.keywords = [self.stop_str]

    def load_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return image

    def generate(self, prompt: list = None, max_new_tokens=450, temperature=0, image_path=None,
                 ouput_0=None, turn_number=0, **kwargs):
        # prompt: [prompt0, prompt1, prompt2]
        prompt0, prompt1, prompt2 = prompt
        model_dtype = next(self.model.parameters()).dtype
        model_device = next(self.model.parameters()).device
        image = self.load_image(image_path)
        image_tensor = process_images([image], self.image_processor, self.model.config)
        if image_tensor.ndim == 5 and image_tensor.shape[1] == 5:
            image_tensor = image_tensor[:, 0, :, :, :]
        image_tensor = image_tensor.squeeze(0)
        image_tensor = image_tensor.to(model_device, dtype=model_dtype)
        image_tensor = image_tensor.unsqueeze(0)
        image_size = (image_tensor.shape[-1], image_tensor.shape[-2])  # (W, H)

        conv = self.conv_template.copy()
        conv.system = f"<|im_start|>system\n{prompt0}"
        roles = self.roles

        # First round
        if turn_number == 0:
            if self.model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt1
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt1

            conv.append_message(roles[0], inp)
            conv.append_message(roles[1], None)

            prompt_text = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt_text, self.tokenizer, IMAGE_TOKEN_INDEX,
                                              return_tensors='pt').unsqueeze(0).to(self.model.device)
            # Use instance stop_str/keywords
            stop_str = self.stop_str
            keywords = self.keywords
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image_size],
                    do_sample=False,
                    max_new_tokens=1000,
                    use_cache=False,
                    stopping_criteria=[stopping_criteria]
                )

            output_tokens = output_ids[0].tolist()
            decoded_output_full = self.tokenizer.decode(output_tokens)
            print("Last 100 output chars:", decoded_output_full[-100:])
            print("Stop string:", stop_str)
            print("Stop string tokens:", self.tokenizer.encode(stop_str))
            print("Last 20 output tokens:", output_tokens[-20:])
            print("Decoded last 20 tokens:", self.tokenizer.decode(output_tokens[-20:]))

            decoded_output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            return decoded_output

        # Second round
        elif turn_number == 1:
            if self.model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt1
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt1

            conv.append_message(roles[0], inp)
            conv.append_message(roles[1], ouput_0.strip())
            conv.append_message(roles[0], prompt2)
            conv.append_message(roles[1], None)

            prompt_text = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt_text, self.tokenizer, IMAGE_TOKEN_INDEX,
                                              return_tensors='pt').unsqueeze(0).to(self.model.device)
            # Use instance stop_str/keywords
            stop_str = self.stop_str
            keywords = self.keywords
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image_size],
                    do_sample=False,
                    max_new_tokens=1000,
                    use_cache=False,
                    stopping_criteria=[stopping_criteria]
                )

            output_tokens = output_ids[0].tolist()
            decoded_output_full = self.tokenizer.decode(output_tokens)
            print("Last 100 output chars:", decoded_output_full[-100:])
            print("Stop string:", stop_str)
            print("Stop string tokens:", self.tokenizer.encode(stop_str))
            print("Last 20 output tokens:", output_tokens[-20:])
            print("Decoded last 20 tokens:", self.tokenizer.decode(output_tokens[-20:]))

            decoded_output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            # Save the second round's prompt to a file (overwrite each time)
            prompt_file_path = os.path.join(os.path.dirname(__file__), 'latest_second_round_prompt.txt')
            with open(prompt_file_path, 'w', encoding='utf-8') as f:
                f.write(prompt_text)
            # Save the second round's output to a file (overwrite each time)
            output_file_path = os.path.join(os.path.dirname(__file__), 'latest_second_round_output.txt')
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(decoded_output)

            return decoded_output
        else:
            raise ValueError("turn_number must be 0 or 1 for two-round conversation.")
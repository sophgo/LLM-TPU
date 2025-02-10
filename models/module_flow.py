import torch
import os
import json
import argparse

from model_export import ModelExporter, GreedyHead
from language_model.python_demo.pipeline import Model as Pipeline_Model

class DialogueMapper(Pipeline_Model):
    """
    A utility class for mapping specific model types to their corresponding tokenizers, 
    special tokens, and dialogue-related interaction logic. This class abstracts 
    model-specific behaviors and provides a unified interface for managing dialogue 
    history and formatting inputs for various models.

    Purpose:
        - Encapsulates the logic for initializing tokenizers, configuring model-specific 
          attributes, and defining helper methods for dialogue management.
        - Simplifies interaction with different models by unifying their input/output 
          formatting and dialogue handling.

    Key Features:
        1. Initializes the tokenizer for a specified model type.
        2. Configures model-specific attributes such as the end-of-sequence (EOS) token 
           and system prompts.
        3. Provides helper methods for appending user and assistant messages to the 
           dialogue history.
        4. Formats dialogue history using a chat template specific to the model.
        5. Hides internal differences between models, offering a consistent interface 
           for dialogue management.

    Example Usage:
        >>> mapper = DialogueMapper()
        >>> mapper.map("qwen2", "path/to/tokenizer")
        >>> mapper.append_user(history, "Hello!")
        >>> mapper.append_assistant(history, "Hi, how can I help you?")
        >>> formatted_input = mapper.apply_chat_template(history)

    Note:
        - This class is designed to be extended or modified to support additional models 
          and their specific requirements.
    """
    def __init__(self, model_type, torch_path):
        self.tokenizer_path = torch_path
        self.map(model_type, self.tokenizer_path)

class TestNetWithMask(ModelExporter):
    """
    Base class for exporting large language models (LLMs). Inherits from `ModelExporter`.

    **Key Features**:
        - Manages model initialization, configuration, and tokenizer setup.
        - Provides a chat interface for user interaction.
        - Handles token encoding, history management, and autoregressive decoding.

    **Initialization**:
        - Loads model configuration and tokenizer based on `args`.
        - Initializes dialogue-related utilities using `DialogueMapper`.

    **Methods**:
        - `chat`: Starts a chat session with the model.
        - `encode_tokens`: Encodes dialogue history into token IDs.
        - `stream_answer`: Performs autoregressive decoding to generate responses.
    """

    def __init__(self, args):
        torch.nn.Module.__init__(self)
        # load_model
        self.init_from_args(args)
        self.validate_args(args)
        self.load_model(args.torch_path)

        # config
        config_path = os.path.join(args.torch_path, "config.json")
        with open(config_path, 'r') as file:
            self.config = json.load(file)
        self.model_type = args.model_type if args.model_type is not None else self.config['model_type']
        self.HIDDEN_SIZE = self.config['hidden_size']
        self.NUM_LAYERS = self.config['num_hidden_layers']

        # map
        self.dialogue_map = DialogueMapper(self.model_type, args.torch_path)
        self.tokenizer = self.dialogue_map.tokenizer
        self.EOS = self.dialogue_map.EOS
        self.append_user = self.dialogue_map.append_user
        self.append_assistant = self.dialogue_map.append_assistant
        self.apply_chat_template = self.dialogue_map.apply_chat_template
        self.system_prompt = self.dialogue_map.system_prompt
        self.init_history()

    def validate_args(self, args):
        required_attrs = ['torch_path', 'seq_length']
        for attr in required_attrs:
            if not hasattr(args, attr):
                raise ValueError(f"Argument 'args' is missing required attribute: {attr}")
    
    def init_from_args(self, args):
        self.torch_path = args.torch_path
        self.seq_length = args.seq_length
        self.visual_length = args.visual_length
        self.device = torch.device(args.device)
        if args.device == "cpu":
            self.dtype = torch.float
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = "0"
            if self.model_type in ["qwen2", "qwen2.5"]:
                self.dtype = torch.bfloat16
            else:
                raise ValueError(f"{self.model_type} not support now")

    def init_history(self):
        self.history = [self.system_prompt]
    
    def encode_tokens(self):
        self.append_user(self.history, self.input_str)
        text = self.apply_chat_template(self.history)
        tokens = self.tokenizer(text).input_ids
        return tokens

    def chat(self, text=None):
        """
        Starts an interactive chat session with the model.

        Args:
            text (str, optional): Initial input text. If None, prompts the user for input interactively.

        Notes:
            - Enter "exit", "q", or "quit" to terminate the session.
            - Enter "clear" or "new" to start a new chat session.
        """
        # Instruct
        print(
            """\n=================================================================
1. If you want to quit, please enter one of [q, quit, exit]
2. To create a new chat session, please enter one of [clear, new]
================================================================="""
        )
        # Stop Chatting with "exit" input
        while True:
            self.input_str = text
            if text is None:
                self.input_str = input("\nQuestion: ")
            else:
                print("\nQuestion: ", self.input_str)
            # Quit
            if self.input_str in ["exit", "q", "quit"]:
                break
            # New Chat
            elif self.input_str in ["clear", "new"]:
                text = None
                self.init_history()
            # Chat
            else:
                tokens = self.encode_tokens()

                # check tokens
                if not tokens:
                    print("Sorry: your question is empty!!")
                    return
                if len(tokens) > self.seq_length:
                    print(
                        "The maximum question length should be shorter than {} but we get {} instead.".format(
                            self.seq_length, len(tokens)
                        )
                    )
                    return

                print("\nAnswer: ", end="")
                self.stream_answer(tokens)

    def stream_answer(self, tokens):
        """
        Generates a response token-by-token using autoregressive decoding.

        Args:
            tokens (list[int]): Input token IDs.

        Notes:
            - Decoding stops when the EOS token is generated or the maximum sequence length is reached.
            - Outputs are streamed to the console in real-time.
        """
        # hidden_states
        token_len = len(tokens)
        ids = tokens + (self.seq_length - token_len) * [0]
        input_ids = torch.tensor(ids).view(self.seq_length)
        hidden_states = self.embed(input_ids).view(1, self.seq_length, self.HIDDEN_SIZE)

        # position_ids
        position_ids = list(range(token_len)) + (self.seq_length - token_len) * [0]
        position_ids = torch.tensor([position_ids]).to(self.device)

        # attention_mask
        attention_mask = torch.ones((self.seq_length, self.seq_length)).float() * -10000.0
        for i in range(token_len):
            for j in range(token_len):
                if j <= i:
                    attention_mask[i][j] = 0.0
        attention_mask = attention_mask.view(1, 1, self.seq_length, self.seq_length).to(self.device)

        # prefill
        k_cache = []
        v_cache = []
        for i in range(self.NUM_LAYERS):
            hidden_states[:, token_len] = 0
            hidden_states, past_kv = self.blocks[i](hidden_states.to(self.dtype), position_ids, attention_mask)
            past_k, past_v = past_kv
            k_cache.append(past_k)
            v_cache.append(past_v)
        hidden_states = hidden_states[:, token_len - 1:token_len].view(1, 1, self.HIDDEN_SIZE)
        greedy_head = GreedyHead()
        token = greedy_head(self.lm(hidden_states.to(self.dtype))).view(1)
        out_ids = [int(token)]        
        word = self.tokenizer.decode([int(token)])
        print(word, end="")

        # decoding
        ID_IM_END = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        ID_END = self.tokenizer.convert_tokens_to_ids("<|end|>")
        while int(token) not in self.EOS and token_len < self.seq_length:
            token_len += 1
            input_id = torch.tensor([token]).to(self.device)
            hidden_states = self.embed(input_id).view(1, 1, self.HIDDEN_SIZE)
            position_id = torch.tensor([token_len - 1]).to(self.device)
            attention_mask = torch.zeros((1, 1, 1, self.seq_length + 1)).float().to(self.device)
            attention_mask[:, :, :, token_len:self.seq_length] = -10000.0
            for i in range(self.NUM_LAYERS):
                past_key_value = tuple([k_cache[i].to(self.device), v_cache[i].to(self.device)])
                hidden_states, past_kv = self.blocks[i](hidden_states.to(self.dtype),
                                                        position_id,
                                                        attention_mask.to(self.dtype),
                                                        past_key_value)
                past_k, past_v = past_kv
                k_cache[i][:,token_len:token_len+1] = past_k
                v_cache[i][:,token_len:token_len+1] = past_v
            token = greedy_head(self.lm(hidden_states.to(self.dtype))).view(1)
            out_ids.append(int(token))
            word = self.tokenizer.decode([int(token)])
            print(word, end="")
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='module_flow_test', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-t', '--torch_path', type=str, required=True, help='torch path, like ./Qwen2-VL-2B-Instruct')
    parser.add_argument('--seq_length', type=int, required=True, help="sequence length")
    parser.add_argument('--visual_length', type=int, default=1024, help="visual length for vision transformer")
    parser.add_argument('--device', type=str, default="cpu", help="device")
    parser.add_argument('--model_type', type=str, help="model_type")

    args = parser.parse_args()
    test = TestNetWithMask(args)
    test.chat()

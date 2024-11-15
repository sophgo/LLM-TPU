import time
import argparse
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModelForCausalLM, Qwen2VLForConditionalGeneration, AutoProcessor, AutoTokenizer, PretrainedConfig, Qwen2VLConfig
from qwen_vl_utils import process_vision_info
import chat
import json
import os
import torch

# Preprocess the images
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size),
                 interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def get_position_ids(processor, config, image_path="./image1.jpg", text="Describe this picture and tell a story about this animal."):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                    "resized_height": 280,
                    "resized_width": 420,
                },
                {"type": "text", "text": text},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    SEQ_LENGTH = config['max_position_embeddings']
    # SEQ_LENGTH = self.SEQLEN
    if SEQ_LENGTH <= inputs.input_ids.shape[-1]:
        raise ValueError(
                f"The input_length must be shorter than model's seq_length (got `input_length`: {inputs.input_ids.shape[-1]}"
                f" and `seq_length`: {SEQ_LENGTH})."
            )
    input_ids = inputs.input_ids
    pixel_values = inputs.pixel_values
    image_grid_thw = inputs.image_grid_thw
    input_ids_prefill = torch.zeros(1, SEQ_LENGTH).to(torch.int32)
    input_ids_prefill[:, :input_ids.shape[-1]] = input_ids
    attention_mask_prefill = torch.zeros(1, SEQ_LENGTH)
    attention_mask_prefill[:, :input_ids.shape[-1]] = inputs.attention_mask
    pretrained_config = PretrainedConfig(**config)
    with open('./../compile/files/Qwen2-VL-2B-Instruct/config.json', 'r') as json_file:
        config_dict = json.load(json_file)
        loaded_config = Qwen2VLConfig(**config_dict)
        # print(loaded_config)
    image_mask = (input_ids_prefill == loaded_config.image_token_id)
    true_indices = torch.nonzero(image_mask, as_tuple=True)[1]

    if true_indices.numel() > 0:
        first_true_index = true_indices[0].item()
    else:
        first_true_index = None
    

    # config = Qwen2VLConfig(
    #     # vocab_size=151936,
    #     # hidden_size=1536,
    #     # num_hidden_layers=28,
    #     # num_attention_heads=12,
    #     # intermediate_size=8960,
    #     # max_position_embeddings=32768,
    #     # 添加其他必要的参数
    #     **config
    # )

    # 创建模型实例
    model = Qwen2VLForConditionalGeneration(loaded_config)
    position_ids, _ = Qwen2VLForConditionalGeneration(loaded_config).get_rope_index(
        input_ids_prefill, image_grid_thw, None, attention_mask_prefill
    )

    return position_ids, inputs, first_true_index

class Qwen2VL():

    def __init__(self, args):
        # devid
        self.device = args.devid
        self.processor = AutoProcessor.from_pretrained(args.processor,
                                                       trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer,
                                                       trust_remote_code=True)
        with open(args.config, 'r') as f:
            self.config = json.load(f)

        # load model
        self.model = chat.Qwen2VL()
        self.model.init(self.device, args.model_path)
        self.model.generation_mode = args.generation_mode
        self.POSITION_IDS, _, _ = get_position_ids(processor=self.processor, config=self.config)
        self.SEQLEN = self.model.SEQLEN
        # self.ID_EOS = self.tokenizer.eos_token_id
        self.ID_END = self.tokenizer.convert_tokens_to_ids("<|end|>")
        self.ID_IM_END = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

    # def load_image(self, image_file):
    #     image = Image.open(image_file).convert('RGB')
    #     pixel_values = self.image_transform(image)
    #     return pixel_values

    # def encode(self):
    #     if not self.image_str:
    #         prompt = self.system_prompt + self.input_str + "<|im_end|><|im_start|>assistant\n"
    #         self.input_ids = self.tokenizer.encode(prompt)
    #         self.image_offset = 0
    #         self.pixel_values = []
    #         return
    #     self.pixel_values = self.load_image(self.image_str).flatten().tolist()
    #     self.image_offset = self.system_offset
    #     prompt_ids = self.tokenizer.encode(
    #         "</img>{}<|im_end|><|im_start|>assistant\n".format(self.input_str))
    #     self.input_ids = self.system_prefix + prompt_ids

    def chat(self):
        """
        Start a chat session.
        """
        # Instruct
        print(
            """\n=================================================================
1. If you want to quit, please enter one of [q, quit, exit]
2. To create a new chat session, please enter one of [clear, new]
=================================================================""")
        # Stop Chatting with "exit" input
        while True:
            # self.input_str = input("\nQuestion: ")
            # # Quit
            # if self.input_str in ["exit", "q", "quit"]:
            #     break
            # self.image_str = input("\nImage Path: ")
            # print("\nAnswer:")
            # if self.image_str:
            #     if not os.path.exists(self.image_str):
            #         print("Can't find image: {}".format(self.image_str))
            #         continue

            # self.encode()
            # self.POSITION_IDS, inputs, image_offset = get_position_ids(processor=self.processor, config=self.config, image_path=self.image_str, text=self.input_str)
            self.POSITION_IDS, inputs, image_offset = get_position_ids(processor=self.processor, config=self.config)
            # messages = [
            #     {
            #         "role": "user",
            #         "content": [
            #             {
            #                 "type": "image",
            #                 "image": self.image_str,
            #             },
            #             {"type": "text", "text": self.input_str},
            #         ],
            #     }
            # ]
            # text = processor.apply_chat_template(
            #     messages, tokenize=False, add_generation_prompt=True
            # )
            # image_inputs, video_inputs = process_vision_info(messages)
            # inputs = processor(
            #     text=[text],
            #     images=image_inputs,
            #     videos=video_inputs,
            #     padding=True,
            #     return_tensors="pt",
            # )
            # config = origin_model.config

            # position_ids, _ = Qwen2VLForConditionalGeneration(config).get_rope_index(
            #     input_ids_prefill, image_grid_thw, None, attention_mask_prefill
            # )
            position_ids = self.POSITION_IDS
            breakpoint()
            # Chat
            first_start = time.time()
            token = self.model.forward_first(inputs.input_ids.squeeze(0).tolist(), position_ids.squeeze(1).tolist(), inputs.pixel_values.flatten().tolist(),
                                             inputs.image_grid_thw.squeeze(0).tolist(), image_offset)
            first_end = time.time()
            tok_num = 1
            # Following tokens
            full_word_tokens = []
            text = ""
            while token not in [self.ID_IM_END, self.ID_END
                                ] and self.model.token_length < self.SEQLEN:
                full_word_tokens.append(token)
                word = self.tokenizer.decode(full_word_tokens,
                                             skip_special_tokens=True)
                if "�" not in word:
                    if len(full_word_tokens) == 1:
                        pre_word = word
                        word = self.tokenizer.decode(
                            [token, token],
                            skip_special_tokens=True)[len(pre_word):]
                    text += word
                    print(word, flush=True, end="")
                    full_word_tokens = []
                token = self.model.forward_next()
                tok_num += 1
            next_end = time.time()
            first_duration = first_end - first_start
            next_duration = next_end - first_end
            tps = tok_num / next_duration
            print(f"\nFTL: {first_duration:.3f} s")
            print(f"TPS: {tps:.3f} token/s")


def main(args):
    model = Qwen2VL(args)
    model.chat()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--model_path',
                        type=str,
                        required=True,
                        help='path to the bmodel file')
    parser.add_argument('-t',
                        '--tokenizer',
                        type=str,
                        default="../support/token_config",
                        help='path to the tokenizer file')
    parser.add_argument('-p',
                        '--processor',
                        type=str,
                        default="../support/processor_config",
                        help='path to the processor file')
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default="../compile/files/Qwen2-VL-2B-Instruct/config.json",
                        help='path to the model config file')
    parser.add_argument('-d', '--devid', type=int,
                        default=0, help='device ID to use')
    parser.add_argument('-g',
                        '--generation_mode',
                        type=str,
                        choices=["greedy", "penalty_sample"],
                        default="greedy",
                        help='mode for generating next token')
    args = parser.parse_args()
    main(args)

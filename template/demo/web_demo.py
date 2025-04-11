# -*- coding: utf-8 -*-
import time
import gradio as gr
from pipeline import Model
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--dir_path", type=str, default="./tmp", help="dir path to the config/embedding/tokenizer")
parser.add_argument('-m', '--model_path', type=str, required=True, help='path to the bmodel file')
parser.add_argument('-n', '--model_name', type=str, required=True, help='LLM model name')
parser.add_argument('-t', '--tokenizer_path', type=str, default="../support/token_config", help='path to the tokenizer file')
parser.add_argument('-d', '--devid', type=str, default='0', help='device ID to use')
parser.add_argument('--enable_history', type=bool, default=True, help="if set, enables storing of history memory.")
parser.add_argument('--temperature', type=float, default=1.0, help='temperature scaling factor for the likelihood distribution')
parser.add_argument('--top_p', type=float, default=1.0, help='cumulative probability of token words to consider as a set of candidates')
parser.add_argument('--repeat_penalty', type=float, default=1.0, help='penalty for repeated tokens')
parser.add_argument('--repeat_last_n', type=int, default=32, help='repeat penalty for recent n tokens')
parser.add_argument('--max_new_tokens', type=int, default=1024, help='max new token length to generate')
parser.add_argument('--generation_mode', type=str, choices=["greedy", "penalty_sample"], default="greedy", help='mode for generating next token')
parser.add_argument('--prompt_mode', type=str, choices=["prompted", "unprompted"], default="prompted", help='use prompt format or original input')
parser.add_argument('--test_input', type=str, default=None, help='the text for test')
parser.add_argument('--test_media', type=str, default=None, help='the media(image/video) path for test')
parser.add_argument('--model_type', type=str, help="model type")
# parser.add_argument('--decode_mode', type=str, default="basic", choices=["basic", "jacobi"], help='mode for decoding')
args = parser.parse_args()

pipeline_model = Model(args)


def gr_update_history():
    if pipeline_model.model.token_length >= pipeline_model.model.SEQLEN:
        # print("... (reach the maximal length)", flush=True, end='')
        gr.Warning("reach the maximal length, Model would clear all history record")
        pipeline_model.history = [{"role": "system", "content": pipeline_model.system_prompt}]
    else:
        pipeline_model.history.append({"role": "assistant", "content": pipeline_model.answer_cur})

def gr_user_combined(user_input, history, image_path=None, video_path=None):

    if user_input:
        pipeline_model.input_str = user_input
        history.append([user_input, None])


    if image_path:
        pipeline_model.test_media = image_path
        history.append([f"Uploaded image: {image_path}", None]) 

    if video_path:
        pipeline_model.test_media = video_path
        history.append([f"Uploaded video: {video_path}", None])

    return "", history


def gr_chat(history):
    """
    Stream the prediction for the given query.
    """
    media_type = "text"
    # tokens = pipeline_model.encode_tokens(pipeline_model.input_str)
    # pipelie_model.init_forward(tokens)

    media_path = ""
    if pipeline_model.test_media:
        media_path = pipeline_model.test_media.strip()

        _, ext = os.path.splitext(media_path)

        if ext in [".jpg", ".jpeg", ".png"]:
            media_type = "image"
            # pipeline_model.process_media_input(media_path, media_type)
        elif ext in [".mp4"]:
            media_type = "video"
            # pipeline_model.process_media_input(media_path, media_type)
        else:
            gr.Warning("Invalid media path!!")
            return
    gr.Warning("Input: ", pipeline_model.input_str)
    tokens = pipeline_model.prefill_phase(pipeline_model.input_str, media_path, media_type)

    # # check tokens
    # if not tokens:
    #     gr.Warning("Sorry: your question is empty!!")
    #     return
    # if len(tokens) > model.SEQLEN:
    #     gr.Warning(
    #         "The maximum question length should be shorter than {} but we get {} instead.".format(
    #             model.SEQLEN, len(tokens)
    #         )
    #     )
    #     gr_update_history()

    pipeline_model.answer_cur = ""
    pipeline_model.answer_token = []
    token_num = 0

    first_start = time.time()
    token = pipeline_model.prefill_phase(inputs, media_type)
    first_end = time.time()

    history[-1][1] = ""
    full_word_tokens = []

    while token !=  pipeline_model.EOS and pipeline_model.model.total_length < pipeline_model.model.SEQLEN:
        full_word_tokens.append(token)
        t_word = pipeline_model.tokenizer.decode(full_word_tokens, skip_special_tokens=True)

        if "�" in t_word:
            token = pipeline_model.model.forward_next()
            token_num += 1
            continue

        pipeline_model.answer_token += full_word_tokens

        history[-1][1] += t_word
        full_word_tokens = []
        yield history
        token = pipeline_model.model.forward_next()
        token_num += 1

    next_end = time.time()
    first_duration = first_end - first_start
    next_duration = next_end - first_end
    tps = token_num / next_duration

    print()
    print(f"FTL: {first_duration:.3f} s")
    print(f"TPS: {tps:.3f} token/s")

    pipeline_model.answer_cur = pipeline_model.tokenizer.decode(pipeline_model.answer_token)
    gr_update_history()


def reset():
    pipeline_model.clear()
    return [[None, None]]



description = """
# Sophon TPU 🏁 
"""
holder = ''.join(["Chat with ", args.model_name])
with gr.Blocks() as demo:
    gr.Markdown(description)
    with gr.Row():
        with gr.Column():
            chatbot = gr.Chatbot(label=args.model_name, height=1050)

            with gr.Row():
                image_input = gr.Image(label="Upload Image", type="filepath")
                video_input = gr.Video(label="Upload Video") #

            with gr.Row():
                user_input = gr.Textbox(show_label=False, placeholder=holder, lines=1, min_width=300, scale=6)
                submitBtn = gr.Button("Submit", variant="primary", scale=1)
                emptyBtn = gr.Button(value="Clear", scale=1)

    submitBtn.click(
        gr_user_combined,  
        [user_input, chatbot, image_input, video_input], 
        [user_input, chatbot] 
    ).then(
        gr_chat, chatbot, chatbot 
    )

    emptyBtn.click(reset, outputs=[chatbot])

demo.queue(max_size=20).launch(share=False, server_name="0.0.0.0", inbrowser=True, server_port=8003)


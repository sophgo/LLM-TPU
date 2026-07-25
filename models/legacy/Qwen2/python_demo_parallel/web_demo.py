import time
import gradio as gr
import mdtex2html
from pipeline import Qwen2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_path', type=str, required=True, help='path to the bmodel file')
parser.add_argument('-t', '--tokenizer_path', type=str, default="../support/token_config", help='path to the tokenizer file')
parser.add_argument('-d', '--devid', type=str, default='0', help='device ID to use')
parser.add_argument('--enable_history', action='store_true', help="if set, enables storing of history memory.")
args = parser.parse_args()

model = Qwen2(args)

def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y

def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def gen(input, history):
    i = 0
    history.append((input, ''))
    res = ''
    while i < 10:
        i += 1
        res += str(i)
        time.sleep(0.05)
        history[-1] = (input, res)
        yield res, history


def predict(input, chatbot, history):

    chatbot.append((parse_text(input), ""))
    for response, history in model.stream_predict(input):
        chatbot[-1] = (parse_text(input), parse_text(response))
        yield chatbot, history


def reset_user_input():
    return gr.update(value='')


def reset_state():
    model.clear()
    return [], [], None

gr.Chatbot.postprocess = postprocess

with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">Qwen2-7B TPU</h1>""")

    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(
                    container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(1, 512, value=512, step=1.0, label="Maximum length", interactive=True)

    history = gr.State([])

    submitBtn.click(predict, [user_input, chatbot, history],
                    [chatbot, history], show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)

demo.queue().launch(share=True, server_name="0.0.0.0", server_port=8003, inbrowser=True)


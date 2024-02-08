from copy import deepcopy
import pyglm2
from typing import List, Dict
import numpy as np


class BmGLM2:
    def __init__(self, model_path) -> None:
        self.instance = pyglm2.bmglm2_create()
        pyglm2.bmglm2_init(
            self.instance, 0, model_path)

    def stream_complete(self, input, length_limit):
        pyglm2.bmglm2_init_stream(self.instance, input, length_limit)

    def stream_complete_token(self, input, length_limit):
        pyglm2.bmglm2_init_tokens(self.instance, input, length_limit)

    def complete_token(self, input, eos_tokens, length_limit):
        return pyglm2.bmglm2_complete_tokens(self.instance, input, eos_tokens, length_limit)

    def get_from_stream(self):
        while True:
            res = pyglm2.bmglm2_get_word(self.instance)
            if res.startswith('##'):
                pyglm2.bmglm2_stop_inference(self.instance)
                yield res
                break
            yield res

    def stop(self):
        pyglm2.bmglm2_stop_inference(self.instance)

    def process_response(self, output, history):
        content = ""
        history = deepcopy(history)
        for response in output.split("<|assistant|>"):
            metadata, content = response.split("\n", maxsplit=1)
            if not metadata.strip():
                content = content.strip()
                history.append(
                    {"role": "assistant", "metadata": metadata, "content": content})
                content = content.replace("[[训练时间]]", "2023年")
            else:
                history.append(
                    {"role": "assistant", "metadata": metadata, "content": content})
                if history[0]["role"] == "system" and "tools" in history[0]:
                    content = "\n".join(content.split("\n")[1:-1])

                    def tool_call(**kwargs):
                        return kwargs
                    parameters = eval(content)
                    content = {"name": metadata.strip(),
                               "parameters": parameters}
                else:
                    content = {"name": metadata.strip(), "content": content}
        return content, history

    def chat(self, tokenizer, query: str, history: List[Dict], role: str = 'user', max_length=512, num_beams=1, do_sample=True, top_p=0.8, temperature=0.8, logits_processor=None,
             **kwargs):
        if history is None:
            history = []
        if logits_processor is None:
            logits_processor = None

        input = tokenizer.build_chat_input(query, history=history, role=role)[
            'input_ids'].squeeze().astype(np.int32)

        eos_token_id = np.array([tokenizer.eos_token_id, tokenizer.get_command("<|user|>"),
                                 tokenizer.get_command("<|observation|>")], dtype=np.int32)

        outputs = self.complete_token(
            input=input, eos_tokens=eos_token_id, length_limit=max_length)
        response = tokenizer.decode(outputs)
        history.append({"role": role, "content": query})
        response, history = self.process_response(response, history)
        return response, history

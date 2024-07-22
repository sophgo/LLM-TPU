import streamlit as st
from pipeline import Qwen
from PIL import Image
import argparse
import configparser
import time

config = configparser.ConfigParser()
config.read('./supports/config.ini')
token_path = config.get('qwenvl','token_path')
bmodel_path = config.get('qwenvl','bmodel_path')
dev_id = str(config.get('qwenvl', 'dev_id'))

args = argparse.Namespace(
    model_path = bmodel_path,
    tokenizer_path = token_path,
    devid = dev_id,
    temperature=1.0,
    top_p=1.0,
    repeat_penalty=1.0,
    repeat_last_n=32,
    max_new_tokens=1024,
    generation_mode='greedy',
    prompt_mode='prompted',
    decode_mode='basic',
    enable_history=False,
)

st.title("Qwen-VL-Chat")

def display_uploaded_image(image):
    st.sidebar.image(image, caption='Uploaded Image', use_column_width=True)

with st.sidebar:
    uploaded_file = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    display_uploaded_image(uploaded_file)

    if "message" not in st.session_state:
        st.session_state.messages = [] 

    if "client" not in st.session_state:
        st.session_state.client = Qwen(args)
        st.success('模型初始化完成！欢迎您根据图片提出问题，我将会为您解答。', icon='�')
        st.balloons()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("请输入您的问题 "):
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            stream = st.session_state.client.chat_stream(prompt, image, history=[[m["role"], m["content"]] for m in st.session_state.messages])
            response = st.write_stream(stream)

            st.session_state.messages.append({"role": "assistant", "content": response})
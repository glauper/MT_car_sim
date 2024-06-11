import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import time
import json
import os
import re
from functions.plot_functions import (plot_simulation_env_0, plot_simulation_env_1, plot_simulation_env_2,
                                      plot_simulation_env_3, plot_simulation_env_4, plot_simulation_env_5,
                                      prep_plot_vehicles, prep_plot_acc_input, prep_plot_steer_input)

st.set_page_config(layout="wide")

# Initialize session state for rectangle position and chat history
if 'results' not in st.session_state:
    results_path = os.path.join(os.path.dirname(__file__), ".", "save_results/results.txt")
    with open(results_path, 'r') as file:
        file_content = file.read()
    results = json.loads(file_content)
    st.session_state.results = results
else:
    results = st.session_state.results
if 'env' not in st.session_state:
    env_path = os.path.join(os.path.dirname(__file__), ".", "save_results/env.txt")
    with open(env_path, 'r') as file:
        file_content = file.read()
    env = json.loads(file_content)
    st.session_state.env = env
else:
    env = st.session_state.env
if 'messages' not in st.session_state:
    path_messages = os.path.join(os.path.dirname(__file__), ".", "prompts/output_LLM/messages.json")
    with open(path_messages, 'r') as file:
        messages = json.load(file)
    st.session_state.messages = messages
else:
    messages = st.session_state.messages

if 'step' not in st.session_state:
    st.session_state.step = 0
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'change_task' not in st.session_state:
    st.session_state.change_task = True
if 'figures' not in st.session_state:
    st.session_state.figures = []

# Divide the page into two columns
col1, col2, col3 = st.columns([1.5,1,1])


# Right column: Plot
with col2:
    st.header(f"Time t = {st.session_state.step}")
    path_fig = os.path.join(os.path.dirname(__file__), ".", f"save_results/images_sim/")
    st.image(path_fig + f'{st.session_state.step}.png')

with col3:
    path_fig = os.path.join(os.path.dirname(__file__), ".", f"save_results/images_acc/")
    st.image(path_fig + f'{st.session_state.step}.png')
    path_fig = os.path.join(os.path.dirname(__file__), ".", f"save_results/images_steer/")
    st.image(path_fig + f'{st.session_state.step}.png')

# Left column: Chat interface
with (col1):
    for k in range(len(messages)):
        if 'User' in messages[k].keys() and messages[k]['time'] == st.session_state.step:
            with st.chat_message('User'):
                message = f'Time t = {st.session_state.step}:\n' + '- ' + messages[k]['User']
                st.markdown(message, unsafe_allow_html=True)
                time.sleep(1)
            if 'Task Planner' in messages[k+1].keys() and messages[k+1]['time'] == st.session_state.step:
                st.session_state.chat_history.insert(0,{'role': 'User', 'content': message})
                #message = f"Time t = {st.session_state.step}:\n"
                message = """"""
                for nr, task in enumerate(messages[k+1]['Task Planner']['tasks']):
                    message += f'{nr+1}. ' + task + '\n'

                st.session_state.chat_history.insert(1, {'role': 'Task Planner','content': message})
                with st.chat_message('Task Planner'):
                    st.markdown(message)
                    time.sleep(1)
            if 'Optimization Designer' in messages[k+1].keys() and messages[k+1]['time'] == st.session_state.step:
                st.session_state.chat_history.insert(2,{'role': 'User', 'content': message})
                #message = f"Time t = {st.session_state.step}:\n" + """
                message = """objective = """ + str(messages[k+1]['Optimization Designer']['objective']) + """
equality_constraints = """ + str(messages[k + 1]['Optimization Designer']['equality_constraints']) + """
inequality_constraints = """ + str(messages[k + 1]['Optimization Designer']['inequality_constraints'])
                st.session_state.chat_history.insert(3, {'role': 'Optimization Designer', 'content': message})
                with st.chat_message('Optimization Designer'):
                    st.code(message)
                    time.sleep(1)
        if 'Vehicle' in messages[k].keys() and messages[k]['time'] == st.session_state.step:
            message = f"Time t = {st.session_state.step}:\n" + '- ' + messages[k]['Vehicle']
            st.session_state.chat_history.insert(0, {'role': 'Vehicle', 'content': message})
            with st.chat_message('Vehicle'):
                st.markdown(message, unsafe_allow_html=True)
                time.sleep(1)

    for chat_message in st.session_state.chat_history:
        with st.chat_message(chat_message['role']):
            if chat_message['role'] == 'User':
                st.markdown(chat_message['content'], unsafe_allow_html=True)
            elif chat_message['role'] == 'Vehicle':
                st.markdown(chat_message['content'], unsafe_allow_html=True)
            elif chat_message['role'] == 'Optimization Designer':
                st.code(chat_message['content'])
            elif chat_message['role'] == 'Task Planner':
                st.markdown(chat_message['content'])

# Rerun the app after a short delay
time.sleep(0.5)
st.session_state.step += 1
if st.session_state.step < len(st.session_state.results['agent 0']['x coord']):
    st.rerun()
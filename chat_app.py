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
                                      prep_plot_vehicles)

st.set_page_config(layout="wide", )

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
    for t in range(len(results['agent 0']['x coord'])):
        fig, ax = plt.subplots()
        if env['env number'] == 0:
            ani, ax, fig = plot_simulation_env_0(env, results, None, None, fig, ax)
        elif env['env number'] == 1:
            ani, ax, fig = plot_simulation_env_1(env, results, None, None, fig, ax)
        elif env['env number'] == 2:
            ani, ax, fig = plot_simulation_env_2(env, results, None, None, fig, ax)
        elif env['env number'] == 3:
            ani, ax, fig = plot_simulation_env_3(env, results, None, None, fig, ax)
        elif env['env number'] == 4:
            ani, ax, fig = plot_simulation_env_4(env, results, None, None, fig, ax)
        elif env['env number'] == 5:
            ani, ax, fig = plot_simulation_env_5(env, results, None, None, fig, ax)

        ax.set_aspect('equal')

        vehicles, labels, lines, ax = prep_plot_vehicles(results, env, t, ax)

        ax.set_xlim(env["State space"]["x limits"][0], env["State space"]["x limits"][1])
        ax.set_ylim(env["State space"]["y limits"][0], env["State space"]["y limits"][1])

        path_fig = os.path.join(os.path.dirname(__file__), ".", f"save_results/images/")
        fig.savefig(path_fig + f'{t}.png')
        plt.close(fig)

# Divide the page into two columns
col1, col2 = st.columns([2,1])

# Left column: Chat interface
with col1:
    new_call = False
    for k in range(len(messages)):
        if 'User' in messages[k].keys() and messages[k]['time'] == st.session_state.step:
            new_call = True
            if 'Task Planner' in messages[k+1].keys() and messages[k+1]['time'] == st.session_state.step:
                st.session_state.chat_history.insert(0,{'role': 'User', 'content': messages[k]['User']})
                message = """
                """
                for nr, task in enumerate(messages[k+1]['Task Planner']['tasks']):
                    message += f'{nr+1}) ' + task + """
                """
                st.session_state.chat_history.insert(1, {'role': 'Task Planner','content': message})
            if 'Optimization Designer' in messages[k+1].keys() and messages[k+1]['time'] == st.session_state.step:
                st.session_state.chat_history.insert(0,{'role': 'User', 'content': messages[k]['User']})
                message = """
                objective = """ + str(messages[k+1]['Optimization Designer']['objective']) + """
                equality_constraints = """ + str(messages[k + 1]['Optimization Designer']['equality_constraints']) + """
                inequality_constraints = """ + str(messages[k + 1]['Optimization Designer']['inequality_constraints'])
                st.session_state.chat_history.insert(1, {'role': 'Optimization Designer', 'content': message})

    # Display chat history
    for chat_message in st.session_state.chat_history:
        with st.chat_message(chat_message['role']):
            if chat_message['role'] == 'User':
                st.markdown(chat_message['content'], unsafe_allow_html=True)
                if new_call:
                    time.sleep(1)
            elif chat_message['role'] == 'Optimization Designer':
                st.code(chat_message['content'])
                if new_call:
                    time.sleep(1)
            elif chat_message['role'] == 'Task Planner':
                st.markdown(chat_message['content'])
                if new_call:
                    time.sleep(1)

# Right column: Plot
with col2:
    st.header(f"Time t = {st.session_state.step}")
    path_fig = os.path.join(os.path.dirname(__file__), ".", f"save_results/images/")
    st.image(path_fig + f'{st.session_state.step}.png')

# Rerun the app after a short delay
time.sleep(0.5)
st.session_state.step += 1
if st.session_state.step < len(st.session_state.results['agent 0']['x coord']):
    st.rerun()
import json
import os
from io import BytesIO
from base64 import b64encode
from matplotlib.animation import HTMLWriter
import streamlit as st
from functions.plot_functions import plot_simulation, input_animation

path_messages = os.path.join(os.path.dirname(__file__), ".", "prompts/output_LLM/messages.json")
with open(path_messages, 'r') as file:
    messages = json.load(file)

results_path = os.path.join(os.path.dirname(__file__), ".", "save_results/results.txt")
with open(results_path, 'r') as file:
    file_content = file.read()
results = json.loads(file_content)

env_path = os.path.join(os.path.dirname(__file__), ".", "save_results/env.txt")
with open(env_path, 'r') as file:
    file_content = file.read()
env = json.loads(file_content)

# Create a function to format messages for display
def display_messages(messages):
    for k in range(len(messages)):
        if 'Task Planner' in messages[k].keys():
            st.write(f"Task Planner t = {messages[k]['time']}")
            st.write(messages[k]['Task Planner'])
        elif 'Optimization Designer' in messages[k].keys():
            st.write(f"Optimization Designer t = {messages[k]['time']}")
            st.write(f"Query: ", messages[k-1]['Task Planner']['tasks'][0])
            st.write(f"Objective: ", messages[k]['Optimization Designer']['objective'])
            st.write(f"Equality constraints:")
            st.write(messages[k]['Optimization Designer']['equality_constraints'])
            st.write(f"Inequality constraints:")
            st.write(messages[k]['Optimization Designer']['inequality_constraints'])
        elif 'Images' in messages[k].keys():
            ani = plot_simulation(env['env number'], env, results, t_start=messages[k]['Images']['t_start'], t_end=messages[k]['Images']['t_end'])
            path = os.path.join(os.path.dirname(__file__), ".", "prompts/output_LLM/")
            ani.save(path + f'animation_{k}.gif', writer='pillow')
            st.image(path + f'animation_{k}.gif')

            #col1, col2, col3 = st.columns(3)
            #with col1:
                #st.image(path + f'animation_{k}.gif', caption='GIF 1', use_column_width=True)
            #ani1, ani2 = input_animation(results, t_start=messages[k]['Images']['t_start'], t_end=messages[k]['Images']['t_end'])

            #ani1.save(path + f'acc_{k}.gif', writer='pillow')
            #with col2:
                #st.image(path + f'acc_{k}.gif', caption='GIF 2', use_column_width=True)
            #st.image(path + f'acc_{k}.gif')

            #ani2.save(path + f'steer_{k}.gif', writer='pillow')
            #with col3:
                #st.image(path + f'steer_{k}.gif', caption='GIF 3', use_column_width=True)
            #st.image(path + f'steer_{k}.gif')

# Streamlit app
st.title("Simulation Environment "+str(env['env number']))
display_messages(messages)

import json
import os
import streamlit as st
from datetime import datetime

path_messages = os.path.join(os.path.dirname(__file__), ".", "prompts/output_LLM/messages.json")
with open(path_messages, 'r') as file:
    messages = json.load(file)

# Create a function to format messages for display
def display_messages(messages):
    for k in range(len(messages)):
        if 'Task Planner' in messages[k].keys():
            st.write("## Task Planner")
            st.write(messages[k]['Task Planner'])
        elif 'Optimization Designer' in messages[k].keys():
            st.write("## Optimization Designer")
            st.write(f"Query:")
            st.write(messages[k-1]['Task Planner']['tasks'][0])
            st.write(f"Objective:")
            st.write(messages[k]['Optimization Designer']['objective'])
            st.write(f"Equality constraints:")
            st.write(messages[k]['Optimization Designer']['equality_constraints'])
            st.write(f"Inequality constraints:")
            st.write(messages[k]['Optimization Designer']['inequality_constraints'])
        elif 'Gif' in messages[k].keys():
            gif_path = os.path.join(os.path.dirname(__file__), "animation/animation.gif")
            st.image(gif_path)

# Streamlit app
st.title("Chat Messages")

display_messages(messages)

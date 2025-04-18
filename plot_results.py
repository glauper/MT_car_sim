import numpy as np
import os
import json
import random
from config.config import SimulationConfig, EnviromentConfig
from functions.plot_functions import input_animation, plot_simulation, save_all_frames
from env_controller import EnvController
from vehicle import Vehicle

results_path = os.path.join(os.path.dirname(__file__), ".","save_results/results.txt")
with open(results_path, 'r') as file:
    file_content = file.read()
results = json.loads(file_content)

env_path = os.path.join(os.path.dirname(__file__), ".","save_results/env.txt")
with open(env_path, 'r') as file:
    file_content = file.read()
env = json.loads(file_content)

plot_simulation(env['env number'], env, results, t_start=0, t_end=len(results['agent 0']['x coord']))

if env['With LLM car']:
    #plot_input_LLM_and_SF(results)
    input_animation(results,t_start=0, t_end=len(results['agent 0']['x coord']))

    save_all_frames(results, env)
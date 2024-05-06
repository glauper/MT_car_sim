import numpy as np
import os
import json
import random
from config.config import SimulationConfig, EnviromentConfig
from functions.plot_functions import plot_simulation
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

plot_simulation(env['env number'], env, results)

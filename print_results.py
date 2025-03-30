import numpy as np
import shutil
import time
import re
import json
import os
from functions.plot_functions import plot_simulation, input_animation, save_all_frames
from functions.sim_functions import (results_update_and_save, sim_init, sim_reload, check_proximity, check_crash,
                                          check_need_replan, check_safe_zone)

import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file
excel_file = "results.xlsx"

# Read the first sheet or specify the sheet name
data_Safe_NARRATE = pd.read_excel(excel_file, sheet_name="Safe-NARRATE")  # Use sheet_name="Sheet1" for a named sheet
data_NARRATE = pd.read_excel(excel_file, sheet_name="NARRATE")
data_LLM_MPC = pd.read_excel(excel_file, sheet_name="LLM-MPC")
data_LLM_safe = pd.read_excel(excel_file, sheet_name="LLM-Safe")

nr_LLM_calls_safe_narrate = data_Safe_NARRATE['#TP']
nr_LLM_calls_narrate = data_NARRATE['TP calls']
nr_LLM_calls_LLM_MPC = data_LLM_MPC['LLM calls']
nr_LLM_calls_LLM_safe = data_LLM_safe['LLM calls']

data = [nr_LLM_calls_safe_narrate,
        nr_LLM_calls_narrate,
        nr_LLM_calls_LLM_MPC,
        nr_LLM_calls_LLM_safe
]

plt.boxplot(data, labels=["Safe-NARRATE", "NARRATE", "LLM-MPC", "LLM-Safe"])

plt.title("Number LLM calls in each run")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

nr_invasions_safe_narrate = data_Safe_NARRATE['Invasion']
nr_invasions_narrate = data_NARRATE['Invasion']
nr_invasions_LLM_MPC = data_LLM_MPC['Invasion']
nr_invasions_LLM_safe = data_LLM_safe['Invasion']

data = [nr_invasions_safe_narrate,
        nr_invasions_narrate,
        nr_invasions_LLM_MPC,
        nr_invasions_LLM_safe
]

plt.boxplot(data, labels=["Safe-NARRATE", "NARRATE", "LLM-MPC", "LLM-Safe"])

plt.title("Number invasions in each run")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

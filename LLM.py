import json
import os
import re
import numpy as np
from openai import OpenAI
from prompts.prompts import get_prompt, get_help_prompt, get_another_help_prompt

def get_TP(system_chioce, simulation_case):

    if system_chioce == "1D Linear System":

        TP_path = os.path.join(os.path.dirname(__file__), ".", "prompts/"+str(simulation_case)+"/one_dim_system/TP_output.txt")

    elif system_chioce == "2D Linear System":

        TP_path = os.path.join(os.path.dirname(__file__), ".", "prompts/"+str(simulation_case)+"/two_dim_system/TP_output.txt")

    elif system_chioce == "Unicycle":

        TP_path = os.path.join(os.path.dirname(__file__), ".", "prompts/"+str(simulation_case)+"/unicycle/TP_output.txt")

    # Upload the Task Planner from ChatGPT
    with open(TP_path, 'r') as file:
        file_content = file.read()
    Task_Panner = json.loads(file_content)

    return Task_Panner

def get_OD(system_chioce, simulation_case, OD_task_status):

    if system_chioce == "1D Linear System":

        OD_path = os.path.join(os.path.dirname(__file__), ".", "prompts/"+str(simulation_case)+"/one_dim_system/OD_output.txt")

    elif system_chioce == "2D Linear System":

        OD_path = os.path.join(os.path.dirname(__file__), ".", "prompts/"+str(simulation_case)+"/two_dim_system/OD_" + str(OD_task_status+1) + ".txt")

    elif system_chioce == "Unicycle":

        OD_path = os.path.join(os.path.dirname(__file__), ".", "prompts/"+str(simulation_case)+"/unicycle/OD_" + str(OD_task_status+1) + ".txt")

    # Upload the Optimization Designer from ChatGPT
    with open(OD_path, 'r') as file:
        file_content = file.read()
    Optimization_Designer = json.loads(file_content)

    return Optimization_Designer

def ask_TP(system_type, simulation_case):
    api_key_path = os.path.join(os.path.dirname(__file__), ".", "keys/api_key.txt")
    with open(api_key_path, 'r') as file:
        api_key = file.read()

    PROMPTS = get_prompt()
    user_input = PROMPTS[simulation_case]['TP'][system_type]
    print('LLM TP Input:', user_input)

    message = [
        {
            "role": "user",
            "content": user_input,
        }
    ]
    Task_Panner = chat(message, api_key)
    # Remove all \ and \n
    Task_Panner = json.loads(Task_Panner)
    message.append({
        "role": "assistant",
        "content": str(Task_Panner),
    })
    print('LLM TP Output:', Task_Panner)
    # Save the output
    save_TP_path = os.path.join(os.path.dirname(__file__), ".", "prompts/output_LLM/TP_output.txt")
    with open(save_TP_path, 'w') as file:
        json.dump(Task_Panner, file)

    return Task_Panner, message

def ask_help_to_TP(message, feasible_state_space, obstacles, simulation_case, x, target, obstacle, task, id_task):
    api_key_path = os.path.join(os.path.dirname(__file__), ".", "keys/api_key.txt")
    with open(api_key_path, 'r') as file:
        api_key = file.read()

    user_input = get_help_prompt(simulation_case, x, target, obstacle, task)
    print(user_input)

    message.append({
        "role": "user",
        "content": user_input,
    })

    Task_Panner = chat(message, api_key)
    # Remove all \ and \n
    Task_Panner = json.loads(Task_Panner)
    message.append({
        "role": "assistant",
        "content": str(Task_Panner),
    })

    print(Task_Panner)
    """while not check_temporary_target(Task_Panner['tasks'][id_task], feasible_state_space, obstacles):

        user_input = get_another_help_prompt(simulation_case, x, target, obstacle, task)

        message.append({
            "role": "user",
            "content": user_input,
        })

        Task_Panner = chat(message, api_key)
        # Remove all \ and \n
        Task_Panner = json.loads(Task_Panner)
        message.append({
            "role": "assistant",
            "content": str(Task_Panner),
        })

        print(Task_Panner)"""

    print('LLM TP Output:', Task_Panner)
    # Save the output
    save_TP_path = os.path.join(os.path.dirname(__file__), ".", "prompts/output_LLM/TP_output"+str(id_task)+".txt")
    with open(save_TP_path, 'w') as file:
        json.dump(Task_Panner, file)

    return Task_Panner, message

def ask_correction_to_TP(message, feasible_state_space, obstacles, simulation_case, x, target, obstacle, task, id_task):
    api_key_path = os.path.join(os.path.dirname(__file__), ".", "keys/api_key.txt")
    with open(api_key_path, 'r') as file:
        api_key = file.read()

    user_input = get_another_help_prompt(simulation_case, x, target, obstacle, task)
    print(user_input)

    message.append({
        "role": "user",
        "content": user_input,
    })

    Task_Panner = chat(message, api_key)
    # Remove all \ and \n
    Task_Panner = json.loads(Task_Panner)
    message.append({
        "role": "assistant",
        "content": str(Task_Panner),
    })

    print(Task_Panner)
    """while not check_temporary_target(Task_Panner['tasks'][id_task], feasible_state_space, obstacles):

        user_input = get_another_help_prompt(simulation_case, x, target, obstacle, task)

        message.append({
            "role": "user",
            "content": user_input,
        })

        Task_Panner = chat(message, api_key)
        # Remove all \ and \n
        Task_Panner = json.loads(Task_Panner)
        message.append({
            "role": "assistant",
            "content": str(Task_Panner),
        })

        print(Task_Panner)"""

    print('LLM TP Output:', Task_Panner)
    # Save the output
    save_TP_path = os.path.join(os.path.dirname(__file__), ".", "prompts/output_LLM/TP_output"+str(id_task)+".txt")
    with open(save_TP_path, 'w') as file:
        json.dump(Task_Panner, file)

    return Task_Panner, message

def check_temporary_target(text, space, obstacles):
    flags = []
    coordinates_str = text.split('(')[-1].split(')')[0]
    coordinates = tuple(map(float, coordinates_str.split(',')))
    pos = np.array([[coordinates[0]], [coordinates[1]]])
    if space['x limits'][0] < coordinates[0] < space['x limits'][1] and space['y limits'][0] < coordinates[1] < space['y limits'][1]:
        flags.append(True)
        if len(obstacles) != 0:
            for id_obst in obstacles:
                M = np.eye(2) / (obstacles[id_obst]['radius'] + obstacles[id_obst]['min distance']) ** 2
                diff = pos - obstacles[id_obst]['center']
                flags.append(diff.T @ M @ diff >= 1)
    else:
        flags.append(False)

    if all(flags) == True:
        return True
    else:
        return False

def ask_OD(system_type, simulation_case, OD_task_status, task, message, obj_prompt):

    api_key_path = os.path.join(os.path.dirname(__file__), ".", "keys/api_key.txt")
    with open(api_key_path, 'r') as file:
        api_key = file.read()

    if OD_task_status == 0:
        PROMPTS = get_prompt()
        user_input = PROMPTS[simulation_case]['OD'][system_type]
        query = "Query: " + task
        user_input = user_input + f"{obj_prompt}\n{query}"
        message = [
            {
                "role": "user",
                "content": user_input,
            }
        ]
    else:
        query = "Query: " + task
        user_input = f"{obj_prompt}\n{query}"
        message.append({
                "role": "user",
                "content": user_input,
            })

    Optimization_Designer = chat(message, api_key)
    Optimization_Designer = json.loads(Optimization_Designer)  # Remove all \ and \n
    message.append({
        "role": "assistant",
        "content": str(Optimization_Designer),
    })
    print(message)

    save_OD_path = os.path.join(os.path.dirname(__file__), ".", "prompts/output_LLM/OD_"+str(OD_task_status+1)+".txt")
    with open(save_OD_path, 'w') as file:
        json.dump(Optimization_Designer, file)

    return Optimization_Designer, message

def chat(message, api_key):

    client = OpenAI(
        # This is the default and can be omitted
        api_key=api_key,
    )

    chat_completion = client.chat.completions.create(
        messages=message,
        model="gpt-4-turbo", # gpt-4-turbo, gpt-3.5-turbo-0125
        response_format={"type": "json_object"},
    )

    return chat_completion.choices[0].message.content


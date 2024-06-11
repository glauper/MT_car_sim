import json
import os
from openai import OpenAI
from prompts.prompts import general_OD_prompt, specific_OD_prompt, general_TP_prompt, specific_TP_prompt
import numpy as np
import casadi as ca
import random

class LLM:
    def __init__(self):
        self.TP_messages = []
        self.TP = {}
        self.task_status = 0
        self.OD_messages = []
        self.OD = {}
        self.final_messages = []

    def call_TP(self, env, query, agents, ego, t):

        prompt = general_TP_prompt()
        description = specific_TP_prompt(env, agents, ego, query)
        user_input = prompt + description
        #print(user_input)

        self.TP_messages = [
            {
                "role": "user",
                "content": user_input,
            }
        ]
        self.TP = self.chat(self.TP_messages)
        # Remove all \ and \n
        self.TP = json.loads(self.TP)
        self.TP_messages.append({
            "role": "assistant",
            "content": str(self.TP),
        })

        self.final_messages.append({'User': query,
                                    'time': t})
        self.final_messages.append({'Task Planner': self.TP,
                                    'time': t})

        # Save the output
        save_TP_path = os.path.join(os.path.dirname(__file__), ".", "prompts/output_LLM/TP_output.json")
        with open(save_TP_path, 'w') as file:
            json.dump(self.TP_messages, file)

    def recall_TP(self, env_nr, query, agents, ego, why, t):

        if why['next_task']:
            motivation = """You have to replan because the task '""" + self.TP['tasks'][self.task_status-1] + """' is finished. In the next the description of the actual situation.
        """
        elif why['SF_kicks_in'] or why['other_agent_too_near'] or  why['MPC_LLM_not_solved'] or why['SF_not_solved']:
            motivation = """The execution of the task '""" + self.TP['tasks'][self.task_status] + """' is not completed, but you have to replan to ensure that the task can be continued. In the next the description of the actual situation.
        """

        description = specific_TP_prompt(env_nr, agents, ego, query)
        user_input = motivation + description
        #print(user_input)

        self.TP_messages.append({
            "role": "user",
            "content": user_input,
        })

        self.TP = self.chat(self.TP_messages)
        # Remove all \ and \n
        self.TP = json.loads(self.TP)
        self.TP_messages.append({
            "role": "assistant",
            "content": str(self.TP),
        })

        if why['next_task']:
            motivation = """Replan because task finished. Task: """
        elif why['other_agent_too_near']:
            motivation = """Replan because other agents is too near. Task: """
        elif why['SF_kicks_in']:
            motivation = """Replan because cost SF too high. Task: """
        elif why['MPC_LLM_not_solved']:
            motivation = """Replan because MPC LLM no success with solver. Task: """
        elif why['SF_not_solved']:
            motivation = """Replan because SF no success with solver. Task: """

        self.final_messages.append({'User': motivation + query,
                                    'time': t})
        self.final_messages.append({'Task Planner': self.TP,
                                    'time': t})

        self.task_status = 0

        # Save the output
        save_TP_path = os.path.join(os.path.dirname(__file__), ".", "prompts/output_LLM/TP_output.json")
        with open(save_TP_path, 'w') as file:
            json.dump(self.TP_messages, file)

    def call_OD(self, env_nr, agents, t):

        prompt = general_OD_prompt()
        objects = specific_OD_prompt(env_nr, agents)
        user_input = prompt + objects + 'Query: ' + self.TP['tasks'][self.task_status]
        #print(user_input)

        self.OD_messages = [
            {
                "role": "user",
                "content": user_input,
            }
        ]

        self.OD = self.chat(self.OD_messages)
        self.OD = json.loads(self.OD)  # Remove all \ and \n
        self.OD_messages.append({
            "role": "assistant",
            "content": str(self.OD),
        })

        self.final_messages.append({'User': self.TP['tasks'][self.task_status],
                                    'time': t})
        self.final_messages.append({'Optimization Designer': self.OD,
                                    'time': t})
        final_messages_path = os.path.join(os.path.dirname(__file__), ".", "prompts/output_LLM/messages.json")
        with open(final_messages_path, 'w') as file:
            json.dump(self.final_messages, file)

        save_OD_path = os.path.join(os.path.dirname(__file__), ".", "prompts/output_LLM/OD_output.json")
        with open(save_OD_path, 'w') as file:
            json.dump(self.OD_messages, file)

    def recall_OD(self, env_nr, agents, t):

        objects = specific_OD_prompt(env_nr, agents)
        user_input = objects + 'Query: ' + self.TP['tasks'][self.task_status]
        #print(user_input)

        self.OD_messages.append({
            "role": "user",
            "content": user_input,
        })

        self.OD = self.chat(self.OD_messages)
        self.OD = json.loads(self.OD)  # Remove all \ and \n
        self.OD_messages.append({
            "role": "assistant",
            "content": str(self.OD),
        })

        self.final_messages.append({'User': self.TP['tasks'][self.task_status],
                                    'time': t})
        self.final_messages.append({'Optimization Designer': self.OD,
                                    'time': t})
        final_messages_path = os.path.join(os.path.dirname(__file__), ".", "prompts/output_LLM/messages.json")
        with open(final_messages_path, 'w') as file:
            json.dump(self.final_messages, file)

        save_OD_path = os.path.join(os.path.dirname(__file__), ".", "prompts/output_LLM/OD_output.json")
        with open(save_OD_path, 'w') as file:
            json.dump(self.OD_messages, file)


    def chat(self, message):

        api_key_path = os.path.join(os.path.dirname(__file__), ".", "keys/api_key.txt")
        with open(api_key_path, 'r') as file:
            api_key = file.read()

        client = OpenAI(
            # This is the default and can be omitted
            api_key=api_key,
        )

        chat_completion = client.chat.completions.create(
            messages=message,
            model="gpt-4o",  # gpt-4-turbo, gpt-3.5-turbo-0125, gpt-4o
            response_format={"type": "json_object"},
        )

        return chat_completion.choices[0].message.content
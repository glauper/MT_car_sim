import json
import os
from openai import OpenAI
from prompts.prompts import (general_OD_prompt, specific_OD_prompt, general_TP_prompt, specific_TP_prompt,
                             general_DE_prompt,specific_DE_prompt, TP_prompt_with_DE, LLM_conditioned_MPC_general,
                             LLM_conditioned_MPC_specific, prompt_LLM_coder_general, prompt_LLM_coder_specific,
                             prompt_LLM_correction_general, general_TP_reasoning_prompt, specific_TP_reasoning_prompt)
from functions.plot_functions import plot_frame_for_describer
import base64
import numpy as np
import casadi as ca
import random

class LLM:
    def __init__(self, DE_active):
        self.TP_messages = []
        self.TP = {}
        self.task_status = 0
        self.OD_messages = []
        self.OD = {}
        self.final_messages = []
        self.DE = {}
        self.DE_active = DE_active
        self.DE_messages = []
        self.reasoning_active = False

        self.LLM_coder_messages = []
        self.LLM_coder = {}
        self.LLM_correction_messages = []
        self.LLM_correction = {}

    def call_TP(self, env, query, agents, ego, t):

        if self.DE_active:
            DE_description = self.call_DE(env, agents, ego, query, t)
            user_input = TP_prompt_with_DE(DE_description)
        elif self.reasoning_active:
            prompt = general_TP_reasoning_prompt()
            description = specific_TP_reasoning_prompt(env, agents, ego, query)
            user_input = prompt + description
        else:
            prompt = general_TP_prompt()
            description = specific_TP_prompt(env, agents, ego, query)
            user_input = prompt + description

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

        final_messages_path = os.path.join(os.path.dirname(__file__), ".", "prompts/output_LLM/messages.json")
        with open(final_messages_path, 'w') as file:
            json.dump(self.final_messages, file)

    def recall_TP(self, env, query, agents, ego, why, t):

        if why['next_task']:
            motivation = """You have to replan because the task '""" + self.TP['tasks'][self.task_status-1] + """' is finished. In the next the description of the actual situation.
        """
        elif why['SF_kicks_in'] or why['other_agent_too_near'] or  why['MPC_LLM_not_solved'] or why['SF_not_solved'] or why['soft_SF_kicks_in'] or why['soft_SF_psi_not_solved'] or why['soft_SF_not_solved']:
            motivation = """The execution of the task '""" + self.TP['tasks'][self.task_status] + """' is not completed, but you have to replan to ensure that the task can be continued. In the next the description of the actual situation.
        """

        if self.DE_active:
            DE_description = self.call_DE(env, agents, ego, query, t)
            user_input = motivation + DE_description
        elif self.reasoning_active:
            description = specific_TP_reasoning_prompt(env, agents, ego, query)
            user_input = motivation + description
        else:
            description = specific_TP_prompt(env, agents, ego, query)
            user_input = motivation + description

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

        final_messages_path = os.path.join(os.path.dirname(__file__), ".", "prompts/output_LLM/messages.json")
        with open(final_messages_path, 'w') as file:
            json.dump(self.final_messages, file)

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

    def call_DE(self, env, agents, ego, query, t):

        plot_frame_for_describer(env['env number'], env, agents, ego, t)
        image_path = os.path.join(os.path.dirname(__file__), ".", f"prompts/output_LLM/frames/frame_{t}.png")
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        general_prompt = general_DE_prompt()
        prompt, objects, name_agents = specific_DE_prompt(ego, agents, query, 2, env, t)
        user_input = general_prompt + prompt

        self.DE_messages.append({
            "role": "user",
            "content": [
                {'type': 'text', 'text': user_input},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ],
        })

        DE_message = [
            {
                "role": "user",
                "content": [
                    {'type': 'text', 'text': user_input},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ],
            }
        ]

        self.DE = self.chat_DE(DE_message)
        # Remove all \ and \n
        self.DE_messages.append({
            "role": "assistant",
            "content": self.DE,
        })

        # Save the output
        save_DE_path = os.path.join(os.path.dirname(__file__), ".", "prompts/output_LLM/DE_output.json")
        with open(save_DE_path, 'w') as file:
            json.dump(self.DE_messages, file)

        output = """
""" + objects + """
""" + name_agents + """
Description: """ + self.DE + """
Query: """ + query

        return output

    def recall_DE(self, env, agents, ego, query, t):

        plot_frame_for_describer(env['env number'], env, agents, ego, t)
        image_path = os.path.join(os.path.dirname(__file__), ".", f"prompts/output_LLM/frames/frame_{t}.png")
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        user_input, objects, name_agents = specific_DE_prompt(ego, agents, query, 2, env, t)
        intro = """The driver need a new a new description base on this current information.
        """
        user_input = intro + prompt

        self.DE_messages.append({
                "role": "user",
                "content": [
                    {'type': 'text', 'text': user_input},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ],
            })


        self.DE = self.chat_DE(self.DE_messages)
        self.DE_messages.append({
            "role": "assistant",
            "content": self.DE,
        })

        # Save the output
        save_DE_path = os.path.join(os.path.dirname(__file__), ".", "prompts/output_LLM/DE_output.json")
        with open(save_DE_path, 'w') as file:
            json.dump(self.DE_messages, file)

        output = """
    """ + objects + """
    """ + name_agents + """
Description: """ + self.DE + """
Query: """ + query

        return output

    def call_LLM(self, env, query, agents, ego, t):

        """if self.DE_active:
            DE_description = self.call_DE(env, agents, ego, query, t)
            user_input = TP_prompt_with_DE(DE_description)
        else:
            prompt = general_TP_prompt()
            description = specific_TP_prompt(env, agents, ego, query)
            user_input = prompt + description"""

        prompt = LLM_conditioned_MPC_general()
        description = LLM_conditioned_MPC_specific(env, agents, ego, query)
        user_input = prompt + description

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
        self.final_messages.append({'LLM': self.TP,
                                    'time': t})

        # Save the output
        save_TP_path = os.path.join(os.path.dirname(__file__), ".", "prompts/output_LLM_conditioned_MPC/LLM_output.json")
        with open(save_TP_path, 'w') as file:
            json.dump(self.TP_messages, file)

        final_messages_path = os.path.join(os.path.dirname(__file__), ".", "prompts/output_LLM_conditioned_MPC/messages.json")
        with open(final_messages_path, 'w') as file:
            json.dump(self.final_messages, file)

    def recall_LLM(self, env, query, agents, ego, why, t):

        if why['next_task']:
            motivation = """You have to replan because the action '""" + self.TP['action chosen'][0] + """' is finished. In the next the description of the actual situation.
        """
        elif why['other_agent_too_near'] or  why['MPC_LLM_not_solved']:
            motivation = """The execution of the action '""" + self.TP['action chosen'][0] + """' is not completed, but you have to replan to ensure that the action can be continued. In the next the description of the actual situation.
        """

        """if self.DE_active:
            DE_description = self.call_DE(env, agents, ego, query, t)
            user_input = motivation + DE_description
        else:
            description = specific_TP_prompt(env, agents, ego, query)
            user_input = motivation + description"""

        description = LLM_conditioned_MPC_specific(env, agents, ego, query)
        user_input = motivation + description

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
        elif why['MPC_LLM_not_solved']:
            motivation = """Replan because MPC LLM no success with solver. Task: """

        self.final_messages.append({'User': motivation + query,
                                    'time': t})
        self.final_messages.append({'LLM': self.TP,
                                    'time': t})

        self.task_status = 0

        # Save the output
        save_TP_path = os.path.join(os.path.dirname(__file__), ".", "prompts/output_LLM_conditioned_MPC/LLM_output.json")
        with open(save_TP_path, 'w') as file:
            json.dump(self.TP_messages, file)

        final_messages_path = os.path.join(os.path.dirname(__file__), ".", "prompts/output_LLM_conditioned_MPC/messages.json")
        with open(final_messages_path, 'w') as file:
            json.dump(self.final_messages, file)

    def call_LLM_coder(self, env, query, agents, ego, t):

        """if self.DE_active:
            DE_description = self.call_DE(env, agents, ego, query, t)
            user_input = TP_prompt_with_DE(DE_description)
        else:
            prompt = general_TP_prompt()
            description = specific_TP_prompt(env, agents, ego, query)
            user_input = prompt + description"""

        prompt = prompt_LLM_coder_general()
        description = prompt_LLM_coder_specific(env, agents, ego, query)
        user_input = prompt + description
        self.LLM_correction = prompt_LLM_correction_general() + """Description at the moment of the planning:""" + description

        self.LLM_coder_messages = [
            {
                "role": "user",
                "content": user_input,
            }
        ]
        self.LLM_coder = self.chat(self.LLM_coder_messages)
        # Remove all \ and \n
        self.LLM_coder = json.loads(self.LLM_coder)
        self.LLM_coder_messages.append({
            "role": "assistant",
            "content": str(self.LLM_coder),
        })

        self.final_messages.append({'User': query,
                                    'time': t})
        self.final_messages.append({'LLM coder': self.LLM_coder,
                                    'time': t})

        self.LLM_correction = self.LLM_correction + """
        
Resulting plan where task_1 have failed:
""" + str(self.LLM_coder)

        # Save the output
        save_TP_path = os.path.join(os.path.dirname(__file__), ".", "prompts/output_LLM_based_safe_control/LLM_coder_output.json")
        with open(save_TP_path, 'w') as file:
            json.dump(self.LLM_coder_messages, file)

        final_messages_path = os.path.join(os.path.dirname(__file__), ".", "prompts/output_LLM_based_safe_control/messages.json")
        with open(final_messages_path, 'w') as file:
            json.dump(self.final_messages, file)

    def recall_LLM_coder(self, env, query, agents, ego, why, t):
        description = prompt_LLM_coder_specific(env, agents, ego, query)

        if why['next_task']:
            motivation = """You have to replan because the first task of your last plan is successfully finished, but you want to make sure the plan still good for the actual situation. In the next the description of the actual situation.
        """
            self.LLM_correction = prompt_LLM_correction_general() + """

Description at the moment of the planning:

""" + description

        elif why['other_agent_too_near'] or  why['MPC_LLM_not_solved']:
            LLM_correction_output = self.call_LLM_correction(env, query, agents, ego, t)
            motivation = """You have probably to replan for the following reason:
""" + LLM_correction_output["reason"][0] + """
A possible solution is the following:
""" + LLM_correction_output["solution"][0] + """
Try to replan based on the following description of the actual situation.
"""

        """if self.DE_active:
            DE_description = self.call_DE(env, agents, ego, query, t)
            user_input = motivation + DE_description
        else:
            description = specific_TP_prompt(env, agents, ego, query)
            user_input = motivation + description"""

        user_input = motivation + description

        self.LLM_coder_messages.append({
            "role": "user",
            "content": user_input,
        })

        self.LLM_coder = self.chat(self.LLM_coder_messages)
        # Remove all \ and \n
        self.LLM_coder = json.loads(self.LLM_coder)
        self.LLM_coder_messages.append({
            "role": "assistant",
            "content": str(self.LLM_coder),
        })

        if why['next_task']:
            motivation = """Replan because task finished. Task: """
        elif why['other_agent_too_near']:
            motivation = """Replan because other agents is too near. Task: """
        elif why['MPC_LLM_not_solved']:
            motivation = """Replan because MPC LLM no success with solver. Task: """

        self.final_messages.append({'User': motivation + query,
                                    'time': t})
        self.final_messages.append({'LLM': self.LLM_coder,
                                    'time': t})

        self.task_status = 0

        # Save the output
        save_TP_path = os.path.join(os.path.dirname(__file__), ".", "prompts/output_LLM_based_safe_control/LLM_coder_output.json")
        with open(save_TP_path, 'w') as file:
            json.dump(self.LLM_coder_messages, file)

        final_messages_path = os.path.join(os.path.dirname(__file__), ".", "prompts/output_LLM_based_safe_control/messages.json")
        with open(final_messages_path, 'w') as file:
            json.dump(self.final_messages, file)

        self.LLM_correction = prompt_LLM_correction_general() + """
Description at the moment of the planning:

""" + description + """
Resulting plan:
""" + str(self.LLM_coder)

    def call_LLM_correction(self, env, query, agents, ego, t):

        """if self.DE_active:
            DE_description = self.call_DE(env, agents, ego, query, t)
            user_input = TP_prompt_with_DE(DE_description)
        else:
            prompt = general_TP_prompt()
            description = specific_TP_prompt(env, agents, ego, query)
            user_input = prompt + description"""

        description = prompt_LLM_coder_specific(env, agents, ego, query)
        user_input = self.LLM_correction + """
Description of the actual situation:
""" + description

        self.LLM_correction_messages.append({
                "role": "user",
                "content": user_input,
            })

        input = [
            {
                "role": "user",
                "content": user_input,
            }
        ]

        LLM_correction_output = self.chat(input)
        # Remove all \ and \n
        LLM_correction_output = json.loads(LLM_correction_output)
        self.LLM_correction_messages.append({
            "role": "assistant",
            "content": str(LLM_correction_output),
        })

        self.final_messages.append({'User': query,
                                    'time': t})
        self.final_messages.append({'LLM correction': LLM_correction_output,
                                    'time': t})

        self.LLM_correction = prompt_LLM_correction_general() + description

        # Save the output
        save_TP_path = os.path.join(os.path.dirname(__file__), ".", "prompts/output_LLM_based_safe_control/LLM_correction_output.json")
        with open(save_TP_path, 'w') as file:
            json.dump(self.LLM_correction_messages, file)

        final_messages_path = os.path.join(os.path.dirname(__file__), ".", "prompts/output_LLM_based_safe_control/messages.json")
        with open(final_messages_path, 'w') as file:
            json.dump(self.final_messages, file)

        return LLM_correction_output

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

    def chat_DE(self, message):

        api_key_path = os.path.join(os.path.dirname(__file__), ".", "keys/api_key.txt")
        with open(api_key_path, 'r') as file:
            api_key = file.read()

        client = OpenAI(
            # This is the default and can be omitted
            api_key=api_key,
        )

        chat_completion = client.chat.completions.create(
            messages=message,
            model="gpt-3.5-turbo-0125f",  # gpt-4-turbo, gpt-3.5-turbo-0125, gpt-4o
            response_format={"type": "text"},
        )

        return chat_completion.choices[0].message.content
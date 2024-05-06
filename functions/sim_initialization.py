import numpy as np
import random
import os
import json
from vehicle import Vehicle

def vehicle_init(vehicles, env):

    options_init_state = list(env['Entrances'].keys())
    for i in range(env['Number Vehicles']):
        key_init = random.choice(options_init_state)
        options_init_state.remove(key_init)
        state = np.zeros((4, 1))
        state[0:2, 0] = env['Entrances'][key_init]['position']
        state[0:2, 0] = env['Entrances'][key_init]['position']
        state[2, 0] = env['Entrances'][key_init]['orientation'] * (np.pi / 180)  # form degrees in radiants
        state[3, 0] = random.randint(0, env['Entrances'][key_init]['speed limit'])

        key_target = str(random.choice(env['Entrances'][key_init]['targets']))
        target = np.zeros((4, 1))
        target[0:2, 0] = env['Exits'][key_target]['position']
        target[2, 0] = env['Exits'][key_target]['orientation'] * (np.pi / 180)  # form degrees in radiants
        target[3, 0] = 0

        # questo dipende troppo da un specifico enviroments!
        if np.linalg.norm(target[0:2, 0] - state[0:2, 0]) <= 4:
            while np.linalg.norm(target[0:2, 0] - state[0:2, 0]) <= 4:
                key_target = random.choice(list(env['Exits'].keys()))
                target[0:2, 0] = env['Exits'][key_target]['position']
                target[2, 0] = env['Exits'][key_target]['orientation'] * (np.pi / 180)  # form degrees in radiants
                target[3, 0] = random.randint(0, env['Exits'][key_target]['speed limit'])

        vehicles[f'{i}'].init_state(state, target)
        vehicles[f'{i}'].init_system_constraints(env["State space"], env['Entrances'][key_init]['speed limit'])

    return vehicles

def results_init(env, agents):
    results = {}
    for id_vehicle in range(len(agents)):
        results[f'agent {id_vehicle}'] = {}
        results[f'agent {id_vehicle}']['type'] = agents[f'{id_vehicle}'].type
        """results[f'agent {id_vehicle}']['x coord'] = [float(agents[f'{id_vehicle}'].x)]
        results[f'agent {id_vehicle}']['y coord'] = [float(agents[f'{id_vehicle}'].y)]
        results[f'agent {id_vehicle}']['velocity'] = [float(agents[f'{id_vehicle}'].velocity)]
        results[f'agent {id_vehicle}']['theta'] = [float(agents[f'{id_vehicle}'].theta)]"""

        results[f'agent {id_vehicle}']['x coord'] = []
        results[f'agent {id_vehicle}']['y coord'] = []
        results[f'agent {id_vehicle}']['velocity'] = []
        results[f'agent {id_vehicle}']['theta'] = []

    results_path = os.path.join(os.path.dirname(__file__), ".", "../save_results/results.txt")
    env_path = os.path.join(os.path.dirname(__file__), ".", "../save_results/env.txt")
    with open(results_path, 'w') as file:
        json.dump(results, file)
    with open(env_path, 'w') as file:
        json.dump(env, file)

    return results

def results_update_and_save(env, agent, id_agent, results):

    results[f'agent {id_agent}']['x coord'].append(float(agent.x))
    results[f'agent {id_agent}']['y coord'].append(float(agent.y))
    results[f'agent {id_agent}']['velocity'].append(float(agent.velocity))
    results[f'agent {id_agent}']['theta'].append(float(agent.theta))

    results_path = os.path.join(os.path.dirname(__file__), ".","../save_results/results.txt")
    env_path = os.path.join(os.path.dirname(__file__), ".", "../save_results/env.txt")
    with open(results_path, 'w') as file:
        json.dump(results, file)
    with open(env_path, 'w') as file:
        json.dump(env, file)

    return results


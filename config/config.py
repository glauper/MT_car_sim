import numpy as np
import os
import json

def SimulationConfig():

    SimulationParam = {'Save middle results': True,
                       'With LLM car': False,
                       'Timestep': 0.5, # [s]
                       'Environment': 0,
                       'Query': 'go to the exit on the right',
                       'Controller': {
                           'Agents': {
                               'Type': "tracking MPC",
                               'Horizon': 20
                           },
                           'Ego': {
                               'Active': True,
                               'Type': "safety filter",
                               'Horizon': 20
                           }
                       }
                       }

    return SimulationParam

def EnviromentConfig(environment_type):
    path_file = os.path.join(os.path.dirname(__file__), "environments/env_"+str(environment_type)+".json")
    with open(path_file, 'r') as file:
        env = json.load(file)

    env = env['env']

    circular_obstacles = {}
    for id in env['Road Limits']:
        circular_obstacles[id] = {'center': [], 'M': []}
        circular_obstacles[id]['center'] = np.array(env['Road Limits'][id]['center']).reshape((2,1))
        circular_obstacles[id]['M'] = np.zeros((2,2))
        if len(env['Road Limits'][id]['radius']) == 2:
            circular_obstacles[id]['M'][0,0] = 1/env['Road Limits'][id]['radius'][0]**2
            circular_obstacles[id]['M'][1,1] = 1/env['Road Limits'][id]['radius'][1]**2
            circular_obstacles[id]['r_x'] = env['Road Limits'][id]['radius'][0]
            circular_obstacles[id]['r_y'] = env['Road Limits'][id]['radius'][1]
        elif len(env['Road Limits'][id]['radius']) == 1:
            circular_obstacles[id]['M'][0, 0] = 1 / env['Road Limits'][id]['radius'][0] ** 2
            circular_obstacles[id]['M'][1, 1] = 1 / env['Road Limits'][id]['radius'][0] ** 2
            circular_obstacles[id]['r_x'] = env['Road Limits'][id]['radius'][0]
            circular_obstacles[id]['r_y'] = env['Road Limits'][id]['radius'][0]

    return env, circular_obstacles

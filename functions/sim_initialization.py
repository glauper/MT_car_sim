import numpy as np
import random
import os
import json
from vehicle import Vehicle

def agents_init(env, delta_t, SimulationParam):
    agents = {}
    options_init_state = list(env['Entrances'].keys())
    nr_type_vehicles = len(env['Vehicle Specification']['types'])

    for i in range(env['Number Vehicles']):
        # put a vehicle for each type different from emergency_car
        if nr_type_vehicles > 1 and i < nr_type_vehicles - 1:
            type = env['Vehicle Specification']['types'][i + 1]
        else:
            type = env['Vehicle Specification']['types'][0]
        info_vehicle = env['Vehicle Specification'][type]
        agents[f'{i}'] = Vehicle(type, info_vehicle, delta_t)
        key_init = random.choice(options_init_state)
        agents[f'{i}'].init_state(env, key_init)
        agents[f'{i}'].init_system_constraints(env["State space"], env['Entrances'][key_init]['speed limit'])
        options_init_state.remove(key_init)
        agents[f'{i}'].init_trackingMPC(SimulationParam['Controller']['Horizon'])
        if SimulationParam['Environment'] == 5:
            agents[f'{i}'] = env_5_init(agents[f'{i}'])

    return agents

def env_5_init(agent):
    same_angle = agent.theta
    shift_angle_90 = agent.theta + np.pi / 2
    if shift_angle_90 > np.pi:
        shift_angle_90 = shift_angle_90 - 2 * np.pi
    shift_angle_180 = agent.theta + np.pi
    if shift_angle_180 > np.pi:
        shift_angle_180 = shift_angle_180 - 2 * np.pi
    target_angle = agent.waypoints_exiting[-1][2]
    if same_angle == target_angle or shift_angle_90 == target_angle or shift_angle_180 == target_angle:
        new = np.zeros((4, 1))
        if agent.theta == 0:
            new[0] = 3
            new[1] = -3
        elif agent.theta == np.pi / 2:
            new[0] = 3
            new[1] = 3
        elif agent.theta == np.pi:
            new[0] = -3
            new[1] = 3
        elif agent.theta == -np.pi / 2:
            new[0] = -3
            new[1] = -3
        new[2] = agent.theta
        new[3] = 0
        agent.waypoints_exiting.insert(0, new)
    if shift_angle_180 == target_angle:
        new = np.zeros((4, 1))
        if agent.theta == 0:
            new[0] = 3
            new[1] = 3
        elif agent.theta == np.pi / 2:
            new[0] = -3
            new[1] = 3
        elif agent.theta == np.pi:
            new[0] = -3
            new[1] = -3
        elif agent.theta == -np.pi / 2:
            new[0] = 3
            new[1] = -3
        new[2] = agent.theta + np.pi / 2
        if new[2] > np.pi:
            new[2] = new[2] - 2 * np.pi
        new[3] = 0
        agent.waypoints_exiting.insert(1, new)

    return agent

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

def normalize_angle(angle):
    normalized_angle = (angle + np.pi) % (2 * np.pi)
    if normalized_angle > np.pi:
        normalized_angle -= 2 * np.pi
    return normalized_angle

def collect_info_for_prompts(env, agents, ego, priority, query):
    info = []
    if env['env number'] == 0:
        info.append(str(ego.velocity[0]))
        info.append(str(env['Ego Entrance']['speed limit']))
        if ego.entering:
            info.append(str(np.linalg.norm(ego.position - ego.entry['state'][0:2])))
        elif ego.inside_cross and ego.exiting:
            if 'right' in query:
                info.append(str(np.linalg.norm(ego.position - ego.right['state'][0:2])))
            elif 'left' in query:
                info.append(str(np.linalg.norm(ego.position - ego.left['state'][0:2])))
            elif 'straight' in query:
                info.append(str(np.linalg.norm(ego.position - ego.straight['state'][0:2])))
        elif ego.inside_cross == False and ego.exiting:
            info.append(str(np.linalg.norm(ego.position - ego.final_target['state'][0:2])))
        for id_agent in agents:
            info.append(str(id_agent))
            info.append(agents[id_agent].type)
            if agents[id_agent].entering:
                if agents[id_agent].target[2] == 0:
                    info.append('coming from your left')
                elif agents[id_agent].target[2] == np.pi:
                    info.append('coming from your right')
                elif agents[id_agent].target[2] == -np.pi/2:
                    info.append('coming from the opposite direction to you in his lane')
            elif agents[id_agent].exiting and all(priority.A_p @ agents[id_agent].position <= priority.b_p):
                info.append('is moving through the cross towards its exit')
            else:
                info.append('is moving out of the cross towards its target')
            info.append(str(np.linalg.norm(ego.position - agents[id_agent].position)))
            info.append(str(agents[id_agent].velocity[0]))

    return info


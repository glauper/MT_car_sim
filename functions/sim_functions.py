import numpy as np
from shapely.geometry import Polygon
import random
import os
import json
from vehicle import Vehicle
import pickle
from config.config import SimulationConfig, EnviromentConfig
from priority_controller import PriorityController
from vehicle import Vehicle
from llm import LLM

def sim_init():
    SimulationParam = SimulationConfig()
    delta_t = SimulationParam['Timestep']
    env, circular_obstacles = EnviromentConfig(SimulationParam['Environment'])
    env['env number'] = SimulationParam['Environment']
    env['With LLM car'] = SimulationParam['With LLM car']

    agents = agents_init(env, delta_t, SimulationParam)

    presence_emergency_car = False
    distance = []
    for name_vehicle in agents:
        if agents[name_vehicle].type == 'emergency car':
            presence_emergency_car = True
            name_emergency_car = name_vehicle
        else:
            distance.append(np.linalg.norm(agents[name_vehicle].position))

    order_optimization = list(agents.keys())
    if presence_emergency_car:
        order_optimization.remove(name_emergency_car)
    pairs = list(zip(order_optimization, distance))
    sorted_pairs = sorted(pairs, key=lambda x: x[1])
    order_optimization = [pair[0] for pair in sorted_pairs]
    if presence_emergency_car:
        order_optimization.insert(0, name_emergency_car)

    if SimulationParam['With LLM car']:
        type = env['Vehicle Specification']['types'][0]
        info_vehicle = env['Vehicle Specification'][type]
        ego_vehicle = Vehicle(type, info_vehicle, delta_t)
        ego_vehicle.init_system_constraints(env["State space"], env['Ego Entrance']['speed limit'])
        ego_vehicle.init_state_for_LLM(env, SimulationParam['Query'], SimulationParam['Controller']['Ego']['LLM']['Horizon'], SimulationParam['Controller']['Ego']['SF']['Horizon'])
    else:
        ego_vehicle = []

    priority = PriorityController(SimulationParam['Controller']['Agents']['Type'], SimulationParam['Environment'], env)

    if SimulationParam['With LLM car']:
        Language_Module = LLM()
        Language_Module.call_TP(env, SimulationParam['Query'], agents, ego_vehicle, 0)

        agents[str(len(agents))] = ego_vehicle
        results = results_init(env, agents)
        agents.pop(str(len(agents) - 1))
        options_entrance = list(env['Entrances'].keys())
    else:
        results = results_init(env, agents)
        options_entrance = list(env['Entrances'].keys())

    path = os.path.join(os.path.dirname(__file__), "..")
    print(path)
    # Save some info to eventually reload the simulation
    with open(path + '/reload_sim/SimulationParam.pkl', 'wb') as file:
        pickle.dump(SimulationParam, file)
    with open(path + '/reload_sim/agents.pkl', 'wb') as file:
        pickle.dump(agents, file)
    with open(path + '/reload_sim/ego_vehicle.pkl', 'wb') as file:
        pickle.dump(ego_vehicle, file)
    if SimulationParam['With LLM car']:
        with open(path + '/reload_sim/Language_Module.pkl', 'wb') as file:
            pickle.dump(Language_Module, file)

    if env['With LLM car']:
        return SimulationParam, env, agents, ego_vehicle, Language_Module, presence_emergency_car, order_optimization, priority, results, circular_obstacles
    else:
        return SimulationParam, env, agents, ego_vehicle, [], presence_emergency_car, order_optimization, priority, results, circular_obstacles

def sim_reload():

    path = os.path.join(os.path.dirname(__file__), "..")
    with open(path + '/reload_sim/SimulationParam.pkl', 'rb') as file:
        SimulationParam = pickle.load(file)
    with open(path + '/reload_sim/agents.pkl', 'rb') as file:
        agents = pickle.load(file)
    with open(path + '/reload_sim/ego_vehicle.pkl', 'rb') as file:
        ego_vehicle = pickle.load(file)

    if SimulationParam['With LLM car']:
        with open(path + '/reload_sim/Language_Module.pkl', 'rb') as file:
            Language_Module = pickle.load(file)

        # If you want to change something, like SF active or not
        SimulationParam['Controller']['Ego']['SF']['Active'] = True

    delta_t = SimulationParam['Timestep']
    env, circular_obstacles = EnviromentConfig(SimulationParam['Environment'])
    env['env number'] = SimulationParam['Environment']
    env['With LLM car'] = SimulationParam['With LLM car']

    presence_emergency_car = False
    distance = []
    for name_vehicle in agents:
        if agents[name_vehicle].type == 'emergency_car':
            presence_emergency_car = True
            name_emergency_car = name_vehicle
        else:
            distance.append(np.linalg.norm(agents[name_vehicle].position))

    order_optimization = list(agents.keys())
    if presence_emergency_car:
        order_optimization.remove(name_emergency_car)
    pairs = list(zip(order_optimization, distance))
    sorted_pairs = sorted(pairs, key=lambda x: x[1])
    order_optimization = [pair[0] for pair in sorted_pairs]
    if presence_emergency_car:
        order_optimization.insert(0, name_emergency_car)

    priority = PriorityController(SimulationParam['Controller']['Agents']['Type'], SimulationParam['Environment'], env)

    if SimulationParam['With LLM car']:
        agents[str(len(agents))] = ego_vehicle
        results = results_init(env, agents)
        agents.pop(str(len(agents) - 1))
        options_entrance = list(env['Entrances'].keys())
    else:
        results = results_init(env, agents)
        options_entrance = list(env['Entrances'].keys())

    if env['With LLM car']:
        return SimulationParam, env, agents, ego_vehicle, Language_Module, presence_emergency_car, order_optimization, priority, results, circular_obstacles
    else:
        return SimulationParam, env, agents, ego_vehicle, [], presence_emergency_car, order_optimization, priority, results, circular_obstacles


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
        agents[f'{i}'].update_velocity_limits(env)
        options_init_state.remove(key_init)
        agents[f'{i}'].init_trackingMPC(SimulationParam['Controller']['Agents']['Horizon'])
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

        results[f'agent {id_vehicle}']['x coord'] = []
        results[f'agent {id_vehicle}']['y coord'] = []
        results[f'agent {id_vehicle}']['velocity'] = []
        results[f'agent {id_vehicle}']['theta'] = []

        results[f'agent {id_vehicle}']['x coord pred'] = []
        results[f'agent {id_vehicle}']['y coord pred'] = []
        if agents[f'{id_vehicle}'].LLM_car:
            results[f'agent {id_vehicle}']['x coord pred SF'] = []
            results[f'agent {id_vehicle}']['y coord pred SF'] = []
            results[f'agent {id_vehicle}']['acc pred SF'] = []
            results[f'agent {id_vehicle}']['acc pred LLM'] = []
            results[f'agent {id_vehicle}']['steering angle pred SF'] = []
            results[f'agent {id_vehicle}']['steering angle pred LLM'] = []
        else:
            agents[f'{id_vehicle}'].trajecotry_estimation()
            results[f'agent {id_vehicle}']['trajectory estimation x'] = []
            results[f'agent {id_vehicle}']['trajectory estimation y'] = []

    results_path = os.path.join(os.path.dirname(__file__), ".", "../save_results/results.txt")
    env_path = os.path.join(os.path.dirname(__file__), ".", "../save_results/env.txt")
    with open(results_path, 'w') as file:
        json.dump(results, file)
    with open(env_path, 'w') as file:
        json.dump(env, file)

    return results

def results_update_and_save(env, agents, results, ego_brake):

    for id_vehicle, name_vehicle in enumerate(agents):
        results[f'agent {id_vehicle}']['x coord'].append(float(agents[name_vehicle].x))
        results[f'agent {id_vehicle}']['y coord'].append(float(agents[name_vehicle].y))
        results[f'agent {id_vehicle}']['velocity'].append(float(agents[name_vehicle].velocity))
        results[f'agent {id_vehicle}']['theta'].append(float(agents[name_vehicle].theta))

        results[f'agent {id_vehicle}']['x coord pred'].append(list(agents[name_vehicle].previous_opt_sol['X'][0,:]))
        results[f'agent {id_vehicle}']['y coord pred'].append(list(agents[name_vehicle].previous_opt_sol['X'][1,:]))
        if agents[name_vehicle].LLM_car:
            results[f'agent {id_vehicle}']['x coord pred SF'].append(list(agents[name_vehicle].previous_opt_sol_SF['X'][0,:]))
            results[f'agent {id_vehicle}']['y coord pred SF'].append(list(agents[name_vehicle].previous_opt_sol_SF['X'][1,:]))
            if ego_brake:
                results[f'agent {id_vehicle}']['acc pred SF'].append(float(agents[name_vehicle].input_brakes[0]))
                results[f'agent {id_vehicle}']['acc pred LLM'].append(float(agents[name_vehicle].input_brakes[0]))
                results[f'agent {id_vehicle}']['steering angle pred SF'].append(float(agents[name_vehicle].input_brakes[1]))
                results[f'agent {id_vehicle}']['steering angle pred LLM'].append(float(agents[name_vehicle].input_brakes[1]))
            else:
                results[f'agent {id_vehicle}']['acc pred SF'].append(float(agents[name_vehicle].previous_opt_sol_SF['U'][0,0]))
                results[f'agent {id_vehicle}']['acc pred LLM'].append(float(agents[name_vehicle].previous_opt_sol['U'][0,0]))
                results[f'agent {id_vehicle}']['steering angle pred SF'].append(float(agents[name_vehicle].previous_opt_sol_SF['U'][1,0]))
                results[f'agent {id_vehicle}']['steering angle pred LLM'].append(float(agents[name_vehicle].previous_opt_sol['U'][1,0]))
        else:
            results[f'agent {id_vehicle}']['trajectory estimation x'].append(list(agents[name_vehicle].traj_estimation[0]))
            results[f'agent {id_vehicle}']['trajectory estimation y'].append(list(agents[name_vehicle].traj_estimation[1]))

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

def check_proximity(ego, agent):
    too_near = False

    R_agent = np.zeros((2, 2))
    R_agent[0, 0] = np.cos(agent.theta)
    R_agent[0, 1] = -np.sin(agent.theta)
    R_agent[1, 0] = np.sin(agent.theta)
    R_agent[1, 1] = np.cos(agent.theta)

    #V_agent = R_agent @ np.array([[1.5], [0]])
    #diff = R_agent @ (ego.position - (agent.position + V_agent))
    diff = R_agent @ (ego.position - agent.position)

    if (diff[0] / agent.a_security_dist) ** 2 + (diff[1] / agent.b_security_dist) ** 2 <= 1:
        too_near = True

    return too_near

def check_crash(ego, agent):

    R_agent = np.zeros((2, 2))
    R_agent[0, 0] = np.cos(agent.theta)
    R_agent[0, 1] = -np.sin(agent.theta)
    R_agent[1, 0] = np.sin(agent.theta)
    R_agent[1, 1] = np.cos(agent.theta)

    point_1 = agent.position + R_agent @ np.array([agent.length / 2, -agent.width / 2]).reshape((2,1))
    point_2 = agent.position + R_agent @ np.array([agent.length / 2, agent.width / 2]).reshape((2, 1))
    point_3 = agent.position + R_agent @ np.array([-agent.length / 2, agent.width / 2]).reshape((2, 1))
    point_4 = agent.position + R_agent @ np.array([-agent.length / 2, -agent.width / 2]).reshape((2, 1))

    vertices = [point_1, point_2, point_3, point_4]
    poly_agent = Polygon(vertices)

    R_ego = np.zeros((2, 2))
    R_ego[0, 0] = np.cos(ego.theta)
    R_ego[0, 1] = -np.sin(ego.theta)
    R_ego[1, 0] = np.sin(ego.theta)
    R_ego[1, 1] = np.cos(ego.theta)

    point_1 = ego.position + R_ego @ np.array([ego.length / 2, -ego.width / 2]).reshape((2, 1))
    point_2 = ego.position + R_ego @ np.array([ego.length / 2, ego.width / 2]).reshape((2, 1))
    point_3 = ego.position + R_ego @ np.array([-ego.length / 2, ego.width / 2]).reshape((2, 1))
    point_4 = ego.position + R_ego @ np.array([-ego.length / 2, -ego.width / 2]).reshape((2, 1))

    vertices = [point_1, point_2, point_3, point_4]
    poly_ego = Polygon(vertices)

    crash_status = poly_agent.intersects(poly_ego)

    return crash_status

def check_need_replan(ego_vehicle, agents, Language_Module, env, SimulationParam, run_simulation, next_task, too_near, t):
    # check if the task is finished, i.e. when LLM car is near enough to a waypoint
    print('Cost LLM ', ego_vehicle.previous_opt_sol['Cost'])
    if 'entry' in Language_Module.TP['tasks'][Language_Module.task_status]:
        ego_vehicle.inside_cross = False
        if ego_vehicle.entering == False:
            ego_vehicle.entering = True
            ego_vehicle.exiting = False
        if np.linalg.norm(ego_vehicle.position - ego_vehicle.entry['position']) <= 1:
            next_task = True
    elif 'exit' in Language_Module.TP['tasks'][Language_Module.task_status]:
        ego_vehicle.inside_cross = True
        if ego_vehicle.exiting == False:
            ego_vehicle.entering = False
            ego_vehicle.exiting = True
            ego_vehicle.target = ego_vehicle.waypoints_exiting.pop(0)
        if np.linalg.norm(ego_vehicle.position - ego_vehicle.exit['position']) <= 1:
            next_task = True
    elif 'final_target' in Language_Module.TP['tasks'][Language_Module.task_status]:
        ego_vehicle.inside_cross = False
        if ego_vehicle.exiting == False:
            ego_vehicle.entering = False
            ego_vehicle.exiting = True
        if np.linalg.norm(ego_vehicle.position - ego_vehicle.final_target['position']) <= 1:
            next_task = True
            print('End simulation: because the position of LLM car is near enough to the the final target.')
            run_simulation = False

    if run_simulation:
        if next_task:
            print('Call TP: because a task is terminated and a new one begins.')
            ego_vehicle.t_subtask = 0
            Language_Module.task_status += 1
            reason = {'next_task': True, 'SF_kicks_in': False, 'other_agent_too_near': False,
                      'MPC_LLM_not_solved': False}
            Language_Module.recall_TP(env, SimulationParam['Query'], agents, ego_vehicle, reason, t)
        elif SimulationParam['Controller']['Ego']['SF']['Replan']['Active'] and ego_vehicle.previous_opt_sol_SF[
            'Cost'] >= SimulationParam['Controller']['Ego']['SF']['Replan']['toll']:
            print('Call TP: because SF cost are high')
            ego_vehicle.t_subtask = 0
            reason = {'next_task': False, 'SF_kicks_in': True, 'other_agent_too_near': False,
                      'MPC_LLM_not_solved': False}
            Language_Module.recall_TP(env, SimulationParam['Query'], agents, ego_vehicle, reason, t)
        elif too_near:
            print('Call TP: because an agent is to near')
            ego_vehicle.t_subtask = 0
            reason = {'next_task': False, 'SF_kicks_in': False, 'other_agent_too_near': True,
                      'MPC_LLM_not_solved': False}
            Language_Module.recall_TP(env, SimulationParam['Query'], agents, ego_vehicle, reason, t)
        elif not ego_vehicle.success_solver_MPC_LLM:
            print(ego_vehicle.state)
            print('Call TP: because no success of MPC LLM.')
            ego_vehicle.t_subtask = 0
            reason = {'next_task': False, 'SF_kicks_in': False, 'other_agent_too_near': False,
                      'MPC_LLM_not_solved': True}
            Language_Module.recall_TP(env, SimulationParam['Query'], agents, ego_vehicle, reason, t)
            ego_vehicle.success_solver_MPC_LLM = True
        elif not ego_vehicle.success_solver_SF:
            print(ego_vehicle.state)
            print('Call TP: because no success of SF.')
            ego_vehicle.t_subtask = 0
            reason = {'next_task': False, 'SF_kicks_in': False, 'other_agent_too_near': False,
                      'MPC_LLM_not_solved': False, 'SF_not_solved': True}
            Language_Module.recall_TP(env, SimulationParam['Query'], agents, ego_vehicle, reason, t)
            ego_vehicle.success_solver_SF = True

    return run_simulation


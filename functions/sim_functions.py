import numpy as np
import time
from shapely.geometry import Polygon
import random
import os
import json
from vehicle import Vehicle
import pickle
from config.config import SimulationConfig, EnviromentConfig
from priority_controller import PriorityController
from vehicle import Vehicle
from pedestrian import Pedestrian
from bicycle import Bicycle
from llm import LLM

def sim_init(counter, type_simulation):
    SimulationParam = SimulationConfig()
    delta_t = SimulationParam['Timestep']
    env, circular_obstacles = EnviromentConfig(SimulationParam['Environment'])
    env['env number'] = SimulationParam['Environment']
    env['With LLM car'] = SimulationParam['With LLM car']

    agents = agents_init(env, delta_t, SimulationParam)

    presence_emergency_car = False
    name_pedestrian = []
    distance = []
    for name_vehicle in agents:
        if agents[name_vehicle].type == 'emergency car':
            presence_emergency_car = True
            name_emergency_car = name_vehicle
        elif agents[name_vehicle].type in env['Pedestrians Specification']['types']:
            name_pedestrian.append(name_vehicle)
        else:
            distance.append(np.linalg.norm(agents[name_vehicle].position))

    order_optimization = list(agents.keys())
    if presence_emergency_car:
        order_optimization.remove(name_emergency_car)
    if len(name_pedestrian) > 0:
        for name_ped in name_pedestrian:
            order_optimization.remove(name_ped)
    pairs = list(zip(order_optimization, distance))
    sorted_pairs = sorted(pairs, key=lambda x: x[1])
    order_optimization = [pair[0] for pair in sorted_pairs]
    if presence_emergency_car:
        order_optimization.append(name_emergency_car)
    if len(name_pedestrian) > 0:
        for name_ped in name_pedestrian:
            order_optimization.insert(0, name_ped)

    if SimulationParam['With LLM car']:
        type = env['Vehicle Specification']['types'][0]
        info_vehicle = env['Vehicle Specification'][type]
        ego_vehicle = Vehicle(type, info_vehicle, delta_t)
        ego_vehicle.init_system_constraints(env["State space"], env['Ego Entrance']['speed limit'])
        ego_vehicle.init_state_for_LLM(env, SimulationParam)
        if not ego_vehicle.use_LLM_output_in_SF:
            ego_vehicle.update_velocity_limits(env)
    else:
        ego_vehicle = []

    priority = PriorityController(SimulationParam['Controller']['Agents']['Type'], SimulationParam['Environment'], env)

    if SimulationParam['With LLM car']:
        if type_simulation == "safe_narrate":
            Language_Module = LLM(SimulationParam['Describer active'])
            if SimulationParam['Controller']['Ego']['LLM']['Reasoning']:
                Language_Module.reasoning_active = True
            start = time.time()
            Language_Module.call_TP(env, SimulationParam['Query'], agents, ego_vehicle, 0)
            end = time.time()
            counter['elapsed time for LLM'].append(end-start)
        elif type_simulation == "llm_conditioned_mpc":
            llm_cond_mpc = LLM(SimulationParam['Describer active'])
            start = time.time()
            llm_cond_mpc.call_LLM(env, SimulationParam['Query'], agents, ego_vehicle, 0)
            end = time.time()
            counter['elapsed time for LLM'].append(end - start)
        elif type_simulation == "llm_based_safe_control":
            llm_based_mpc = LLM(SimulationParam['Describer active'])
            start = time.time()
            llm_based_mpc.call_LLM_coder(env, SimulationParam['Query'], agents, ego_vehicle, 0)
            end = time.time()
            counter['elapsed time for LLM'].append(end - start)

        agents[str(len(agents))] = ego_vehicle
        results = results_init(env, agents)
        agents.pop(str(len(agents) - 1))
        options_entrance = list(env['Entrances'].keys())
    else:
        results = results_init(env, agents)
        options_entrance = list(env['Entrances'].keys())

    if type_simulation == "safe_narrate":
        path = os.path.join(os.path.dirname(__file__), "..")
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
        if type_simulation == "safe_narrate":
            return SimulationParam, env, agents, ego_vehicle, Language_Module, presence_emergency_car, order_optimization, priority, results, circular_obstacles, counter
        elif type_simulation == "llm_conditioned_mpc":
            return SimulationParam, env, agents, ego_vehicle, llm_cond_mpc, presence_emergency_car, order_optimization, priority, results, circular_obstacles, counter
        elif type_simulation == "llm_based_safe_control":
            return SimulationParam, env, agents, ego_vehicle, llm_based_mpc, presence_emergency_car, order_optimization, priority, results, circular_obstacles, counter

    else:
        return SimulationParam, env, agents, ego_vehicle, [], presence_emergency_car, order_optimization, priority, results, circular_obstacles, counter

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
    nr_emergency_vehicles = env['Number Emergency Vehicles']
    count_emergency_vehicles = 0

    for i in range(env['Number Vehicles']):
        # put a vehicle for each type different from emergency_car
        if count_emergency_vehicles < nr_emergency_vehicles:
            type = env['Vehicle Specification']['types'][1]
            count_emergency_vehicles += 1
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

    nr_vehicles = len(agents)
    for i in range(env['Number Pedestrians']):
        if i % 2 == 0:
            type = 'adult'
            info_adult = env['Pedestrians Specification']['adult']
            agents[f'{i + nr_vehicles}'] = Pedestrian(type, info_adult, delta_t)
            agents[f'{i + nr_vehicles}'].init(env['State space'],
                                              env['Pedestrians Specification']['adult']['velocity limits'],
                                              SimulationParam['Controller']['Agents']['Horizon'])
            agents[f'{i + nr_vehicles}'].trajecotry_estimation()
        else:
            type = 'children'
            info_children = env['Pedestrians Specification']['children']
            agents[f'{i + nr_vehicles}'] = Pedestrian(type, info_children, delta_t)
            agents[f'{i + nr_vehicles}'].init(env['State space'],
                                              env['Pedestrians Specification']['children']['velocity limits'],
                                              SimulationParam['Controller']['Agents']['Horizon'])
            agents[f'{i + nr_vehicles}'].trajecotry_estimation()

    nr_agents = len(agents)
    for i in range(env['Number Bicycles']):
        info_bicycle = env['Bicycle Specification']['bicycle']
        type = 'bicycle'
        agents[f'{i + nr_agents}'] = Bicycle(type, info_bicycle, delta_t)
        key_init = random.choice(options_init_state)
        agents[f'{i + nr_agents}'].init_state(env, key_init)
        agents[f'{i + nr_agents}'].init_system_constraints(env["State space"], env['Entrances'][key_init]['speed limit'])
        options_init_state.remove(key_init)
        agents[f'{i + nr_agents}'].init_trackingMPC(SimulationParam['Controller']['Agents']['Horizon'])

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

        if results[f'agent {id_vehicle}']['type'] == 'adult' or results[f'agent {id_vehicle}']['type'] == 'children':
            results[f'agent {id_vehicle}']['x coord'] = []
            results[f'agent {id_vehicle}']['y coord'] = []
            results[f'agent {id_vehicle}']['v_x coord'] = []
            results[f'agent {id_vehicle}']['v_y coord'] = []
            results[f'agent {id_vehicle}']['trajectory estimation x'] = []
            results[f'agent {id_vehicle}']['trajectory estimation y'] = []

        else:
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
        if results[f'agent {id_vehicle}']['type'] == 'adult' or results[f'agent {id_vehicle}']['type'] == 'children':
            results[f'agent {id_vehicle}']['x coord'].append(float(agents[name_vehicle].x))
            results[f'agent {id_vehicle}']['y coord'].append(float(agents[name_vehicle].y))
            results[f'agent {id_vehicle}']['v_x coord'].append(float(agents[name_vehicle].v_x))
            results[f'agent {id_vehicle}']['v_y coord'].append(float(agents[name_vehicle].v_y))
            results[f'agent {id_vehicle}']['trajectory estimation x'].append(list(agents[name_vehicle].traj_estimation[0]))
            results[f'agent {id_vehicle}']['trajectory estimation y'].append(list(agents[name_vehicle].traj_estimation[1]))
        else:
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
    R_agent[0, 0] = np.cos(-agent.theta)
    R_agent[0, 1] = -np.sin(-agent.theta)
    R_agent[1, 0] = np.sin(-agent.theta)
    R_agent[1, 1] = np.cos(-agent.theta)

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

def check_need_replan(ego_vehicle, agents, Language_Module, env, SimulationParam, run_simulation, next_task, too_near, t, counter):
    # check if the task is finished, i.e. when LLM car is near enough to a waypoint
    print('Cost LLM ', ego_vehicle.previous_opt_sol['Cost'])
    if 'entry' in Language_Module.TP['tasks'][Language_Module.task_status]:
        ego_vehicle.target = ego_vehicle.entry['state']
        ego_vehicle.inside_cross = False
        if ego_vehicle.entering == False:
            ego_vehicle.entering = True
            ego_vehicle.exiting = False
        if np.linalg.norm(ego_vehicle.position - ego_vehicle.entry['position']) <= 1:
            next_task = True
    elif 'exit' in Language_Module.TP['tasks'][Language_Module.task_status]:
        ego_vehicle.inside_cross = True
        ego_vehicle.target = ego_vehicle.exit['state']
        if ego_vehicle.exiting == False:
            ego_vehicle.entering = False
            ego_vehicle.exiting = True
            if len(ego_vehicle.waypoints_exiting) != 0:
                ego_vehicle.target = ego_vehicle.waypoints_exiting.pop(0)
        if np.linalg.norm(ego_vehicle.position - ego_vehicle.exit['position']) <= 1:
            next_task = True
    elif 'final_target' in Language_Module.TP['tasks'][Language_Module.task_status]:
        ego_vehicle.inside_cross = False
        ego_vehicle.target = ego_vehicle.final_target['state']
        if ego_vehicle.exiting == False:
            ego_vehicle.entering = False
            ego_vehicle.exiting = True
        if np.linalg.norm(ego_vehicle.position - ego_vehicle.final_target['position']) <= 1:
            next_task = True
            print('End simulation: because the position of LLM car is near enough to the the final target.')
            run_simulation = False

    if run_simulation:
        reason = {'next_task': False,
                  'SF_kicks_in': False,
                  'other_agent_too_near': False,
                  'MPC_LLM_not_solved': False,
                  'SF_not_solved': False,
                  'soft_SF_kicks_in': False,
                  'soft_SF_psi_not_solved': False,
                  'soft_SF_not_solved': False}
        if next_task:
            print('Call TP: because a task is terminated and a new one begins.')
            ego_vehicle.t_subtask = 0
            Language_Module.task_status += 1
            reason['next_task'] = True
            start = time.time()
            Language_Module.recall_TP(env, SimulationParam['Query'], agents, ego_vehicle, reason, t)
            end = time.time()
            counter['elapsed time for LLM'].append(end - start)
            counter['TP calls'] += 1
        elif SimulationParam['Controller']['Ego']['SF']['Replan']['Active'] and ego_vehicle.previous_opt_sol_SF[
            'Cost'] >= SimulationParam['Controller']['Ego']['SF']['Replan']['toll']:
            print('Call TP: because SF cost are high')
            ego_vehicle.t_subtask = 0
            reason['SF_kicks_in'] = True
            start = time.time()
            Language_Module.recall_TP(env, SimulationParam['Query'], agents, ego_vehicle, reason, t)
            end = time.time()
            counter['elapsed time for LLM'].append(end - start)
            counter['TP calls'] += 1
        elif too_near:
            print('Call TP: because an agent is to near')
            ego_vehicle.t_subtask = 0
            reason['other_agent_too_near'] = True
            start = time.time()
            Language_Module.recall_TP(env, SimulationParam['Query'], agents, ego_vehicle, reason, t)
            end = time.time()
            counter['elapsed time for LLM'].append(end - start)
            counter['TP calls'] += 1
        elif not ego_vehicle.success_solver_MPC_LLM:
            print('Call TP: because no success of MPC LLM.')
            ego_vehicle.t_subtask = 0
            reason['MPC_LLM_not_solved'] = True
            start = time.time()
            Language_Module.recall_TP(env, SimulationParam['Query'], agents, ego_vehicle, reason, t)
            end = time.time()
            counter['elapsed time for LLM'].append(end - start)
            counter['TP calls'] += 1
            ego_vehicle.success_solver_MPC_LLM = True
        elif SimulationParam['Controller']['Ego']['SF']['Active'] and not ego_vehicle.success_solver_SF:
            ego_vehicle.t_subtask = 0
            if SimulationParam['Controller']['Ego']['SF']['Soft']:
                print('Call TP: because no success for solver soft SF.')
                reason['soft_SF_not_solved'] = True
            else:
                print('Call TP: because no success of SF.')
                reason['SF_not_solved'] = True
            start = time.time()
            Language_Module.recall_TP(env, SimulationParam['Query'], agents, ego_vehicle, reason, t)
            end = time.time()
            counter['elapsed time for LLM'].append(end - start)
            counter['TP calls'] += 1
            ego_vehicle.success_solver_SF = True
        elif SimulationParam['Controller']['Ego']['SF']['Soft']:
            replan_flag = False
            soft_SF_trashold = 1
            for k in range(ego_vehicle.N_SF + 1):
                if np.linalg.norm(ego_vehicle.previous_opt_sol_SF['psi_b_x'][:, k]) > soft_SF_trashold:
                    replan_flag = True
                    counter['slack SF'].append(['time ' + str(t) + ': psi_b_x k = ' + str(k), list(ego_vehicle.previous_opt_sol_SF['psi_b_x'][:, k])])
                #elif np.linalg.norm(ego_vehicle.previous_opt_sol_SF['psi_v_limit'][k]) > soft_SF_trashold:
                #    replan_flag = True
                elif np.linalg.norm(ego_vehicle.previous_opt_sol_SF['psi_agents'][:, k]) > soft_SF_trashold:
                    replan_flag = True
                    counter['slack SF'].append(['time ' + str(t) + ': psi_agents k = ' + str(k), list(ego_vehicle.previous_opt_sol_SF['psi_agents'][:, k])])
                elif np.linalg.norm(ego_vehicle.previous_opt_sol_SF['psi_obst'][:, k]) > soft_SF_trashold:
                    replan_flag = True
                    counter['slack SF'].append(['time ' + str(t) + ': psi_obst k = ' + str(k), list(ego_vehicle.previous_opt_sol_SF['psi_obst'][:, k])])
            if replan_flag == True:
                print('Call TP: because the slack variable of soft SF are higher then treshold.')
                ego_vehicle.t_subtask = 0
                reason['soft_SF_kicks_in'] = True
                start = time.time()
                Language_Module.recall_TP(env, SimulationParam['Query'], agents, ego_vehicle, reason, t)
                end = time.time()
                counter['elapsed time for LLM'].append(end - start)
                counter['TP calls'] += 1
                ego_vehicle.success_solver_SF = True

    return run_simulation, counter


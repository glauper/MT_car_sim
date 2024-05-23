import numpy as np
import re
import pickle
import random
from config.config import SimulationConfig, EnviromentConfig
from functions.plot_functions import plot_simulation
from functions.sim_initialization import results_init, results_update_and_save, agents_init
from env_controller import EnvController
from priority_controller import PriorityController
from vehicle import Vehicle
from llm import LLM

with open('reload_sim/SimulationParam.pkl', 'rb') as file:
    SimulationParam = pickle.load(file)
with open('reload_sim/agents.pkl', 'rb') as file:
    agents = pickle.load(file)
with open('reload_sim/ego_vehicle.pkl', 'rb') as file:
    ego_vehicle = pickle.load(file)

if SimulationParam['With LLM car']:
    with open('reload_sim/Language_Module.pkl', 'rb') as file:
        Language_Module = pickle.load(file)

    #If you want to change something, like SF acive or not
    SimulationParam['Controller']['Ego']['Active'] = True

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
    agents.pop(str(len(agents)-1))
    options_entrance = list(env['Entrances'].keys())
else:
    results = results_init(env, agents)
    options_entrance = list(env['Entrances'].keys())

t = 0
run_simulation = True
while run_simulation:
    print("Simulation time: ", t)
    if SimulationParam['With LLM car']:
        print('Task: ', Language_Module.TP['tasks'][Language_Module.task_status])

    # Save the results
    if SimulationParam['With LLM car']:
        agents[str(len(agents))] = ego_vehicle
        results = results_update_and_save(env, agents, results)
        agents.pop(str(len(agents) - 1))
    else:
        results = results_update_and_save(env, agents, results)

    # This is a controller that optimize the trajectory of one agent at time
    other_agents = {}
    input = {}
    #for id_vehicle, name_vehicle in enumerate(agents):
    for name_vehicle in order_optimization:
        id_vehicle = int(name_vehicle)
        if agents[name_vehicle].entering or agents[name_vehicle].exiting:
            input[f'agent {id_vehicle}'] = agents[name_vehicle].trackingMPC(other_agents, ego_vehicle, circular_obstacles, t)
            other_agents[name_vehicle] = agents[name_vehicle]

    if SimulationParam['With LLM car']:
        ego_brake = False
        next_task = False
        if 'brakes()' in Language_Module.TP['tasks'][Language_Module.task_status]:
            ego_brake = True
            # This such because when it brakes does not call all the time TP base on the old SF cost
            ego_vehicle.previous_opt_sol_SF['Cost'] = 0
            if 'wait' in Language_Module.TP['tasks'][Language_Module.task_status]:
                # ??? Here is necessay or we simply replan until the LLM decid to move on???
                pattern = r'agent (\d+)'
                match = re.search(pattern, Language_Module.TP['tasks'][Language_Module.task_status])
                if match:
                    id_agent = match.group(1)
                    if 'agent ' + id_agent in Language_Module.TP['tasks'][Language_Module.task_status]:
                        if all(priority.A_p @ agents[id_agent].position <= priority.b_p):
                            next_task = False
                        else:
                            next_task = True
            elif abs(ego_vehicle.velocity) <= 0.01:
                next_task = True
        else:
            # Check if necessary to have new Optimization Design
            if len(Language_Module.OD) == 0:
                print('Call OD the first time, for :', Language_Module.TP['tasks'][Language_Module.task_status])
                Language_Module.call_OD(SimulationParam['Environment'], agents)
            elif ego_vehicle.t_subtask == 0:
                print('Call OD for :', Language_Module.TP['tasks'][Language_Module.task_status])
                Language_Module.recall_OD(SimulationParam['Environment'], agents)

            input_ego = ego_vehicle.MPC_LLM(agents, circular_obstacles, t, Language_Module)
            if SimulationParam['Controller']['Ego']['Active']:
                input_ego = ego_vehicle.SF(input_ego, agents, circular_obstacles, t)
                print('Cost SF ', ego_vehicle.previous_opt_sol_SF['Cost'])

            if all(priority.A_p @ ego_vehicle.position <= priority.b_p):
                ego_vehicle.inside_cross = True
            else:
                ego_vehicle.inside_cross = False

            # check if the task is finished with the cost of the optimization problem
            print('Cost LLM ', ego_vehicle.previous_opt_sol['Cost'])
            if 'entry' in Language_Module.TP['tasks'][Language_Module.task_status]:
                if np.linalg.norm(ego_vehicle.position - ego_vehicle.entry['position']) <= 1:
                    next_task = True
            elif 'exit' in Language_Module.TP['tasks'][Language_Module.task_status]:
                if np.linalg.norm(ego_vehicle.position - ego_vehicle.exit['position']) <= 1:
                    next_task = True
            elif 'final_target' in Language_Module.TP['tasks'][Language_Module.task_status]:
                if np.linalg.norm(ego_vehicle.position - ego_vehicle.final_target['position']) <= 1:
                    next_task = True
                    print('End simulation: because the position of LLM car is near enough to the the final target.')
                    run_simulation = False

    if SimulationParam['With LLM car']:
        # Dynamics propagation
        if ego_brake:
            ego_vehicle.brakes()
        else:
            ego_vehicle.dynamics_propagation(input_ego)

    # Dynamics propagation
    if presence_emergency_car:
        # Dynamics propagation if there is an emergency car
        for id_vehicle, name_vehicle in enumerate(agents):
            if agents[name_vehicle].type == 'emergency_car':
                agents[name_vehicle].dynamics_propagation(input[f'agent {id_vehicle}'])
            else:
                agents[name_vehicle].brakes()
    else:
        # Dynamics propagation if there isn't an emergency car
        for id_vehicle, name_vehicle in enumerate(agents):
            if agents[name_vehicle].entering or agents[name_vehicle].exiting:
                agents[name_vehicle].dynamics_propagation(input[f'agent {id_vehicle}'])
            else:
                agents[name_vehicle].brakes()

    t += 1
    if SimulationParam['With LLM car']:
        ego_vehicle.t_subtask += 1
        # I don't think is the best way to do that...
        agents[str(len(agents))] = ego_vehicle
        agents = priority.SwissPriority(agents, order_optimization, SimulationParam['With LLM car'])
        ego_vehicle = agents.pop(str(len(agents)-1))
    else:
        agents = priority.SwissPriority(agents, order_optimization, SimulationParam['With LLM car'])

    # update the velocity limit in the new street for other agents
    for name_agent in agents:
        agents[name_agent].update_velocity_limits(env)

    if SimulationParam['With LLM car']:
        """if next_task:
            ego_vehicle.t_subtask = 0
            Language_Module.task_status += 1

        if Language_Module.task_status >= len(Language_Module.TP['tasks']):
            if np.linalg.norm(ego_vehicle.position - ego_vehicle.final_target['position']) <= 1:
                # Questo si potrebbe spostare a prima
                print('End simulation: because the position of LLM car is near enough to the the final target.')
                run_simulation = False
            elif ego_vehicle.entering == False and ego_vehicle.exiting == False:
                print('End simulation: because entering and exiting flags are false.')
                run_simulation = False
            else:
                print('End simulation: because there are no more tasks in the TP.')
                run_simulation = False"""
        if run_simulation:
            if next_task:
                ego_vehicle.t_subtask = 0
                Language_Module.task_status += 1
                print('Call TP: because a task is terminated and a new one begins.')
                Language_Module.recall_TP(env, SimulationParam['Query'], agents, ego_vehicle, {'next_task': True, 'SF_kicks_in': False})
                # If safety filter have to correct then we need to replan...how to proceed?
            elif SimulationParam['Controller']['Ego']['Active'] and ego_vehicle.previous_opt_sol_SF['Cost'] >= 10:
                ego_vehicle.t_subtask = 0
                print('Call TP: because SF cost are high')
                Language_Module.recall_TP(env, SimulationParam['Query'], agents, ego_vehicle, {'next_task': False, 'SF_kicks_in': True})
    else:
        reach_end_target = []
        for name_agent in agents:
            if agents[name_agent].entering == False and agents[name_agent].exiting == False:
                reach_end_target.append(True)
            else:
                reach_end_target.append(False)
            if agents[name_agent].exiting and agents[name_agent].type == 'emergency_car':
                presence_emergency_car = False
        if all(reach_end_target):
            run_simulation = False

    if t == 200:
        print('End simulation: because max simulation steps are reached.')
        run_simulation = False

# Save the results
if SimulationParam['With LLM car']:
    agents[str(len(agents))] = ego_vehicle

results = results_update_and_save(env, agents, results)

plot_simulation(SimulationParam['Environment'], env, results)

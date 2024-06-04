import numpy as np
import re
import json
import os
import pickle
import random
from config.config import SimulationConfig, EnviromentConfig
from functions.plot_functions import plot_simulation, input_animation
from functions.sim_initialization import results_update_and_save, sim_init, sim_reload
from env_controller import EnvController
from priority_controller import PriorityController
from vehicle import Vehicle
from llm import LLM

"""SimulationParam = SimulationConfig()
delta_t = SimulationParam['Timestep']
env, circular_obstacles = EnviromentConfig(SimulationParam['Environment'])
env['env number'] = SimulationParam['Environment']
env['With LLM car'] = SimulationParam['With LLM car']

agents = agents_init(env, delta_t, SimulationParam)

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

if SimulationParam['With LLM car']:
    type = env['Vehicle Specification']['types'][0]
    info_vehicle = env['Vehicle Specification'][type]
    ego_vehicle = Vehicle(type, info_vehicle, delta_t)
    ego_vehicle.init_system_constraints(env["State space"], env['Ego Entrance']['speed limit'])
    ego_vehicle.init_state_for_LLM(env, SimulationParam['Query'], SimulationParam['Controller']['Ego']['Horizon'])
else:
    ego_vehicle = []

priority = PriorityController(SimulationParam['Controller']['Agents']['Type'], SimulationParam['Environment'], env)

if SimulationParam['With LLM car']:
    Language_Module = LLM()
    Language_Module.call_TP(env, SimulationParam['Query'], agents, ego_vehicle)

    agents[str(len(agents))] = ego_vehicle
    results = results_init(env, agents)
    agents.pop(str(len(agents)-1))
    options_entrance = list(env['Entrances'].keys())
else:
    results = results_init(env, agents)
    options_entrance = list(env['Entrances'].keys())

# Save some info to eventually reload the simulation
with open('reload_sim/SimulationParam.pkl', 'wb') as file:
    pickle.dump(SimulationParam, file)
with open('reload_sim/agents.pkl', 'wb') as file:
    pickle.dump(agents, file)
with open('reload_sim/ego_vehicle.pkl', 'wb') as file:
    pickle.dump(ego_vehicle, file)
if SimulationParam['With LLM car']:
    with open('reload_sim/Language_Module.pkl', 'wb') as file:
        pickle.dump(Language_Module, file)"""

(SimulationParam, env, agents, ego_vehicle, Language_Module, presence_emergency_car, order_optimization, priority,
 results, circular_obstacles) = sim_init()
#(SimulationParam, env, agents, ego_vehicle, Language_Module, presence_emergency_car, order_optimization, priority,
# results, circular_obstacles) = sim_reload()

t = 0
run_simulation = True
ego_brake = False
while run_simulation:
    print("Simulation time: ", t)
    if SimulationParam['With LLM car']:
        print('Task: ', Language_Module.TP['tasks'][Language_Module.task_status])

    # Save the results
    if SimulationParam['With LLM car']:
        agents[str(len(agents))] = ego_vehicle
        results = results_update_and_save(env, agents, results, ego_brake)
        agents.pop(str(len(agents) - 1))
    else:
        results = results_update_and_save(env, agents, results, ego_brake)

    # This is a controller that optimize the trajectory of one agent at time
    other_agents = {}
    input = {}
    too_near = False
    for name_vehicle in order_optimization:
        id_vehicle = int(name_vehicle)
        if agents[name_vehicle].entering or agents[name_vehicle].exiting:
            input[f'agent {id_vehicle}'] = agents[name_vehicle].trackingMPC(other_agents, ego_vehicle, circular_obstacles, t)
            other_agents[name_vehicle] = agents[name_vehicle]
            agents[name_vehicle].trajecotry_estimation()
            if np.linalg.norm(ego_vehicle.position - agents[name_vehicle].position) <= min(agents[name_vehicle].security_dist,ego_vehicle.security_dist):
                print('distance other', np.linalg.norm(ego_vehicle.position - agents[name_vehicle].position))
                if 'brakes()' in Language_Module.TP['tasks'][Language_Module.task_status]:
                    too_near = False
                else:
                    too_near = True

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
                if ego_vehicle.t_subtask > 30:
                    next_task = True
            elif abs(ego_vehicle.velocity) <= 0.01:
                next_task = True
        elif 'wait' in Language_Module.TP['tasks'][Language_Module.task_status]:
            ego_brake = True
            ego_vehicle.previous_opt_sol_SF['Cost'] = 0
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
            if ego_vehicle.t_subtask > 30:
                next_task = True
        elif too_near:
            ego_brake = True
            ego_vehicle.previous_opt_sol_SF['Cost'] = 0
            if abs(ego_vehicle.velocity) <= 0.01:
                next_task = True
        else:
            # Check if necessary to have new Optimization Design
            if len(Language_Module.OD) == 0:
                print('Call OD the first time, for :', Language_Module.TP['tasks'][Language_Module.task_status])
                Language_Module.call_OD(SimulationParam['Environment'], agents, t)
                final_messages_path = os.path.join(os.path.dirname(__file__), ".", "prompts/output_LLM/messages.json")
                with open(final_messages_path, 'w') as file:
                    json.dump(Language_Module.final_messages, file)
            elif ego_vehicle.t_subtask == 0:
                print('Call OD for :', Language_Module.TP['tasks'][Language_Module.task_status])
                Language_Module.recall_OD(SimulationParam['Environment'], agents, t)
                final_messages_path = os.path.join(os.path.dirname(__file__), ".", "prompts/output_LLM/messages.json")
                with open(final_messages_path, 'w') as file:
                    json.dump(Language_Module.final_messages, file)

            input_ego = ego_vehicle.MPC_LLM(agents, circular_obstacles, t, Language_Module)
            if SimulationParam['Controller']['Ego']['SF']['Active']:
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
        #agents = priority.NoPriority(agents, order_optimization, SimulationParam['With LLM car'])
        ego_vehicle = agents.pop(str(len(agents)-1))
    else:
        agents = priority.SwissPriority(agents, order_optimization, SimulationParam['With LLM car'])

    # update the velocity limit in the new street for other agents
    for name_agent in agents:
        agents[name_agent].update_velocity_limits(env)

    if SimulationParam['With LLM car']:
        if run_simulation:
            if next_task:
                print('Call TP: because a task is terminated and a new one begins.')
                ego_vehicle.t_subtask = 0
                Language_Module.task_status += 1
                reason = {'next_task': True, 'SF_kicks_in': False}
                Language_Module.recall_TP(env, SimulationParam['Query'], agents, ego_vehicle, reason, t)
                # If safety filter have to correct then we need to replan...how to proceed?
            elif SimulationParam['Controller']['Ego']['SF']['Replan']['Active'] and ego_vehicle.previous_opt_sol_SF['Cost'] >= SimulationParam['Controller']['Ego']['SF']['Replan']['toll']:
                    print('Call TP: because SF cost are high')
                    ego_vehicle.t_subtask = 0
                    reason = {'next_task': False, 'SF_kicks_in': True}
                    Language_Module.recall_TP(env, SimulationParam['Query'], agents, ego_vehicle, reason, t)
            elif too_near:
                print('Call TP: because an agent is to near')
                ego_vehicle.t_subtask = 0
                reason = {'next_task': False, 'SF_kicks_in': True}
                Language_Module.recall_TP(env, SimulationParam['Query'], agents, ego_vehicle, reason, t)

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
    final_messages_path = os.path.join(os.path.dirname(__file__), ".", "prompts/output_LLM/messages.json")
    with open(final_messages_path, 'w') as file:
        json.dump(Language_Module.final_messages, file)

results = results_update_and_save(env, agents, results, ego_brake)

plot_simulation(SimulationParam['Environment'], env, results, t_start=0, t_end=len(results['agent 0']['x coord']))
input_animation(results, t_start=0, t_end=len(results['agent 0']['x coord']))

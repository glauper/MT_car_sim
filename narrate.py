import numpy as np
import shutil
import time
import re
import json
import os
from functions.plot_functions import plot_simulation, input_animation, save_all_frames
from functions.sim_functions import (results_update_and_save, sim_init, sim_reload, check_proximity, check_crash,
                                          check_need_replan)

path_folder= os.path.join(os.path.dirname(__file__), ".", f"prompts/output_LLM/frames/")
shutil.rmtree(path_folder)
os.makedirs(path_folder)

counter = {'crash': 0,
           'TP calls': 1,
           'OD calls':0,
           'elapsed time for LLM': [],
           'too_near': 0,
           'recall due to CM': 0,
           'slack SF': []}

type_simulation = "safe_narrate"
(SimulationParam, env, agents, ego_vehicle, Language_Module, presence_emergency_car, order_optimization, priority,
 results, circular_obstacles, counter) = sim_init(counter, type_simulation)
#(SimulationParam, env, agents, ego_vehicle, Language_Module, presence_emergency_car, order_optimization, priority,
# results, circular_obstacles) = sim_reload()

t = 0
run_simulation = True
ego_brake = False
while run_simulation:
    print("Simulation time: ", t)
    if SimulationParam['With LLM car']:
        print('Task: ', Language_Module.TP['tasks'][Language_Module.task_status])
        if SimulationParam['Controller']['Ego']['LLM']['Reasoning']:
            print('Reasons: ', Language_Module.TP['reasons'][Language_Module.task_status])

    # Save the results
    if SimulationParam['With LLM car']:
        agents[str(len(agents))] = ego_vehicle
        results = results_update_and_save(env, agents, results, ego_brake)
        agents.pop(str(len(agents) - 1))
    else:
        results = results_update_and_save(env, agents, results, ego_brake)

    # Controller to decide the trajectory of other agents
    other_agents = {}
    input = {}
    #too_near = False
    for name_vehicle in order_optimization:
        id_vehicle = int(name_vehicle)
        if agents[name_vehicle].type in env['Pedestrians Specification']['types']:
            agents[name_vehicle].trajecotry_estimation()
            agents[name_vehicle].trajectory_area_estimation()
            if agents[name_vehicle].type == 'adult':
                away_flag = True
                for other_agent in order_optimization:
                    if other_agent != name_vehicle:
                        if agents[other_agent].type in env['Vehicle Specification']['types']:
                            if np.linalg.norm(agents[name_vehicle].position - agents[other_agent].position) <= 5:
                                away_flag = False
                if not away_flag:
                    if not agents[name_vehicle].inside_street():
                        input[f'agent {id_vehicle}'] = agents[name_vehicle].brakes()
                    else:
                        if np.linalg.norm(agents[name_vehicle].position - agents[other_agent].position) <= agents[name_vehicle].security_dist:
                            input[f'agent {id_vehicle}'] = agents[name_vehicle].brakes()
                        else:
                            #input[f'agent {id_vehicle}'] = agents[name_vehicle].move()
                            input[f'agent {id_vehicle}'] = agents[name_vehicle].trackingMPC(other_agents, ego_vehicle,circular_obstacles, t)
                else:
                    #input[f'agent {id_vehicle}'] = agents[name_vehicle].move()
                    input[f'agent {id_vehicle}'] = agents[name_vehicle].trackingMPC(other_agents, ego_vehicle, circular_obstacles, t)
            elif agents[name_vehicle].type == 'children':
                input[f'agent {id_vehicle}'] = agents[name_vehicle].trackingMPC(other_agents, ego_vehicle, circular_obstacles, t)
                #input[f'agent {id_vehicle}'] = agents[name_vehicle].move()

            other_agents[name_vehicle] = agents[name_vehicle]
        elif agents[name_vehicle].entering or agents[name_vehicle].exiting:
            agents[name_vehicle].trajecotry_estimation()
            agents[name_vehicle].trajectory_area_estimation()
            input[f'agent {id_vehicle}'] = agents[name_vehicle].trackingMPC(other_agents, ego_vehicle, circular_obstacles, t)
            other_agents[name_vehicle] = agents[name_vehicle]
            if SimulationParam['With LLM car']:
                # Here if a vehicle is more near then the security distance from the LLM car, the LLM car will brake and replan
                if check_proximity(ego_vehicle, agents[name_vehicle]):
                    counter['too_near'] += 1
                    #too_near = True
                    """if 'brakes()' in Language_Module.TP['tasks'][Language_Module.task_status]:
                        too_near = False
                    else:
                        if not SimulationParam['Controller']['Ego']['SF']['Soft']:
                            too_near = True
                        else:
                            too_near = False"""

    # Here all the steps for the LLM car
    if SimulationParam['With LLM car']:
        ego_brake = False
        next_task = False
        if 'brakes()' in Language_Module.TP['tasks'][Language_Module.task_status]:
            ego_brake = True
            ego_vehicle.safe_set['computed in optimization'] = False
            # This such because when it brakes does not call all the time TP base on the old SF cost
            ego_vehicle.previous_opt_sol_SF['Cost'] = 0
            ego_vehicle.previous_opt_sol['Cost'] = 0
            ego_vehicle.success_solver_MPC_LLM = True
            ego_vehicle.success_solver_SF = True
            if 'wait' in Language_Module.TP['tasks'][Language_Module.task_status]:
                # ??? Here is necessay or we simply replan until the LLM decid to move on???
                pattern = r'agent (\d+)'
                match = re.search(pattern, Language_Module.TP['tasks'][Language_Module.task_status])
                if match:
                    id_agent = match.group(1)
                    if 'agent ' + id_agent in Language_Module.TP['tasks'][Language_Module.task_status]:
                        if agents[id_agent].type in env['Pedestrians Specification']['types']:
                            if agents[id_agent].inside_street():
                                next_task = False
                            else:
                                next_task = True
                        elif agents[id_agent].type in env['Vehicle Specification']['types'] or agents[id_agent].type in \
                                env['Bicycle Specification']['types']:
                            if all(priority.A_p @ agents[id_agent].position <= priority.b_p):
                                next_task = False
                            else:
                                next_task = True
            if ego_vehicle.t_subtask > 15:
                next_task = True
            #if too_near:
            #    next_task = False
            #    if ego_vehicle.t_subtask > 50:
            #        next_task = True
        elif 'wait' in Language_Module.TP['tasks'][Language_Module.task_status]:
            ego_brake = True
            ego_vehicle.safe_set['computed in optimization'] = False
            # This such because when it brakes does not call all the time TP base on the old SF cost
            ego_vehicle.previous_opt_sol_SF['Cost'] = 0
            ego_vehicle.previous_opt_sol['Cost'] = 0
            ego_vehicle.success_solver_MPC_LLM = True
            ego_vehicle.success_solver_SF = True
            # ??? Here is necessay or we simply replan until the LLM decid to move on???
            pattern = r'agent (\d+)'
            match = re.search(pattern, Language_Module.TP['tasks'][Language_Module.task_status])
            if match:
                id_agent = match.group(1)
                if 'agent ' + id_agent in Language_Module.TP['tasks'][Language_Module.task_status]:
                    if agents[id_agent].type in env['Pedestrians Specification']['types']:
                        if agents[id_agent].inside_street():
                            next_task = False
                        else:
                            next_task = True
                    elif agents[id_agent].type in env['Vehicle Specification']['types'] or agents[id_agent].type in env['Bicycle Specification']['types']:
                        if all(priority.A_p @ agents[id_agent].position <= priority.b_p):
                            next_task = False
                        else:
                            next_task = True
            if ego_vehicle.t_subtask > 15:
                next_task = True
            #if too_near:
            #    next_task = False
            #    if ego_vehicle.t_subtask > 50:
            #        next_task = True
        #elif too_near:
        #    ego_brake = True
        #    error()
        #    Language_Module.final_messages.append({'Vehicle': 'brakes() because there is a car too near',
        #                                           'time': t})
        #    ego_vehicle.previous_opt_sol_SF['Cost'] = 0
        #    if abs(ego_vehicle.velocity) <= 0.01:
        #        next_task = True
        else:
            ego_vehicle.find_safe_set(agents, circular_obstacles)
            ego_vehicle.safe_set['computed in optimization'] = True
            # Check if necessary to have new Optimization Design
            if len(Language_Module.OD) == 0:
                print('Call OD the first time, for :', Language_Module.TP['tasks'][Language_Module.task_status])
                start = time.time()
                Language_Module.call_OD(SimulationParam['Environment'], agents, t)
                end = time.time()
                counter['elapsed time for LLM'].append(end-start)
                counter['OD calls'] += 1
            elif ego_vehicle.t_subtask == 0:
                print('Call OD for :', Language_Module.TP['tasks'][Language_Module.task_status])
                start = time.time()
                Language_Module.recall_OD(SimulationParam['Environment'], agents, t)
                end = time.time()
                counter['elapsed time for LLM'].append(end-start)
                counter['OD calls'] += 1

            if ego_vehicle.entering:
                info = {'street speed limit': ego_vehicle.entry['max vel']}
            elif ego_vehicle.exiting and ego_vehicle.inside_cross:
                info = {'street speed limit': ego_vehicle.exit['max vel']}
            else:
                info = {'street speed limit': ego_vehicle.final_target['max vel']}
            input_ego = ego_vehicle.MPC_LLM(agents, circular_obstacles, t, Language_Module, info)
            #input_ego = ego_vehicle.Control_Module(agents, circular_obstacles, t, Language_Module)
            if not ego_vehicle.success_solver_MPC_LLM:
                Language_Module.final_messages.append({'Vehicle': 'No success for MPC LLM solver',
                                                       'time': t})
                counter['fail solver MPC LLM'] += 1

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
            if agents[name_vehicle].type == 'emergency car':
                agents[name_vehicle].dynamics_propagation(input[f'agent {id_vehicle}'])
            else:
                agents[name_vehicle].brakes()
    else:
        # Dynamics propagation if there isn't an emergency car
        for id_vehicle, name_vehicle in enumerate(agents):
            if agents[name_vehicle].type in env['Pedestrians Specification']['types']:
                agents[name_vehicle].dynamics_propagation(input[f'agent {id_vehicle}'])
            elif agents[name_vehicle].entering or agents[name_vehicle].exiting:
                agents[name_vehicle].dynamics_propagation(input[f'agent {id_vehicle}'])
            else:
                agents[name_vehicle].brakes()

    t += 1

    # Priorities
    if SimulationParam['With LLM car']:
        ego_vehicle.t_subtask += 1
        # I don't think is the best way to do that...
        agents[str(len(agents))] = ego_vehicle
        agents = priority.SwissPriority(agents, order_optimization, SimulationParam['With LLM car'], presence_emergency_car)
        #agents = priority.NoPriority(agents, order_optimization, SimulationParam['With LLM car'])
        ego_vehicle = agents.pop(str(len(agents)-1))
    else:
        agents = priority.SwissPriority(agents, order_optimization, SimulationParam['With LLM car'], presence_emergency_car)
        #agents = priority.NoPriority(agents, order_optimization, SimulationParam['With LLM car'])

    # update the velocity limit in the new street for other agents and check i there are crush
    for name_agent in agents:
        if SimulationParam['With LLM car']:
            if agents[name_agent].type in env['Vehicle Specification']['types']:
                if agents[name_agent].entering or agents[name_agent].exiting:
                    if check_crash(ego_vehicle, agents[name_agent]):
                        counter['crash'] += 1
            else:
                if check_crash(ego_vehicle, agents[name_agent]):
                    counter['crash'] += 1
        if agents[name_agent].type in env['Vehicle Specification']['types']:
            if agents[name_agent].entering or agents[name_agent].exiting:
                agents[name_agent].update_velocity_limits(env)

    # Check if some flag say that a replan of TP is needed for LLM car
    if SimulationParam['With LLM car']:
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
                counter[
                    'motivation'] = 'End simulation: because the position of LLM car is near enough to the the final target.'
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

            elif not ego_brake:
                replan_flag = False
                for k in range(ego_vehicle.N_SF + 1):  # Think if +1 make sense...
                    pos_slack = np.linalg.norm(ego_vehicle.previous_opt_sol['psi_b_x'][:, k])
                    if pos_slack > 5:
                        replan_flag = True
                if abs(ego_vehicle.previous_opt_sol['psi_v']) > 4:
                    replan_flag = True
                if abs(ego_vehicle.previous_opt_sol['psi_f']) > 4:
                    replan_flag = True
                if np.shape(ego_vehicle.previous_opt_sol['epsilon_LLM_constraints']) == int:
                    if ego_vehicle.previous_opt_sol['epsilon_LLM_constraints'] > 1:
                        replan_flag = True
                else:
                    for k in range(ego_vehicle.N_SF + 1):
                        norm_slack = np.linalg.norm(ego_vehicle.previous_opt_sol['epsilon_LLM_constraints'][:, k])
                        if norm_slack > 1:
                            replan_flag = True

                if replan_flag:
                    print('Call TP: because slack variables of MPC LLM are high.')
                    ego_vehicle.t_subtask = 0
                    reason['MPC_LLM_not_solved'] = True
                    start = time.time()
                    Language_Module.recall_TP(env, SimulationParam['Query'], agents, ego_vehicle, reason, t)
                    end = time.time()
                    counter['elapsed time for LLM'].append(end - start)
                    counter['TP calls'] += 1
                    counter['recall due to CM'] += 1
                    ego_vehicle.success_solver_MPC_LLM = True

        for name_agent in agents:
            if agents[name_agent].type == 'emergency car':
                if agents[name_agent].exiting:
                    presence_emergency_car = False
    else:
        reach_end_target = []
        for name_agent in agents:
            if agents[name_agent].type in env['Pedestrians Specification']['types']:
                reach_end_target.append(True)
            else:
                if agents[name_agent].entering == False and agents[name_agent].exiting == False:
                    reach_end_target.append(True)
                else:
                    reach_end_target.append(False)
                if agents[name_agent].exiting and agents[name_agent].type == 'emergency car':
                    presence_emergency_car = False

        if all(reach_end_target):
            run_simulation = False

    if t == 400:
        counter['motivation'] = 'End simulation: because max simulation steps are reached.'
        run_simulation = False
    elif counter['TP calls'] >= 25:
        counter['motivation'] = 'End simulation: because LLM is called to many times.'
        run_simulation = False
    elif counter['crash'] >= 1:
        counter['motivation'] = 'End simulation: because LLM car crushes to another agent.'
        run_simulation = False

# Save the results
if SimulationParam['With LLM car']:
    agents[str(len(agents))] = ego_vehicle
    final_messages_path = os.path.join(os.path.dirname(__file__), ".", "prompts/output_LLM/messages.json")
    with open(final_messages_path, 'w') as file:
        json.dump(Language_Module.final_messages, file)
    #final_messages_path = os.path.join(os.path.dirname(__file__), ".", "prompts/output_LLM/DE_output.json")
    #with open(final_messages_path, 'w') as file:
    #    json.dump(Language_Module.DE, file)

    print('How many crash: ', counter['crash'])
    print('How many times TP is called: ', counter['TP calls'])
    print('How many times OD is called: ', counter['OD calls'])
    print('How many invasion of security area: ', counter['too_near'])
    print('How many times LLM is called due to CM: ', counter['recall due to CM'])
    counter['Mean time for LLM'] = np.mean(counter['elapsed time for LLM'])
    print('Mean elapsed time for LLM to give output: ', counter['Mean time for LLM'])
    print('Motivation: ', counter['motivation'])

    path = os.path.join(os.path.dirname(__file__), ".", "save_results/counter.json")
    with open(path, 'w') as file:
        json.dump(counter, file)

results = results_update_and_save(env, agents, results, ego_brake)

plot_simulation(SimulationParam['Environment'], env, results, t_start=0, t_end=len(results['agent 0']['x coord']))
input_animation(results, t_start=0, t_end=len(results['agent 0']['x coord']))
save_all_frames(results, env)

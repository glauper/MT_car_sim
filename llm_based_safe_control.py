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
if os.path.exists("prompts/output_LLM_based_safe_control/LLM_correction_output.json"):
    os.remove("prompts/output_LLM_based_safe_control/LLM_correction_output.json")

counter = {'crash': 0,
           'LLM calls': 1,
           'elapsed time for LLM': [],
           'too_near': 0,
           'fail solver MPC LLM': 0}

type_simulation = "llm_based_safe_control"

(SimulationParam, env, agents, ego_vehicle, LLM, presence_emergency_car, order_optimization, priority,
 results, circular_obstacles, counter) = sim_init(counter, type_simulation)
#(SimulationParam, env, agents, ego_vehicle, Language_Module, presence_emergency_car, order_optimization, priority,
# results, circular_obstacles) = sim_reload()

t = 0
run_simulation = True
ego_brake = False
while run_simulation:
    print("Simulation time: ", t)

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
    too_near = False
    for name_vehicle in order_optimization:
        id_vehicle = int(name_vehicle)
        if agents[name_vehicle].type in env['Pedestrians Specification']['types']:
            agents[name_vehicle].trajecotry_estimation()
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
                        if np.linalg.norm(agents[name_vehicle].position - agents[other_agent].position) <= agents[
                            name_vehicle].security_dist:
                            input[f'agent {id_vehicle}'] = agents[name_vehicle].brakes()
                        else:
                            # input[f'agent {id_vehicle}'] = agents[name_vehicle].move()
                            input[f'agent {id_vehicle}'] = agents[name_vehicle].trackingMPC(other_agents, ego_vehicle,
                                                                                            circular_obstacles, t)
                else:
                    # input[f'agent {id_vehicle}'] = agents[name_vehicle].move()
                    input[f'agent {id_vehicle}'] = agents[name_vehicle].trackingMPC(other_agents, ego_vehicle,
                                                                                    circular_obstacles, t)
            elif agents[name_vehicle].type == 'children':
                input[f'agent {id_vehicle}'] = agents[name_vehicle].trackingMPC(other_agents, ego_vehicle,
                                                                                circular_obstacles, t)
                # input[f'agent {id_vehicle}'] = agents[name_vehicle].move()

            other_agents[name_vehicle] = agents[name_vehicle]
        elif agents[name_vehicle].entering or agents[name_vehicle].exiting:
            agents[name_vehicle].trajecotry_estimation()
            input[f'agent {id_vehicle}'] = agents[name_vehicle].trackingMPC(other_agents, ego_vehicle,
                                                                            circular_obstacles, t)
            other_agents[name_vehicle] = agents[name_vehicle]
            if SimulationParam['With LLM car']:
                # Here if a vehicle is more near than the security distance from the LLM car, the LLM car will brake and replan
                if check_proximity(ego_vehicle, agents[name_vehicle]):
                    counter['too_near'] += 1
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
        if too_near:
            ego_brake = True
            LLM.final_messages.append({'Vehicle': 'brakes because there is a car too near',
                                       'time': t})
            ego_vehicle.previous_opt_sol_SF['Cost'] = 0
            if abs(ego_vehicle.velocity) <= 0.01:
                next_task = True
        elif len(LLM.LLM_coder[next(iter(LLM.LLM_coder))]) == 1:
            if 'brakes' in LLM.LLM_coder[next(iter(LLM.LLM_coder))]['action']:
                ego_brake = True
                # This such because when it brakes does not call all the time TP base on the old SF cost
                ego_vehicle.previous_opt_sol_SF['Cost'] = 0
                ego_vehicle.previous_opt_sol['Cost'] = 0
                ego_vehicle.success_solver_MPC_LLM = True
                ego_vehicle.success_solver_SF = True
                if ego_vehicle.t_subtask > 15:
                    next_task = True
                elif abs(ego_vehicle.velocity) <= 0.01:
                    next_task = True
        elif len(LLM.LLM_coder[next(iter(LLM.LLM_coder))]) == 3:
            if ego_vehicle.entering:
                info = {'street speed limit': ego_vehicle.entry['max vel']}
            elif ego_vehicle.exiting and ego_vehicle.inside_cross:
                info = {'street speed limit': ego_vehicle.exit['max vel']}
            else:
                info = {'street speed limit': ego_vehicle.final_target['max vel']}
            LLM.OD = LLM.LLM_coder[next(iter(LLM.LLM_coder))]
            input_ego = ego_vehicle.MPC_LLM(agents, circular_obstacles, t, LLM)
            if not ego_vehicle.success_solver_MPC_LLM:
                LLM.final_messages.append({'Vehicle': 'No success for MPC LLM solver',
                                                       'time': t})
                counter['fail solver MPC LLM'] += 1
                ego_brake = True
        else:
            print('Output strange')
            print(LLM.LLM_coder)
            error()

        #Here on

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
        if len(LLM.LLM_coder[next(iter(LLM.LLM_coder))]) == 3:
            if 'entry' in LLM.LLM_coder[next(iter(LLM.LLM_coder))]['objective']:
                ego_vehicle.target = ego_vehicle.entry['state']
                ego_vehicle.inside_cross = False
                if ego_vehicle.entering == False:
                    ego_vehicle.entering = True
                    ego_vehicle.exiting = False
                if np.linalg.norm(ego_vehicle.position - ego_vehicle.entry['position']) <= 1:
                    next_task = True
            elif 'exit' in LLM.LLM_coder[next(iter(LLM.LLM_coder))]['objective']:
                ego_vehicle.inside_cross = True
                ego_vehicle.target = ego_vehicle.exit['state']
                if ego_vehicle.exiting == False:
                    ego_vehicle.entering = False
                    ego_vehicle.exiting = True
                    if len(ego_vehicle.waypoints_exiting) != 0:
                        ego_vehicle.target = ego_vehicle.waypoints_exiting.pop(0)
                if np.linalg.norm(ego_vehicle.position - ego_vehicle.exit['position']) <= 1:
                    next_task = True
            elif 'final_target' in LLM.LLM_coder[next(iter(LLM.LLM_coder))]['objective']:
                ego_vehicle.inside_cross = False
                ego_vehicle.target = ego_vehicle.final_target['state']
                if ego_vehicle.exiting == False:
                    ego_vehicle.entering = False
                    ego_vehicle.exiting = True
                if np.linalg.norm(ego_vehicle.position - ego_vehicle.final_target['position']) <= 1:
                    next_task = True
                    counter['motivation'] = 'End simulation: because the position of LLM car is near enough to the the final target.'
                    run_simulation = False

        if run_simulation:
            reason = {'next_task': False,
                      'other_agent_too_near': False,
                      'MPC_LLM_not_solved': False}
            if next_task:
                print('Call LLM: because a task is terminated and a new one begins.')
                ego_vehicle.t_subtask = 0
                reason['next_task'] = True
                start = time.time()
                LLM.recall_LLM_coder(env, SimulationParam['Query'], agents, ego_vehicle, reason, t)
                end = time.time()
                counter['elapsed time for LLM'].append(end - start)
                counter['LLM calls'] += 1
            elif too_near:
                print('Call LLM: because an agent is to near')
                ego_vehicle.t_subtask = 0
                reason['other_agent_too_near'] = True
                start = time.time()
                LLM.recall_LLM_coder(env, SimulationParam['Query'], agents, ego_vehicle, reason, t)
                end = time.time()
                counter['elapsed time for LLM'].append(end - start)
                counter['LLM calls'] += 1
            elif not ego_vehicle.success_solver_MPC_LLM:
                print('Call LLM: because no success of MPC LLM.')
                ego_vehicle.t_subtask = 0
                reason['MPC_LLM_not_solved'] = True
                start = time.time()
                LLM.recall_LLM_coder(env, SimulationParam['Query'], agents, ego_vehicle, reason, t)
                end = time.time()
                counter['elapsed time for LLM'].append(end - start)
                counter['LLM calls'] += 1
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

    if t == 300:
        counter['motivation'] = 'End simulation: because max simulation steps are reached.'
        run_simulation = False
    elif counter['LLM calls'] >= 25:
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
        json.dump(LLM.final_messages, file)
    #final_messages_path = os.path.join(os.path.dirname(__file__), ".", "prompts/output_LLM/DE_output.json")
    #with open(final_messages_path, 'w') as file:
    #    json.dump(Language_Module.DE, file)

print('How many crash: ', counter['crash'])
print('How many times LLM is called: ', counter['LLM calls'])
print('How many invasion of security area: ', counter['too_near'])
print('How many times solver MPC LLM failed: ', counter['fail solver MPC LLM'])
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

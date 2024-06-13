import numpy as np
import time
import re
import json
import os
from functions.plot_functions import plot_simulation, input_animation
from functions.sim_functions import (results_update_and_save, sim_init, sim_reload, check_proximity, check_crash,
                                          check_need_replan)

(SimulationParam, env, agents, ego_vehicle, Language_Module, presence_emergency_car, order_optimization, priority,
 results, circular_obstacles) = sim_init()
#(SimulationParam, env, agents, ego_vehicle, Language_Module, presence_emergency_car, order_optimization, priority,
# results, circular_obstacles) = sim_reload()

t = 0
run_simulation = True
ego_brake = False
crash_count = 0
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

    # Controller to decide the trajectory of other agents
    other_agents = {}
    input = {}
    too_near = False
    for name_vehicle in order_optimization:
        id_vehicle = int(name_vehicle)
        if agents[name_vehicle].entering or agents[name_vehicle].exiting:
            agents[name_vehicle].trajecotry_estimation()
            input[f'agent {id_vehicle}'] = agents[name_vehicle].trackingMPC(other_agents, ego_vehicle, circular_obstacles, t)
            other_agents[name_vehicle] = agents[name_vehicle]
            if SimulationParam['With LLM car']:
                # Here if a vehicle is more near then the security distance from the LLM car, the LLM car will brake and replan
                if check_proximity(ego_vehicle, agents[name_vehicle]):
                    if 'brakes()' in Language_Module.TP['tasks'][Language_Module.task_status]:
                        too_near = False
                    else:
                        too_near = True
                if check_crash(ego_vehicle, agents[name_vehicle]):
                    crash_count += 1

    # Here all the steps for the LLM car
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
            Language_Module.final_messages.append({'Vehicle': 'brakes() because there is a car too near',
                                                   'time': t})
            ego_vehicle.previous_opt_sol_SF['Cost'] = 0
            if abs(ego_vehicle.velocity) <= 0.01:
                next_task = True
        else:
            # Check if necessary to have new Optimization Design
            if len(Language_Module.OD) == 0:
                print('Call OD the first time, for :', Language_Module.TP['tasks'][Language_Module.task_status])
                Language_Module.call_OD(SimulationParam['Environment'], agents, t)
            elif ego_vehicle.t_subtask == 0:
                print('Call OD for :', Language_Module.TP['tasks'][Language_Module.task_status])
                Language_Module.recall_OD(SimulationParam['Environment'], agents, t)
            # Here MPC with LLM output
            input_ego = ego_vehicle.MPC_LLM(agents, circular_obstacles, t, Language_Module)
            if not ego_vehicle.success_solver_MPC_LLM:
                Language_Module.final_messages.append({'Vehicle': 'No success for MPC LLM solver',
                                                       'time': t})

            # Here SF
            if SimulationParam['Controller']['Ego']['SF']['Active']:
                input_ego = ego_vehicle.SF(input_ego, agents, circular_obstacles, t, Language_Module)
                print('Cost SF ', ego_vehicle.previous_opt_sol_SF['Cost'])
                if not ego_vehicle.success_solver_SF:
                    Language_Module.final_messages.append({'Vehicle': 'No success for SF solver',
                                                           'time': t})

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
            if agents[name_vehicle].entering or agents[name_vehicle].exiting:
                agents[name_vehicle].dynamics_propagation(input[f'agent {id_vehicle}'])
            else:
                agents[name_vehicle].brakes()

    t += 1

    # Priorities
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

    # Check if some flag say that a replan of TP is needed for LLM car
    if SimulationParam['With LLM car']:
        run_simulation = check_need_replan(ego_vehicle, agents, Language_Module, env, SimulationParam,
                                           run_simulation, next_task, too_near, t)
        """# check if the task is finished, i.e. when LLM car is near enough to a waypoint
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
                reason = {'next_task': True, 'SF_kicks_in': False, 'other_agent_too_near': False, 'MPC_LLM_not_solved': False}
                Language_Module.recall_TP(env, SimulationParam['Query'], agents, ego_vehicle, reason, t)
            elif SimulationParam['Controller']['Ego']['SF']['Replan']['Active'] and ego_vehicle.previous_opt_sol_SF['Cost'] >= SimulationParam['Controller']['Ego']['SF']['Replan']['toll']:
                print('Call TP: because SF cost are high')
                ego_vehicle.t_subtask = 0
                reason = {'next_task': False, 'SF_kicks_in': True, 'other_agent_too_near': False, 'MPC_LLM_not_solved': False}
                Language_Module.recall_TP(env, SimulationParam['Query'], agents, ego_vehicle, reason, t)
            elif too_near:
                print('Call TP: because an agent is to near')
                ego_vehicle.t_subtask = 0
                reason = {'next_task': False, 'SF_kicks_in': False, 'other_agent_too_near': True, 'MPC_LLM_not_solved': False}
                Language_Module.recall_TP(env, SimulationParam['Query'], agents, ego_vehicle, reason, t)
            elif not ego_vehicle.success_solver_MPC_LLM:
                print(ego_vehicle.state)
                print('Call TP: because no success of MPC LLM.')
                ego_vehicle.t_subtask = 0
                reason = {'next_task': False, 'SF_kicks_in': False, 'other_agent_too_near': False, 'MPC_LLM_not_solved': True}
                Language_Module.recall_TP(env, SimulationParam['Query'], agents, ego_vehicle, reason, t)
                ego_vehicle.success_solver_MPC_LLM = True
            elif not ego_vehicle.success_solver_SF:
                print(ego_vehicle.state)
                print('Call TP: because no success of SF.')
                ego_vehicle.t_subtask = 0
                reason = {'next_task': False, 'SF_kicks_in': False, 'other_agent_too_near': False, 'MPC_LLM_not_solved': False, 'SF_not_solved': True}
                Language_Module.recall_TP(env, SimulationParam['Query'], agents, ego_vehicle, reason, t)
                ego_vehicle.success_solver_SF = True"""

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

    if t == 400:
        print('End simulation: because max simulation steps are reached.')
        run_simulation = False

# Save the results
if SimulationParam['With LLM car']:
    agents[str(len(agents))] = ego_vehicle
    final_messages_path = os.path.join(os.path.dirname(__file__), ".", "prompts/output_LLM/messages.json")
    with open(final_messages_path, 'w') as file:
        json.dump(Language_Module.final_messages, file)

print('How many crush: ', crash_count)

results = results_update_and_save(env, agents, results, ego_brake)

plot_simulation(SimulationParam['Environment'], env, results, t_start=0, t_end=len(results['agent 0']['x coord']))
input_animation(results, t_start=0, t_end=len(results['agent 0']['x coord']))

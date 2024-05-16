import numpy as np
import random
from config.config import SimulationConfig, EnviromentConfig
from functions.plot_functions import plot_simulation
from functions.sim_initialization import results_init, results_update_and_save, agents_init, collect_info_for_prompts
from env_controller import EnvController
from priority_controller import PriorityController
from vehicle import Vehicle
from llm import LLM

SimulationParam = SimulationConfig()
delta_t = SimulationParam['Timestep']
env, circular_obstacles = EnviromentConfig(SimulationParam['Environment'])
env['env number'] = SimulationParam['Environment']

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

type = env['Vehicle Specification']['types'][0]
info_vehicle = env['Vehicle Specification'][type]
ego_vehicle = Vehicle(type, info_vehicle, delta_t)
ego_vehicle.init_state_for_LLM(env, SimulationParam['Query'], SimulationParam['Controller']['Horizon'])
ego_vehicle.init_system_constraints(env["State space"], env['Ego Entrance']['speed limit'])

info = collect_info_for_prompts(env, agents, ego_vehicle)

Language_Module = LLM()
Language_Module.call_TP(SimulationParam['Environment'], SimulationParam['Query'], info, ego_vehicle)

priority = PriorityController("tracking MPC", SimulationParam['Environment'], env)

t = 0
run_simulation = True
agents[str(len(agents))] = ego_vehicle
results = results_init(env, agents)
agents.pop(str(len(agents)-1))
options_entrance = list(env['Entrances'].keys())

while run_simulation:
    print("Simulation time: ", t)
    # Save the results
    agents[str(len(agents))] = ego_vehicle
    for id_vehicle, name_vehicle in enumerate(agents):
        results = results_update_and_save(env, agents[name_vehicle], id_vehicle, results)
    agents.pop(str(len(agents)-1))

    # This is a controller that optimize the trajectory of one agent at time
    other_agents = {}
    input = {}
    #for id_vehicle, name_vehicle in enumerate(agents):
    for name_vehicle in order_optimization:
        id_vehicle = int(name_vehicle)
        if agents[name_vehicle].entering or agents[name_vehicle].exiting:
            input[f'agent {id_vehicle}'] = agents[name_vehicle].trackingMPC(other_agents, circular_obstacles, t)
            other_agents[name_vehicle] = agents[name_vehicle]

    if 'brakes()' in Language_Module.TP['tasks'][Language_Module.task_status]:
        ego_vehicle.brakes()
        ego_vehicle.t_subtask += 1
        # How to go out from here? Probably replan...
    elif 'wait' in Language_Module.TP['tasks'][Language_Module.task_status]:
        ego_vehicle.brakes()
        ego_vehicle.t_subtask += 1

        # not the best solution
        if 'agent 1' in Language_Module.TP['tasks'][Language_Module.task_status]:
            if all(priority.A_p @ agents['1'].position <= priority.b_p):
                print('Here')
            else:
                ego_vehicle.t_subtask = 0
                Language_Module.task_status += 1
        elif 'agent 0' in Language_Module.TP['tasks'][Language_Module.task_status]:
            if all(priority.A_p @ agents['0'].position <= priority.b_p):
                print('Here')
            else:
                ego_vehicle.t_subtask = 0
                Language_Module.task_status += 1
    else:
        # Check if necessary to have new Optimization Design
        if len(Language_Module.OD) == 0:
            Language_Module.call_OD(SimulationParam['Environment'])
        elif ego_vehicle.t_subtask == 0:
            Language_Module.recall_OD(SimulationParam['Environment'])

        input_ego = ego_vehicle.MPC_LLM(agents, circular_obstacles, t, Language_Module)
        #input_ego = ego_vehicle.SF(input_ego, agents, circular_obstacles, t)
        #print(ego_vehicle.previous_opt_sol_SF['Cost'])
        # Dynamics propagation
        ego_vehicle.dynamics_propagation(input_ego)
        ego_vehicle.t_subtask += 1

        # check if the task is finished with the cost of the optimization problem
        if ego_vehicle.previous_opt_sol_LLM['Cost'] <= 3:
            ego_vehicle.t_subtask = 0
            Language_Module.task_status += 1

    """# Check somehow that replan is needed or not
    if ego_vehicle.previous_opt_sol_SF['Cost'] >= 1:
        info = collect_info_for_prompts(env, agents, ego_vehicle)
        Language_Module.recall_TP(SimulationParam['Environment'], SimulationParam['Query'], info, ego_vehicle)"""

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

    agents[str(len(agents))] = ego_vehicle
    #agents = priority.NoPriority(agents)
    agents = priority.SwissPriority(agents, order_optimization)

    # not sure...
    ego_vehicle = agents.pop(str(len(agents)-1))

    # update the velocity limit in the new street
    for name_agent in agents:
        agents[name_agent].update_velocity_limits(env)

    reach_end_target = []
    for name_agent in agents:
        if agents[name_agent].entering == False and agents[name_agent].exiting == False:
            reach_end_target.append(True)
        else:
            reach_end_target.append(False)

        if agents[name_agent].exiting and agents[name_agent].type == 'emergency_car':
            presence_emergency_car = False

    """if all(reach_end_target):
        run_simulation = False"""

    if Language_Module.task_status >= len(Language_Module.TP['tasks']):
        run_simulation = False

    t += 1
    if t == 200:
        run_simulation = False

# Save the results
for id_vehicle, name_vehicle in enumerate(agents):
    results = results_update_and_save(env, agents[name_vehicle], id_vehicle, results)

plot_simulation(SimulationParam['Environment'], env, results)

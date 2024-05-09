import numpy as np
import random
from config.config import SimulationConfig, EnviromentConfig
from functions.plot_functions import plot_simulation
from functions.sim_initialization import results_init, results_update_and_save
from env_controller import EnvController
from priority_controller import PriorityController
from vehicle import Vehicle

SimulationParam = SimulationConfig()
delta_t = SimulationParam['Timestep']
env, circular_obstacles = EnviromentConfig(SimulationParam['Environment'])
agents = {}
options_init_state = list(env['Entrances'].keys())
for i in range(env['Number Vehicles']):
    type = env['Vehicle Specification']['types'][0]
    info_vehicle = env['Vehicle Specification'][type]
    agents[f'{i}'] = Vehicle(type, info_vehicle, delta_t)
    key_init = random.choice(options_init_state)
    agents[f'{i}'].init_state(env, key_init)
    agents[f'{i}'].init_system_constraints(env["State space"], env['Entrances'][key_init]['speed limit'])
    options_init_state.remove(key_init)
    agents[f'{i}'].init_trackingMPC(SimulationParam['Controller']['Horizon'])

type = env['Vehicle Specification']['types'][0]
info_vehicle = env['Vehicle Specification'][type]
ego_vehicle = Vehicle(type, info_vehicle, delta_t)
ego_vehicle.init_state(env, "Ego Entrance")

priority = PriorityController("tracking MPC", SimulationParam['Environment'], env)

t = 0
run_simulation = True
results = results_init(env, agents)
options_entrance = list(env['Entrances'].keys())
order_optimization = list(agents.keys())

while run_simulation:
    print("Simulation time: ", t)
    # Save the results
    for id_vehicle, name_vehicle in enumerate(agents):
        results = results_update_and_save(env, agents[name_vehicle], id_vehicle, results, SimulationParam['Environment'])

    # Controller for ego vehicle

    # This is a controller that optimize the trajectory of one agent at time
    other_agents = {}
    input = {}
    #for id_vehicle, name_vehicle in enumerate(agents):
    for name_vehicle in order_optimization:
        id_vehicle = int(name_vehicle)
        if agents[name_vehicle].entering or agents[name_vehicle].exiting:
            input[f'agent {id_vehicle}'] = agents[name_vehicle].trackingMPC(other_agents, circular_obstacles, t)
            other_agents[name_vehicle] = agents[name_vehicle]

    # Dynamics propagation
    for id_vehicle, name_vehicle in enumerate(agents):
        if agents[name_vehicle].entering or agents[name_vehicle].exiting:
            agents[name_vehicle].dynamics_propagation(input[f'agent {id_vehicle}'])
        else:
            agents[name_vehicle].brakes()

    """# Check if any agents have reach a target and change it in case
    for id_vehicle in range(env['Number Vehicles']):
        distance_target = np.linalg.norm(agents[f'{id_vehicle}'].position - agents[f'{id_vehicle}'].target[0:2])
        if agents[f'{id_vehicle}'].waypoints_status != 0 and distance_target <= 1:
            agents[f'{id_vehicle}'].target = agents[f'{id_vehicle}'].waypoints.pop(0)
            agents[f'{id_vehicle}'].waypoints_status = len(agents[f'{id_vehicle}'].waypoints)

        elif agents[f'{id_vehicle}'].waypoints_status == 0 and distance_target <= 1:
            options_entrance = list(env['Entrances'].keys())
            key_init = random.choice(options_entrance)
            agents[f'{id_vehicle}'].init_state(env, key_init)
            flag = True
            while flag:
                checks = []
                for id_other_vehicle in range(env['Number Vehicles']):
                    if id_vehicle != id_other_vehicle:
                        dist = np.linalg.norm(agents[f'{id_vehicle}'].position - agents[f'{id_other_vehicle}'].position)
                        if dist <= max(agents[f'{id_other_vehicle}'].security_dist, agents[f'{id_vehicle}'].security_dist):
                            checks.append(False)
                            options_entrance.remove(key_init)
                            key_init = random.choice(options_entrance)
                            agents[f'{id_vehicle}'].init_state(env, key_init)
                            #Change the type of vehicle!
                        else:
                            checks.append(True)
                if all(checks) == True:
                    flag = False
                else:
                    flag = True"""

    #agents = priority.NoPriority(agents)
    agents = priority.SwissPriority(agents, order_optimization)

    reach_end_target = []
    for name_agent in agents:
        if agents[name_agent].entering == False and agents[name_agent].exiting == False:
            reach_end_target.append(True)
        else:
            reach_end_target.append(False)

    if all(reach_end_target):
        run_simulation = False

    t += 1
    if t == 200:
        run_simulation = False

# Save the results
for id_vehicle, name_vehicle in enumerate(agents):
    results = results_update_and_save(env, agents[name_vehicle], id_vehicle, results, SimulationParam['Environment'])

plot_simulation(SimulationParam['Environment'], env, results)

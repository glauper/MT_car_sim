import numpy as np
import random
from config.config import SimulationConfig, EnviromentConfig
from functions.plot_functions import plot_simulation
from functions.sim_initialization import results_init, results_update_and_save, agents_init
from env_controller import EnvController
from priority_controller import PriorityController
from vehicle import Vehicle

SimulationParam = SimulationConfig()
delta_t = SimulationParam['Timestep']
env, circular_obstacles = EnviromentConfig(SimulationParam['Environment'])

agents = agents_init(env, delta_t, SimulationParam)

"""agents = {}
options_init_state = list(env['Entrances'].keys())
nr_type_vehicles = len(env['Vehicle Specification']['types'])

for i in range(env['Number Vehicles']):
    # put a vehicle for each type different from emergency_car
    if  nr_type_vehicles > 1 and i < nr_type_vehicles - 1:
        type = env['Vehicle Specification']['types'][i+1]
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
        same_angle = agents[f'{i}'].theta
        shift_angle_90 = agents[f'{i}'].theta + np.pi/2
        if shift_angle_90 > np.pi:
            shift_angle_90 = shift_angle_90 - 2*np.pi
        shift_angle_180 = agents[f'{i}'].theta + np.pi
        if shift_angle_180 > np.pi:
            shift_angle_180 = shift_angle_180 - 2*np.pi
        target_angle = agents[f'{i}'].waypoints_exiting[-1][2]
        if same_angle == target_angle or shift_angle_90 == target_angle or shift_angle_180 == target_angle:
            new = np.zeros((4,1))
            if agents[f'{i}'].theta == 0:
                new[0] = -3
                new[1] = -3
            elif agents[f'{i}'].theta == np.pi / 2:
                new[0] = 3
                new[1] = 3
            elif agents[f'{i}'].theta == np.pi:
                new[0] = -3
                new[1] = 3
            elif agents[f'{i}'].theta == -np.pi / 2:
                new[0] = -3
                new[1] = -3
            new[2] = agents[f'{i}'].theta
            new[3] = 0
            agents[f'{i}'].waypoints_exiting.insert(0, new)
        if shift_angle_180 == target_angle:
            new = np.zeros((4, 1))
            if agents[f'{i}'].theta == 0:
                new[0] = 3
                new[1] = 3
            elif agents[f'{i}'].theta == np.pi / 2:
                new[0] = -3
                new[1] = 3
            elif agents[f'{i}'].theta == np.pi:
                new[0] = -3
                new[1] = -3
            elif agents[f'{i}'].theta == -np.pi / 2:
                new[0] = 3
                new[1] = -3
            new[2] = agents[f'{i}'].waypoints_exiting[-1][2]
            new[3] = 0
            agents[f'{i}'].waypoints_exiting.insert(1, new)"""

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
ego_vehicle.init_state(env, "Ego Entrance")

priority = PriorityController("tracking MPC", SimulationParam['Environment'], env)

t = 0
run_simulation = True
results = results_init(env, agents)
options_entrance = list(env['Entrances'].keys())

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

    if presence_emergency_car:
        # Dynamics propagation
        for id_vehicle, name_vehicle in enumerate(agents):
            if agents[name_vehicle].type == 'emergency_car':
                agents[name_vehicle].dynamics_propagation(input[f'agent {id_vehicle}'])
            else:
                agents[name_vehicle].brakes()
    else:
        # Dynamics propagation
        for id_vehicle, name_vehicle in enumerate(agents):
            if agents[name_vehicle].entering or agents[name_vehicle].exiting:
                    agents[name_vehicle].dynamics_propagation(input[f'agent {id_vehicle}'])
            else:
                agents[name_vehicle].brakes()

    #agents = priority.NoPriority(agents)
    agents = priority.SwissPriority(agents, order_optimization)

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

    if all(reach_end_target):
        run_simulation = False

    t += 1
    if t == 200:
        run_simulation = False

# Save the results
for id_vehicle, name_vehicle in enumerate(agents):
    results = results_update_and_save(env, agents[name_vehicle], id_vehicle, results, SimulationParam['Environment'])

plot_simulation(SimulationParam['Environment'], env, results)

import numpy as np
import casadi as ca
import random
from vehicle import Vehicle

class PriorityController:
    def __init__(self, vehicle_controller, env_number, env):
        self.vehicle_controller = vehicle_controller
        self.env_nr = env_number
        self.env = env


    def NoPriority(self, agents):

        # Check if any agents have reach a target and change it in case
        for id_vehicle in range(self.env['Number Vehicles']):
            distance_target = np.linalg.norm(agents[f'{id_vehicle}'].position - agents[f'{id_vehicle}'].target[0:2])
            if agents[f'{id_vehicle}'].waypoints_status != 0 and distance_target <= 1:
                agents[f'{id_vehicle}'].target = agents[f'{id_vehicle}'].waypoints.pop(0)
                agents[f'{id_vehicle}'].waypoints_status = len(agents[f'{id_vehicle}'].waypoints)

            elif agents[f'{id_vehicle}'].waypoints_status == 0 and distance_target <= 1:
                options_entrance = list(self.env['Entrances'].keys())
                key_init = random.choice(options_entrance)
                agents[f'{id_vehicle}'].init_state(self.env, key_init)
                flag = True
                while flag:
                    checks = []
                    for id_other_vehicle in range(self.env['Number Vehicles']):
                        if id_vehicle != id_other_vehicle:
                            dist = np.linalg.norm(
                                agents[f'{id_vehicle}'].position - agents[f'{id_other_vehicle}'].position)
                            if dist <= max(agents[f'{id_other_vehicle}'].security_dist,
                                           agents[f'{id_vehicle}'].security_dist):
                                checks.append(False)
                                options_entrance.remove(key_init)
                                key_init = random.choice(options_entrance)
                                agents[f'{id_vehicle}'].init_state(self.env, key_init)
                                # Change the type of vehicle!
                            else:
                                checks.append(True)
                    if all(checks) == True:
                        flag = False
                    else:
                        flag = True

        return agents

    def SwissPriority(self, agents):

        if self.env_nr == 0:
            agents = self.env_0_SiwssPriority(agents)
        elif self.env_nr == 1:
            agents = self.env_1_SiwssPriority(agents)
        else:
            print('Not defined')
            error()

        return agents

    def env_0_SiwssPriority(self, agents):

        for id_vehicle in range(self.env['Number Vehicles']):

            print('Boh')

        return agents

    def env_1_SiwssPriority(self, agents):

        return agents
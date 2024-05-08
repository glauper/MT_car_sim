import numpy as np
import casadi as ca
import random
from vehicle import Vehicle

class PriorityController:
    def __init__(self, vehicle_controller, env_number, env):
        self.vehicle_controller = vehicle_controller
        self.env_nr = env_number
        self.env = env
        A = np.zeros((4, 2))
        A[0, 0] = 1
        A[1, 0] = -1
        A[2, 1] = 1
        A[3, 1] = -1

        b = np.zeros((4, 1))
        b[0,0] = env['Priority space']['x limits'][1]
        b[1,0] = -env['Priority space']['x limits'][0]
        b[2,0] = env['Priority space']['y limits'][1]
        b[3,0] = -env['Priority space']['y limits'][0]

        self.A_p = A
        self.b_p = b


    def NoPriority(self, agents):

        # Check if any agents have reach a target and change it in case
        for id_vehicle in range(self.env['Number Vehicles']):
            distance_target = np.linalg.norm(agents[f'{id_vehicle}'].position - agents[f'{id_vehicle}'].target[0:2])
            if len(agents[f'{id_vehicle}'].waypoints_entering) != 0 and distance_target <= 1:
                agents[f'{id_vehicle}'].target = agents[f'{id_vehicle}'].waypoints_entering.pop(0)
            elif len(agents[f'{id_vehicle}'].waypoints_entering) == 0 and len(agents[f'{id_vehicle}'].waypoints_exiting) != 0 and distance_target <= 1:
                agents[f'{id_vehicle}'].target = agents[f'{id_vehicle}'].waypoints_exiting.pop(0)
                agents[f'{id_vehicle}'].exiting = True
                agents[f'{id_vehicle}'].entering = False
            elif len(agents[f'{id_vehicle}'].waypoints_exiting) == 0 and distance_target <= 1:
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

                agents[f'{id_vehicle}'].exiting = False
                agents[f'{id_vehicle}'].entering = True

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
        ids_entering = []
        ids_exiting = []
        for id_agent in range(len(agents)):
            if agents[f'{id_agent}'].entering:
                ids_entering.append(f'{id_agent}')
            elif agents[f'{id_agent}'].exiting:
                ids_exiting.append(f'{id_agent}')

        # Here is assumed there are no way point between the start and the waypoint before the cross!
        if len(ids_entering) != 0:
            if len(ids_entering) == 1:
                priority = [True]
            if len(ids_entering) > 1:
                priority = [False] * len(ids_entering)
                for i, id_agent in enumerate(ids_entering):
                    priority_i = True
                    for j, id_other_agent in enumerate(ids_entering):
                        if id_agent != id_other_agent:
                            dist_i = np.linalg.norm(agents[id_agent].target[0:2] - agents[id_agent].position)
                            dist_j = np.linalg.norm(agents[id_other_agent].target[0:2] - agents[id_other_agent].position)
                            """# If the car is near enough to the target we consider some priorities
                            if dist_i <= 10:
                                # If the other car is less than 5m more distant from the target, then we give anyway the priority
                                if dist_j <= dist_i + 10:
                                    # If the car is on the right we give the priority
                                    if theta <= agents[id_agent].target[2]:
                                        priority_i = False
                            else: # If the car is far away from the target we consider, have not priority
                                priority_i = False"""
                            # If the other car is less than 5m more distant from the target, then we give anyway the priority
                            if dist_j <= dist_i + 10:
                                # If the car is on the right we give the priority
                                if agents[id_agent].target[2] == 0:
                                    if agents[id_other_agent].target[2] == np.pi/2:
                                        priority_i = False
                                elif agents[id_agent].target[2] == np.pi/2:
                                    if agents[id_other_agent].target[2] == np.pi:
                                        priority_i = False
                                elif agents[id_agent].target[2] == np.pi:
                                    if agents[id_other_agent].target[2] == -np.pi/2:
                                        priority_i = False
                                elif agents[id_agent].target[2] == -np.pi/2:
                                    if agents[id_other_agent].target[2] == 0:
                                        priority_i = False
                    if priority_i:
                        priority[i] = True

            if True in priority:
                id_priority_vehicle = ids_entering[priority.index(True)]
                if len(id_priority_vehicle) > 1:
                    error()

                if len(ids_exiting) >= 1:
                    check_traffic = True
                    for id_other_agent in ids_exiting:
                        if all(self.A_p @ agents[id_other_agent].position <= self.b_p):
                            dist_agent_to_agent = np.linalg.norm(agents[id_priority_vehicle].position - agents[id_other_agent].position)
                            dist_agent_to_target = np.linalg.norm(agents[id_other_agent].position - agents[id_other_agent].target[0:2])
                            dist_own_target = np.linalg.norm(agents[id_priority_vehicle].position - agents[id_priority_vehicle].target[0:2])
                            if dist_agent_to_agent <= max(agents[id_priority_vehicle].security_dist, agents[id_other_agent].security_dist) + 0.5:
                                check_traffic = False
                            elif dist_agent_to_target >= 2:
                                check_traffic = False
                            elif dist_own_target >= 1:
                                check_traffic = False

                    if check_traffic:
                        agents[id_priority_vehicle].target = agents[id_priority_vehicle].waypoints_exiting.pop(0)
                        agents[id_priority_vehicle].exiting = True
                        agents[id_priority_vehicle].entering = False

                else:
                    dist_own_target = np.linalg.norm(agents[id_priority_vehicle].position - agents[id_priority_vehicle].target[0:2])
                    if dist_own_target <= 1:
                        agents[id_priority_vehicle].target = agents[id_priority_vehicle].waypoints_exiting.pop(0)
                        agents[id_priority_vehicle].exiting = True
                        agents[id_priority_vehicle].entering = False

        for id_agent in ids_exiting:
            distance_target = np.linalg.norm(agents[id_agent].position - agents[id_agent].target[0:2])
            if len(agents[id_agent].waypoints_entering) != 0 and distance_target <= 1:
                agents[id_agent].target = agents[id_agent].waypoints_entering.pop(0)
            elif len(agents[id_agent].waypoints_exiting) != 0 and distance_target <= 2:
                agents[id_agent].target = agents[id_agent].waypoints_exiting.pop(0)
            elif len(agents[id_agent].waypoints_exiting) == 0 and distance_target <= 6:
                agents[id_agent].exiting = False
                agents[id_agent].entering = False

        return agents

    def env_1_SiwssPriority(self, agents):

        return agents
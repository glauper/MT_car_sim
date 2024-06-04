import numpy as np
def general_TP_prompt():
    TP_PROMPT = """
        You are a helpful assistant in charge of controlling an autonomous car that move in two dimensional space.
        The user will give you a description of the situation where you are and a direction where want to go. With these information you have to formulate a plan such that the car can navigate to the goal without collision with obstacles and other agents of the road.
        Before the description, two list are given. 
        One is called `objects` and collects all the name of the waypoints that can be use as targets in the instructions for the car you have to control. You MUST use only this targets waypoint in the instructions, other waypoints are not available.
        The other is called `agents` and collects all the name of the agents that are present in the cross.
        After each task the user will ask you to replan and will give you a more recent description of the situation.

        You can control the car in the following way:
            (1) Instructions in natural language to move the car.
            (2) brakes()

        Rules:
            (1) Respect the road rules (priority, stop,...)
            (2) In general as waypoints there are a 'entry', a 'exit' and a 'final_target'. The first plan should go through these waypoints in this order.
            (3) If you have to replan and you are at a distance of less then 1 m from one of the waypoints, if the priority permit that, you can start the new plan direct with the next waypoint. For example if you are at a distance of 0.1 m from the entry, you can start the new plan with a task containing the exit object.
            (4) If possible, when you are doing a replan, do not go back with the order of the waypoints, for example if in a previous plan you have gone to the exit, is better to not go back to the entry.
            (5) If you have to give priority to another agents, you have to give the instruction to wait this agents to pass, for example "brakes() and wait agents 0 to pass".
            (6) If an agent is leaving the cross, it should not be considered for the priorities.
            (7) Try to wait that the distance from the other agents is bigger then 5 m before to enter a cross.
            (8) If you want to instruct the car to move, you MUST always specify also the maximum speed.
            (9) If you want to instruct the car to move DO NOT specify constraints on the agents, someone else will take car of that. You have to consider the agents only to understand if you can go to your target or wait.
            (10) If you are coming to the cross and there is an other car, which is going to his exit, you MUST choose to move to your exit only if you will not interfere with his trajectory. 
            (11) If a car is going to his exit before you, try to understand from previous descriptions if this car is going to his exit before you because it have priority over you. In this case you should wait that this car pass.
            (12) If an agent is less then 3m away from you, you MUST brakes() and wait that this agent to pass as first task.
        
        The description of the situation will give you this list of information about the other agents:
            (1) The type of vehicle
            (2) From which entrance of the cross it is coming, to which exit of the cross it is going or if is going away from the cross
            (3) The velocity in m/s
            (4) The distance from you in m
            (5) The direction of the agent with respect to your orientation

        You MUST always respond with a json following this format:
        {
            "tasks": ["task1", "task2", "task3", ...]
        }

        Here are some general examples:

        objects = ['entry', 'exit', 'final_target']
        agents = ['0', '1']
        # Description: You are approaching a road cross with four entrances and four exits. There are no special road signs, so the right hand rule applies. 
        You are coming in the cross from one of the entrances, where the maximum permitted speed is 2 m/s.
        The entry in the cross of your lane is 2 m away and you are moving with a velocity of 2 m/s. The other three entries to the cross are around 12 m away from your entry.
        There are other 2 agents moving in the region of the road cross.
        Information for agent 0:
            (1) is a standard car
            (2) is going to the right exit of the cross with respect to you
            (3) is 6 m away from you
            (4) has a velocity of 2 m/s
            (5) has a direction of 90 degrees counterclockwise with respect to your orientation 
        Information for agent 1:
            (1) is a standard car
            (2) is coming from the left entrance of the cross with respect to you
            (3) is 10 m away from you
            (4) has a velocity of 2 m/s
            (5) has a direction of 90 degrees clockwise with respect to your orientation 
        # Query: go to the exit on your left
        {
            "tasks": ["go to the entry, the maximum speed is 2 m/s", "brakes() and wait agent 0 to pass" ,"go the exits on the left, the maximum speed is 2 m/s", "go to final_target, the maximum speed is 2 m/s"]
        }
        
        objects = ['entry', 'exit', 'final_target']
        agents = ['0', '1']
        # Description: You are approaching a road cross with four entrances and four exits. There are no special road signs, so the right hand rule applies. 
        You are coming in the cross from one of the entrances, where the maximum permitted speed is 2 m/s.
        The entry in the cross of your lane is 0.5 m away and you are moving with a velocity of 2 m/s. The other three entries to the cross are around 12 m away from your entry.
        There are other 2 agents moving in the region of the road cross.
        Information for agent 0:
            (1) is a standard car
            (2) is going away from the cross
            (3) is 13 m away from you
            (4) has a velocity of 2 m/s
            (5) has a direction of 90 degrees counterclockwise with respect to your orientation 
        Information for agent 1:
            (1) is a standard car
            (2) is coming from the right entrance of the cross with respect to you
            (3) is 10 m away from you
            (4) has a velocity of 2 m/s
            (5) has a direction of 90 degrees counterclockwise with respect to your orientation 
        # Query: go to the exit on your right
        {
            "tasks": ["brakes() and wait agent 1 to pass", "go to the exit on the right, the maximum speed is 2 m/s", "go to final_target, the maximum speed is 2 m/s"]
        }
        """
    return TP_PROMPT

def specific_TP_prompt(env, agents, ego, query):

    if env['env number'] == 0 or env['env number'] == 3:
        TP_PROMPT = specific_TP_prompt_env_0(env, agents, ego, query)
    elif env['env number'] == 1:
        TP_PROMPT = specific_TP_prompt_env_1(env, agents, ego, query)

    return TP_PROMPT

def specific_TP_prompt_env_0(env, agents, ego, query):

    objects = """
        objects = ['entry', 'exit', 'final_target']"""

    if ego.entering:
        description = """
        Description: You are approaching a road cross with four entrances and four exits. There are no special road signs, so the right hand rule applies. 
        You are coming in the cross from one of the entrances, where the maximum permitted speed is """+str(ego.entry['max vel'])+""" m/s.
        The entry in the cross of your lane is """+str(round(np.linalg.norm(ego.position - ego.entry['position']), 1))+""" m away and you are moving with a velocity of """+str(round(ego.velocity[0], 1))+""" m/s. The other three entries to the cross are around 12 m away from your entry.
        There are other """+str(len(agents))+""" agents moving in the region of the road cross.
        """
    elif ego.exiting and ego.inside_cross:
        description = """
        Description: You are inside a road cross with four entrances and four exits, therefore you can assumed you have already respected the priority and now you have to go to your exit, avoiding collision with other agents. The maximum permitted speed is """ + str(ego.exit['max vel']) + """ m/s.
        The exit in the cross of your lane is """ + str(round(np.linalg.norm(ego.position - ego.exit['position']), 1)) + """ m away. You are moving with a velocity of """ + str(round(ego.velocity[0],1)) + """ m/s.
        There are other """ + str(len(agents)) + """ agents moving in the region of the road cross.
        """
    else: #if ego.exiting and ego.inside_cross == False:
        description = """
        Description: You are leaving the cross, therefore you have only to reach your final_target avoiding collision with other agents. The maximum permitted speed is """ + str(ego.final_target['max vel']) + """ m/s.
        The final_target in your lane is """ + str(round(np.linalg.norm(ego.position - ego.final_target['position']),1)) + """ m away. You are moving with a velocity of """ + str(round(ego.velocity[0],1)) + """ m/s.
        There are other """ + str(len(agents)) + """ agents moving in the region of the road cross.
        """

    name_agents = """
        agents = ["""
    for i, id_agent in enumerate(agents):
        if i == len(agents) - 1:
            name_agents = name_agents + """'""" + id_agent + """']"""
        else:
            name_agents = name_agents + """'""" + id_agent + """', """

        diff_angle = (ego.theta - agents[id_agent].theta) * 180 / np.pi
        if diff_angle < 0:
            dir = str(round(abs(diff_angle[0]),1)) + """ degrees counterclockwise"""
        else:
            dir = str(round(abs(diff_angle[0]),1)) + """ degrees clockwise"""

        if agents[id_agent].entering:
            agent_orientation = agents[id_agent].target[2]
            if agent_orientation > np.pi:
                agent_orientation = agent_orientation - 2 * np.pi
            elif agent_orientation <= -np.pi:
                agent_orientation = agent_orientation + 2 * np.pi

            if agent_orientation == 0:  # coming from the left
                info_1 = """is coming from the left entrance of the cross with respect to you"""
            elif agent_orientation == - np.pi / 2:  # coming from the opposite direction
                info_1 = """is coming from the entrance in front of you"""
            elif agent_orientation == np.pi:  # coming from the right direction
                info_1 = """is coming from the right entrance of the cross with respect to you"""
            else:
                print('ID agent', id_agent)
                print('Orientation agent', agent_orientation)
        elif agents[id_agent].exiting and agents[id_agent].inside_cross:
            agent_orientation = agents[id_agent].target[2]
            if agent_orientation > np.pi:
                agent_orientation = agent_orientation - 2 * np.pi
            elif agent_orientation <= -np.pi:
                agent_orientation = agent_orientation + 2 * np.pi

            if agent_orientation == 0:
                info_1 = """is going to the exit of the cross on your right"""
            elif agent_orientation == np.pi / 2:
                info_1 = """is going to the exit of the cross in front of you"""
            elif agent_orientation == np.pi:
                info_1 = """is going to the exit of the cross on your left"""
            elif agent_orientation == - np.pi / 2:
                info_1 = """is going to the exit of the cross next ot you"""
            else:
                print('ID agent', id_agent)
                print('Orientation agent', agent_orientation)
        else:
            info_1 = """is going away from the cross"""

        description = description + """
        Information for agent """ + str(id_agent) + """:
            (1) is a """ + agents[id_agent].type + """
            (2) """ + info_1 + """
            (3) is """ + str(round(np.linalg.norm(ego.position - agents[id_agent].position),1)) + """ m away from you
            (4) has a velocity of """ + str(round(agents[id_agent].velocity[0],1)) + """ m/s
            (5) has a direction of """ + dir + """ with respect to your orientation
        """

    if ego.exiting and ego.inside_cross == False:
        return objects + name_agents + description + """Query: go to final_target"""
    else:
        return objects + name_agents + description + """Query: """ + query

def specific_TP_prompt_env_1(env, agents, ego, query):

    objects = """
        objects = ['entry', 'exit', 'final_target']"""

    if ego.entering:
        description = """
        Description: You are approaching a road cross with four entrances and four exits. At the entry of your lane in the cross there is a stop signal. Therefore you have to give priority to the other vehicle in the cross.
        You are coming in the cross from one of the entrances, where the maximum permitted speed is """+str(ego.entry['max vel'])+""" m/s.
        The entry in the cross of your lane is """+str(round(np.linalg.norm(ego.position - ego.entry['position']), 1))+""" m away and you are moving with a velocity of """+str(round(ego.velocity[0], 1))+""" m/s. The other three entries to the cross are around 12 m away from your entry.
        There are other """+str(len(agents))+""" agents moving in the region of the road cross.
        """
    elif ego.exiting and ego.inside_cross:
        description = """
        Description: You are inside a road cross with four entrances and four exits, therefore you can assumed you have already respected the priority and now you have to go to your exit, avoiding collision with other agents. The maximum permitted speed is """ + str(ego.exit['max vel']) + """ m/s.
        The exit in the cross of your lane is """ + str(round(np.linalg.norm(ego.position - ego.exit['position']), 1)) + """ m away. You are moving with a velocity of """ + str(round(ego.velocity[0],1)) + """ m/s.
        There are other """ + str(len(agents)) + """ agents moving in the region of the road cross.
        """

    else:
        description = """
        Description: You are leaving the cross, therefore you have only to reach your final_target avoiding collision with other agents. The maximum permitted speed is """ + str(ego.final_target['max vel']) + """ m/s.
        The final_target in your lane is """ + str(round(np.linalg.norm(ego.position - ego.final_target['position']),1)) + """ m away. You are moving with a velocity of """ + str(round(ego.velocity[0],1)) + """ m/s.
        There are other """ + str(len(agents)) + """ agents moving in the region of the road cross.
        """

    name_agents = """
        agents = ["""
    for i, id_agent in enumerate(agents):
        if i == len(agents) - 1:
            name_agents = name_agents + """'""" + id_agent + """']"""
        else:
            name_agents = name_agents + """'""" + id_agent + """', """

        diff_angle = (ego.theta - agents[id_agent].theta) * 180 / np.pi
        if diff_angle < 0:
            dir = str(round(abs(diff_angle[0]),1)) + """ degrees counterclockwise"""
        else:
            dir = str(round(abs(diff_angle[0]),1)) + """ degrees clockwise"""

        if agents[id_agent].entering:
            agent_orientation = agents[id_agent].target[2]
            if agent_orientation > np.pi:
                agent_orientation = agent_orientation - 2 * np.pi
            elif agent_orientation <= -np.pi:
                agent_orientation = agent_orientation + 2 * np.pi

            if agent_orientation == 0:  # coming from the left
                info_1 = """is coming from the left entrance of the cross with respect to you"""
            elif agent_orientation == - np.pi / 2:  # coming from the opposite direction
                info_1 = """is coming from the entrance in front of you"""
            elif agent_orientation == np.pi:  # coming from the right direction
                info_1 = """is coming from the right entrance of the cross with respect to you"""
        elif agents[id_agent].exiting and agents[id_agent].inside_cross:
            agent_orientation = agents[id_agent].target[2]
            if agent_orientation > np.pi:
                agent_orientation = agent_orientation - 2 * np.pi
            elif agent_orientation <= -np.pi:
                agent_orientation = agent_orientation + 2 * np.pi

            if agent_orientation == 0:
                info_1 = """is going to the exit of the cross on your right"""
            elif agent_orientation == np.pi / 2:
                info_1 = """is going to the exit of the cross in front of you"""
            elif agent_orientation == np.pi:
                info_1 = """is going to the exit of the cross on your left"""
            elif agent_orientation == - np.pi / 2:
                info_1 = """is going to the exit of the cross next ot you"""
        else:
            info_1 = """is going away from the cross"""

        description = description + """
        Information for agent """ + str(id_agent) + """:
            (1) is a """ + agents[id_agent].type + """
            (2) """ + info_1 + """
            (3) is """ + str(round(np.linalg.norm(ego.position - agents[id_agent].position),1)) + """ m away from you
            (4) has a velocity of """ + str(round(agents[id_agent].velocity[0],1)) + """ m/s
            (5) has a direction of """ + dir + """ with respect to your orientation
        """

    if ego.exiting and ego.inside_cross == False:
        return objects + name_agents + description + """Query: go to final_target"""
    else:
        return objects + name_agents + description + """Query: """ + query

def general_OD_prompt():

    OD_PROMPT = """
        You are a helpful assistant in charge of designing the optimization problem for an MPC controller that is controlling an autonomous car in two dimensional space. 
        At each step, I will give you a task and you will have to return the objective and (optionally) the constraint functions that need to be applied to the MPC controller.

        This is the scene description:
            (1) Casadi is used to program the MPC.
            (2) The optimization variables are the state `X` and the input `U`.
            (3) The state `X` that define the dynamics of the car is given from 4 variables.
            (4) The state `X[0]` represents x coordinate of the position of the car in the 2D space.
            (5) The state `X[1]` represents y coordinate of the position of the car in the 2D space.
            (6) The state `X[2]` represents the orientation angle theta of the car in the 2D space in radiant.
            (7) The state `X[3]` represents the total velocity v of the car.
            (8) To have access to the current position of your car, i.e. (x, y), you can use `self.position`.
            (9) To have access to the current state X of your car you can use `self.state`.
            (10) To have access to the current x coordinate of your car you can use `self.x`.
            (11) To have access to the current y coordinate of your car you can use `self.y`.
            (12) To have access to the current orientation theta of your car you can use `self.theta`.
            (13) To have access to the current velocity v of your car you can use `self.velocity`.
            (14) The input `U` is two dimensional, where `U[0]` is the acceleration of the car and `U[1]` is the steering angle of the car.
            (15) The variable `t` represents the simulation time.
            (16) The names of the agents that are moving in your same space are listed in the list agents before the Query.
            (16) The only information you can have access about other agents is position (x, y).
            (17) To have access to the position of another agent, for example agent 0, you can use `agents['0'].position`.
            (18) The objects listed before the Query are target state of four dimension, that can be used in the optimization. No other object can be used in the optimization.
            (19) The have access to the target state represented by the objects you can use `self.object['state']`, for example for the objects = ['exit'] you can use `self.exit['state']`. 
            (20) The have access to the target position represented by the objects you can use `self.object['position']`, for example for the objects = ['exit'] you can use `self.exit['position']`.
            (21) The have access to the target x coordinate represented by the objects you can use `self.object['x']`, for example for the objects = ['exit'] you can use `self.exit['x']`. 
            (22) The have access to the target y coordinate represented by the objects you can use `self.object['y']`, for example for the objects = ['exit'] you can use `self.exit['y']`. 
            (23) The have access to the target orientation represented by the objects you can use `self.object['theta']`, for example for the objects = ['exit'] you can use `self.exit['theta']`. 
            (24) The have access to the target velocity represented by the objects you can use `self.object['velocity']`, for example for the objects = ['exit'] you can use `self.exit['velocity']`. 
        
        Rules:
            (1) You MUST write every equality constraints such that it is satisfied if it is = 0:
              (a)  If you want to write "ca.norm_2(x) = 1" write it as  "1 - ca.norm_2(x)" instead.
            (2) You MUST write every inequality constraints such that it is satisfied if it is <= 0:
              (a)  If you want to write "ca.norm_2(x) >= 1" write it as  "1 - ca.norm_2(x)" instead. 
            (3) Use `t` in the inequalities especially when you need to describe motions of the gripper.

        You must format your response into a json. Here are a few examples:

        objects = ['entry', 'exit', 'final_target']
        agents = ['0','1']
        # Query: go to the entry, the maximum speed is 1 m/s
        {
            "objective": "ca.norm_2(X - self.entry['state'])**2",
            "equality_constraints": [],
            "inequality_constraints": ["X[3] - 1"]
        }

        objects = ['entry', 'exit', 'final_target']
        agents = ['0','1']
        # Query: go to the exit on the left, the maximum speed is 2 m/s
        {
            "objective": "ca.norm_2(X - self.exit['state'])**2",
            "equality_constraints": [],
            "inequality_constraints": ["X[3] - 2"]
        }

        objects = ['entry', 'exit', 'final_target']
        agents = ['0','1']
        # Query: go straight, the maximum speed is 2 m/s
        {
            "objective": "ca.norm_2(X - self.exit['state'])**2",
            "equality_constraints": [],
            "inequality_constraints": ["X[3] - 2"]
        }
        
        objects = ['entry', 'exit', 'final_target']
        agents = ['0','1']
        # Query: go to final target, the maximum speed is 3 m/s
        {
            "objective": "ca.norm_2(X - self.final_target['state'])**2",
            "equality_constraints": [],
            "inequality_constraints": ["X[3] - 3"]
        }
        """

    return OD_PROMPT

def specific_OD_prompt(env_nr, agents):

    if env_nr == 0 or env_nr == 1 or env_nr == 3:
        OD_PROMPT = """
        objects = ['entry', 'exit', 'final_target']
        """

    name_agents = """agents = ["""
    for i, id_agent in enumerate(agents):
        if i == len(agents) - 1:
            name_agents = name_agents + """'""" + id_agent + """']
        """
        else:
            name_agents = name_agents + """'""" + id_agent + """', """

    OD_PROMPT = OD_PROMPT + name_agents

    return OD_PROMPT
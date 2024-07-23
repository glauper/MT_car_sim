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
    (7) If you want to instruct the car to move, you MUST always specify also the maximum speed.
    (8) If you want to instruct the car to move DO NOT specify constraints on the agents, someone else will take car of that. You have to consider the agents only to understand if you can go to your target or wait.
    (9) If you are coming to the cross and there is an other car, which is going to his exit, you MUST choose to move to your exit only if you will not interfere with his trajectory. 
    (10) If a car is going to his exit before you, try to understand from previous descriptions if this car is going to his exit before you because it have priority over you. In this case you should wait that this car pass.
    (11) If an agent is less then 3m away from you, you MUST brakes() and wait that this agent to pass as first task.

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

    if env['env number'] == 0 or env['env number'] == 3 or env['env number'] == 4:
        TP_PROMPT = specific_TP_prompt_env_0(env, agents, ego, query)
    elif env['env number'] == 1:
        TP_PROMPT = specific_TP_prompt_env_1(env, agents, ego, query)
    elif env['env number'] == 2:
        TP_PROMPT = specific_TP_prompt_env_2(env, agents, ego, query)

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

def specific_TP_prompt_env_2(env, agents, ego, query):

    objects = """
        objects = ['entry', 'exit', 'final_target']"""

    if ego.entering:
        description = """
Description: You are approaching a road cross with four entrances and four exits. You are coming in the cross from one of the entrances, where the maximum permitted speed is """+str(ego.entry['max vel'])+""" m/s.
You are on the main road and can see that the entrances of the road cross on your left and right have stop signals.
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

def TP_prompt_with_DE(DE_description):

    TP_PROMPT = """
You are a helpful assistant in charge of controlling an autonomous car that move in two dimensional space.
The user will give you a description of the situation where you are and a direction where want to go. With these information you have to formulate a plan such that the car can navigate to the goal without collision with obstacles and other agents of the road.
Before the description, two list are given. One is called objects and collects all the name of the waypoints that can be use as targets in the instructions for the car you have to control. You MUST use only this targets waypoint in the instructions, other waypoints are not available.
The other is called agents and collects all the name of the agents that are present in the cross.
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
    (7) If you want to instruct the car to move, you MUST always specify also the maximum speed.
    (8) If you want to instruct the car to move DO NOT specify constraints on the agents, someone else will take car of that. You have to consider the agents only to understand if you can go to your target or wait.
    (9) If you are coming to the cross and there is an other car, which is going to his exit, you MUST choose to move to your exit only if you will not interfere with his trajectory. 
    (10) If a car is going to his exit before you, try to understand from previous descriptions if this car is going to his exit before you because it have priority over you. In this case you should wait that this car pass.
    (11) If an agent is less then 3m away from you, you MUST brakes() and wait that this agent to pass as first task.
        
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
# Description: Based on the provided image and additional information, here is the description of the current road situation:

Current Situation:

Road Layout:
- You are approaching a four-way intersection. 
- The coordinate origin is at the center of the intersection.
- The waypoints you need to pass through are marked with blue points labeled as entry, exit, and final_target.

Your Vehicle:
- Type: Standard car
- Position: (3.0, -20.0) meters
- Velocity: 0.0 m/s (currently stationary)
- Orientation: 90.0° (facing northward)
- Intention: To proceed to the exit on the left by passing all marked waypoints (entry, exit, final_target)
- Speed Limit: 2 m/s in your lane

Other Agents:
- Agent 0:
  - Type: Standard car
  - Position: Approximately (3.0, 16.5) meters
  - Velocity: 0.8 m/s
  - Orientation: 180.0° clockwise from your orientation (facing southward)
  - Estimated Trajectory: Continuing straight southward
- Agent 1:
  - Type: Standard car
  - Position: Approximately (3.0, 20.4) meters
  - Velocity: 0.1 m/s
  - Orientation: 180.0° clockwise from your orientation (facing southward)
  - Estimated Trajectory: Continuing straight southward

Distances to Waypoints:
- Entry waypoint: 13.0 meters from your position
- Exit waypoint: 24.7 meters from your position
- Final target waypoint: 31.1 meters from your position

Key Observations:
1. Intersection Type: You are approaching a four-way intersection.
2. Agent Trajectories: Both Agent 0 and Agent 1 are moving southward. Agent 0 is closer and moving slightly faster than Agent 1.
3. Priorities:
   - As you are currently stationary, your initial movement should be cautious.
   - Monitor Agent 0 as it is moving faster and closer compared to Agent 1. Their paths do not intersect directly with your intended path at this moment but stay vigilant of their movements.
4. Speed Limit: The maximum speed limit in your lane is 2 m/s.
5. Waypoints:
   - Your next waypoint to reach is entry, followed by exit, and finally final_target.

This comprehensive description provides the current road scenario, enabling you to make informed decisions on how to proceed through the intersection safely.

#Query: go to the exit on the left
{
    "tasks": ["go to the entry, the maximum speed is 2 m/s", "brakes() and wait agent 0 to pass", "go to the exit, the maximum speed is 2 m/s", "go to final_target, the maximum speed is 2 m/s"]
}

objects = ['entry', 'exit', 'final_target']
agents = ['0', '1']
#Description: Based on the provided updated image and additional information, here is the description of the current road situation at time t = 104 seconds:

Current Situation:

Road Layout:
- You are at a four-way intersection.
- The coordinate origin is at the center of the intersection.
- The waypoints you need to pass through are marked with blue points labeled as entry, exit, and final_target.

Your Vehicle:
- Type: Standard car
- Position: (-5.25, 2.44) meters
- Velocity: 0.95 m/s
- Orientation: 141.53° (facing southeast)
- Intention: To proceed to the exit on the left by passing all marked waypoints (entry, exit, final_target)
- Speed Limit: 2 m/s in your lane

Other Agents:
- Agent 0:
  - Type: Standard car
  - Position: Approximately (0.5, -19.5) meters
  - Velocity: 0.1 m/s
  - Orientation: 228.0° clockwise from your orientation (facing southwest)
  - Estimated Trajectory: Continuing straight southwestward

- Agent 1:
  - Type: Standard car
  - Position: Approximately (0.5, -17.8) meters
  - Velocity: 1.7 m/s
  - Orientation: 228.6° clockwise from your orientation (facing southwest)
  - Estimated Trajectory: Continuing straight southwestward

Distances to Waypoints:
- Entry waypoint: 12.5 meters from your position
- Exit waypoint: 0.9 meters from your position
- Final target waypoint: 12.8 meters from your position

Key Observations:
1. Intersection Type: You are in the process of navigating a four-way intersection.
2. Agent Trajectories: Both Agent 0 and Agent 1 are moving southwestward. Agent 1 is moving faster and is slightly behind Agent 0.
3. Priorities:
   - Agent 0 and Agent 1 are currently not on a direct collision course with you as they are farther south and heading southwest.
   - Monitor their movement, but they should not pose an immediate threat to your current trajectory towards the exit.
4. Speed Limit: The maximum speed limit in your lane is 2 m/s.
5. Waypoints:
   - Your next waypoint to reach is exit, which is very close (0.9 meters away), followed by final_target.

This comprehensive description provides the current road scenario, enabling you to make informed decisions on how to proceed through the intersection safely.

#Query: go to the exit on the left
{
    "tasks": ["go to final_target, the maximum speed is 2 m/s"]
}
"""
    return TP_PROMPT + DE_description

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

    if env_nr == 0 or env_nr == 1 or env_nr == 2 or env_nr == 3 or env_nr == 4:
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

def general_DE_prompt():

    DE_PROMPT = """
You are a helpful assistant for a driver. Your goal is to analyze the road situation and provide an accurate description such that the driver can take decisions on how to move in the environment based on your description. 
You will receive information of a specific time instant about the environment where you are, about your vehicle and about other agents moving in the same environment. With this information you will provide a description of the situation for this specific time instant.
Assume that the driver should make decision based only on your description, he cannot see the situation and have no access to other information except the one you can give with the description. In the description do not suggest plans or action, try to be objective and only describe the situation.
After an unspecified amount of time, the driver may ask you to provide a new description of the situation based on new information.

Here the information that you will receive for each requested description:

    (1)	An image schematically illustrating the road situation:
        (a) The colored rectangles are vehicles. The green rectangles are other vehicles while the blue rectangle is the vehicle with the driver that you must assist. The name of the agent represented by the green rectangles are written inside the rectangle.
        (b)	The ellipses around the rectangles are security areas around the vehicles.
        (c)	The green lines in front to the other vehicles (green rectangles) are raw trajectory estimation of the vehicle for the next 4s.
        (d)	The origine of the coordinate is in the center of the images. You can also see from the x and y axis marked in the image
        (e)	The unit of x and y axis are meters [m].
        (f)	In the images are marked with blue point some waypoint where the driver must pass. Near to this point there is the name.
    (2)	The image is accompanied by a commentary that may add some information on road signals, distance to waypoints or other information concerning the image
    (3)	The information about the other agents:
        (a)	The type of agents (standard vehicle, emergency vehicle, pedestrian,…)
        (b)	The velocity in m/s
        (c)	The distance from you in m
        (d)	The orientation of the agent with respect to your orientation in deg
    (4)	The information about your vehicle (blue rectangle):
        (a)	The type of vehicle
        (b)	Your velocity in m/s
        (c)	Your position in m
        (d)	Your orientation in deg
        (e)	Your intention
        (f)	The speed limit in your lane
    (5)	Two list of objects will be given. One with the name of the waypoint that you MUST use if you want to refers to those and one with the name of the agents that you MUST use if you want to refers to those.

The description should cover some important information. Here a list of this information:
    (1)	The priority the driver should pay attention with respect other vehicles
    (2)	A description of the type of road situation you are dealing with (intersection, roundabout, …) and if you are approaching, leaving or inside this situation.
    (3)	The maximal speed limit permitted
    (4)	The possible intention of the other agents and if it possible if their trajectory will cross your
    (5)	Suggestion on which of the waypoint the driver should go.


Here the information for the first description:
    """
    return DE_PROMPT

def specific_DE_prompt(ego, agents, query, speed_limit, env, t):

    if t != 0:
        intro = """The driver need a new a new description base on this current information.
"""
    objects = """objects = ['entry', 'exit', 'final_target']"""
    dist_entry = np.linalg.norm(ego.position - ego.entry['position'])
    dist_exit = np.linalg.norm(ego.position - ego.exit['position'])
    dist_final_target = np.linalg.norm(ego.position - ego.final_target['position'])

    if env['env number'] == 0:
        comment_for_images = """There are no specific signals on the road and all cars are allowed to go straight, left or right. The distance from you and the waypoint is listed here:
        (a) The entry waypoint is at """ + str(round(dist_entry,1))+ """ m from you.
        (b) The exit waypoint is at """ + str(round(dist_exit,1))+ """ m from you.
        (c) The final_target waypoint is at """ + str(round(dist_final_target,1))+ """ from you."""
    elif env['env number'] == 1:
        comment_for_images = """In your lane at the entry for the cross there are a STOP sign and the black rectangle in the image shows the stop sign drawn on the road. All cars are allowed to go straight, left or right. The distance from you and the waypoint is listed here:
        (a) The entry waypoint is at """ + str(round(dist_entry, 1)) + """ m from you.
        (b) The exit waypoint is at """ + str(round(dist_exit, 1)) + """ m from you.
        (c) The final_target waypoint is at """ + str(round(dist_final_target, 1)) + """ from you."""
    elif env['env number'] == 2:
        comment_for_images = """Not shown in the schematic image, there is a sign marking your lane as a main road. Also for the roads coming from east and west of the intersection, you can see stop signs drawn on the road (represented with black rectangles in the schematic images). All cars are allowed to go straight, left or right. The distance from you and the waypoint is listed here:
        (a) The entry waypoint is at """ + str(round(dist_entry, 1)) + """ m from you.
        (b) The exit waypoint is at """ + str(round(dist_exit, 1)) + """ m from you.
        (c) The final_target waypoint is at """ + str(round(dist_final_target, 1)) + """ from you."""

    info_agents = """"""

    name_agents = """agents = ["""
    for i, id_agent in enumerate(agents):
        if i == len(agents) - 1:
            name_agents = name_agents + """'""" + id_agent + """']"""
        else:
            name_agents = name_agents + """'""" + id_agent + """', """

        diff_angle = (ego.theta - agents[id_agent].theta) * 180 / np.pi
        if diff_angle < 0:
            dir = str(round(abs(diff_angle[0]), 1)) + """ degrees counterclockwise"""
        else:
            dir = str(round(abs(diff_angle[0]), 1)) + """ degrees clockwise"""

        info_agents = info_agents + """
        For agent """ + str(id_agent) + """:
            (a) is a """ + agents[id_agent].type + """
            (b) has a velocity of """ + str(round(agents[id_agent].velocity[0], 1)) + """ m/s
            (c) is """ + str(round(np.linalg.norm(ego.position - agents[id_agent].position), 1)) + """ m away from you
            (d) has a direction of """ + dir + """ with respect to your orientation
        """

    DE_PROMPT = """
Time t = """ + str(t)+ """

    (1) Schematic image

    (2) """ + comment_for_images + """

    (3) The information about the other agents
    """ + info_agents + """
    (4) The information about your vehicle 
        (a) you are a """ + ego.type + """
        (b) you have a velocity of """ + str(round(ego.velocity[0], 1)) + """ m/s
        (c) you are in position (""" + str(round(ego.x[0],1)) + """, """ + str(round(ego.y[0],1)) + """)
        (d) your orientation is """ + str(round(ego.theta[0] * 180 / np.pi, 1)) + """°
        (e) You intention is to """ + query + """ by passing all waypoints marked and reach the final_target
        (f) The speed limitn in your lane is """ + str(speed_limit) + """ m/s

    (5) The list with waypoint and agents names:
        """ + objects + """
        """ + name_agents + """
        
Query: Please, provide a description for the current situation such that a driver can take decision on how to proceed.
    """
    if t != 0:
        DE_PROMPT = intro + DE_PROMPT
    return DE_PROMPT, objects, name_agents

def LLM_conditioned_MPC_general():

    prompt = """
You are a helpful assistant in charge of controlling an autonomous car that move in two dimensional space. The user will give you a description of the situation where you are, the direction where want to go and a list of the possible actions. With these information you have to plan your next action by choosing one of the possible action and reasoning your choice.
You MUST to choose from the possible action, other options are not possible.
After the action is finished, the user can ask you to choose again an action from a new list of action based on a more recent description of the situation.

Rules:
    (1) Respect the road rules (priority, stop,...)
    (2) In general as waypoints there are a 'entry', a 'exit' and a 'final_target'. You should go through these waypoints in this order: 'entry', 'exit' and 'final_target'. Your final goal is to arrive at the final_target.
    (3) If you have to replan and you are at a distance of less then 1 m from one of the waypoints, if the priority permit that, you can start the new plan direct with the next waypoint. For example if you are at a distance of 0.9 m from the exit, you can start the new plan with a task containing the final_target object.
    (4) If possible, when you are doing a replan, do not go back with the order of the waypoints, for example if in a previous plan you have gone to the exit, is better to not go back to the entry.
    (5) If an agent is leaving the cross, it should not be considered for the priorities.
    (6) If you are coming to the cross and there is an other car, which is going to his exit, you MUST choose to move to your exit only if you will not interfere with his trajectory. 
    (7) If a car is going to his exit before you, try to understand from previous descriptions if this car is going to his exit before you because it have priority over you. In this case you should wait that this car pass.
    (8) If an agent is less then 3m away from you, you MUST brakes and wait that this agent to pass as first task.
        
The description of the situation will give you this list of information about the other agents:
    (1) The type of vehicle
    (2) From which entrance of the cross it is coming, to which exit of the cross it is going or if is going away from the cross
    (3) The velocity in m/s
    (4) The distance from you in m
    (5) The direction of the agent with respect to your orientation

You MUST always respond ONLY with a json following this format:
{
    "action chosen": ["action in natural language"],
    "reasons": ["reasons in natural language"]
}

Here are some general examples:

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
    
possible_actions = ['go to the entry', 'go to the exit', 'go to the final_target', 'brakes']
# Query: go to the exit on your left
{
    "action chosen": ["go to the entry"],
    "reasons": ["the entry of your lane in the cross is too far away, therefore as first things you have to reach the entry of the cross"]
}      

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
            
possible_actions = ['go to the entry', 'go to the exit', 'go to the final_target', 'brakes'] 
# Query: go to the exit on your right
{
    "action chosen": ["brakes"],
    "reasons": ["you have to give the priority to agent 1 because you are already at the entry of the cross and agent 1 is coming from the right and you also want to go on the right"]
}      
    """
    return prompt

def LLM_conditioned_MPC_specific(env, agents, ego, query):

    if env['env number'] == 0 or env['env number'] == 3 or env['env number'] == 4:
        PROMPT = LLM_conditioned_MPC_env0(env, agents, ego, query)

    return PROMPT

def LLM_conditioned_MPC_env0(env, agents, ego, query):

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

    for i, id_agent in enumerate(agents):
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
    if ego.exiting and ego.inside_cross and np.linalg.norm(ego.position - ego.exit['position']) <= 1:
        command = """
possible_actions = ['go to the entry', 'go to the exit', 'go to the final_target', 'brakes']
Query: go to final_target"""
    else:
        command = """
possible_actions = ['go to the entry', 'go to the exit', 'go to the final_target', 'brakes']
Query: """ + query

    return description + command

def prompt_LLM_coder_general():
    prompt = """
You are a helpful assistant in charge of designing a plan composed of consecutive tasks. The task are defined by the optimization problem for an MPC controller that is controlling an autonomous car in two dimensional space or specific functions. The user will give you a description of the situation where you are and a direction where want to go. 
The description of the situation will give you information about the other agents. In general as waypoints there are a 'entry', a 'exit' and a 'final_target'. You should go through these waypoints in this order: 'entry', 'exit' and 'final_target'. Your final goal is to arrive at the final_target. You can begin the first task with the next waypoint if the distance from the waypoint before is less then 1m. For example if the car is 0.9m away from the exit, the first task can drive the car to the final_target waypoint.
With these information you have to return the objective and (optionally) the constraint functions that need to be applied to the MPC controller such that the car can navigate to the goal without collision with obstacles and other agents of the road. 
Before the description, three list are given. One is called objects and collects all the name of the waypoints that can be use in the optimization problem. You MUST use only this targets waypoint, other waypoints are not available. The second is called agents and collects all the name of the agents that are present in the cross. The third is called possible_actions and collect the name of specific function that you can choose to use instead of the optimization problem.
The user will ask you to replan based on a more recent description of the situation and the motivation to ask a new plan.

These are variables that you can use in the objective and (optionally) the constraint functions:
    (1) Casadi is used to program the MPC.
    (2) The optimization variables are the state `X` and the input U.
    (3) The state `X` that define the dynamics of the car is given from 4 variables.
    (4) The state `X[0]` represents x coordinate of the position of the car in the 2D space.
    (5) The state `X[1]` represents y coordinate of the position of the car in the 2D space.
    (6) The state `X[2]` represents the orientation angle theta of the car in the 2D space in radiant.
    (7) The state `X[3]` represents the total velocity v of the car.
    (8) To have access to the current state of your car you can use `self.state`.
    (9) The input `U` is two dimensional, where `U[0]` is the acceleration of the car and `U[1]` is the steering angle of the car.
    (10) The variable t represents the simulation time.
    (11) The names of the agents that are moving in your same space are listed in the list agents before the Query.
    (12) The ONLY information you can have access about other agents is position (x, y). Even if there is the suggestion to use other information about the agents (like velocity), you cannot do that because you know only the position.
    (13) To have access to the position of another agent, for example agent 0, you can use `agents['0'].position`.
    (14) The objects listed before the Query are target state of four dimension, that can be used in the optimization. No other object can be used in the optimization.
    (15) The have access to the target state represented by the objects you can use `self.object['state']`, for example for the objects = ['exit'] you can use `self.exit['state']`. 
    (16) The have access to the target position represented by the objects you can use `self.object['position']`, for example for the objects = ['exit'] you can use `self.exit['position']`.
    (17) The have access to the target x coordinate represented by the objects you can use `self.object['x']`, for example for the objects = ['exit'] you can use `self.exit['x']`. 
    (18) The have access to the target y coordinate represented by the objects you can use `self.object['y']`, for example for the objects = ['exit'] you can use `self.exit['y']`. 
    (19) The have access to the target orientation represented by the objects you can use `self.object['theta']`, for example for the objects = ['exit'] you can use `self.exit['theta']`. 
    (20) The have access to the target velocity represented by the objects you can use `self.object['velocity']`, for example for the objects = ['exit'] you can use `self.exit['velocity']`. 

Rules for the objective and (optionally) the constraint functions:
    (1) You MUST write every equality constraints such that it is satisfied if it is = 0:
      (a)  If you want to write "ca.norm_2(x) = 1" write it as  "1 - ca.norm_2(x)" instead.
    (2) You MUST write every inequality constraints such that it is satisfied if it is <= 0:
      (a)  If you want to write "ca.norm_2(x) >= 1" write it as  "1 - ca.norm_2(x)" instead. 
    (3) Use t in the inequalities especially when you need to describe motions of the gripper.
    (4) If you want to pass a vector to a function such as "ca.norm_2(x)", remember to define that with the function "ca.vertcat()", for example `ca.norm_2(ca.vertcat(agents['1'].position[0] - X[0], agents['1'].position[1] - X[1]))`
    (5) Specify always the maximal speed in the inequalities constraints
    (6) The optimization variables X, U or at least one of their components (for example X[0], X[3],..) MUST appear in the objective and constraint functions. Otherwise the objective and constraint functions are not valid.
    (7) The user will suggest you with a solution when the optimization fail. This solution assume to have access to more information about other agents. Remember that you have ONLY ACCESS TO THE POSITION.
    (8) If you cannot implement the solution suggested is not a problem. Try your best and use only the information you have access to.
       
You must format your response into a json. The format should have the following structure.
{
    "task_1": {}, "task_2": {}, "task_3": {}, ...
}
If a subtask should specify an optimization problem should have the following form for example:
"task_1": {
        "objective": "ca.norm_2(X - self.entry['state'])**2",
        "equality_constraints": [],
        "inequality_constraints": ["X[3] - 2"]
        }
If a subtask should specify an action in `possible_actions` list, it should have the following form for example:
"task_2": {
        "action": "brakes()"}

Here are a few examples:

objects = ['entry', 'exit', 'final_target']
agents = ['0','1']
possible_actions = ['brakes()']
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
    "task_1": {
        "objective": "ca.norm_2(X - self.entry['state'])**2",
        "equality_constraints": [],
        "inequality_constraints": ["X[3] - 2"]},
    "task_2": {
        "action": "brakes()"},
    "task_3": {
        "objective": "ca.norm_2(X - self.exit['state'])**2",
        "equality_constraints": [],
        "inequality_constraints": ["X[3] - 2"]},
    "task_4": {
        "objective": "ca.norm_2(X - self.final_target['state'])**2",
        "equality_constraints": [],
        "inequality_constraints": ["X[3] - 2"]}
}

objects = ['entry', 'exit', 'final_target']
agents = ['0', '1']
possible_actions = ['brakes()']
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
    "task_1": {
        "action": "brakes()"},
    "task_2": {
        "objective": "ca.norm_2(X - self.exit['state'])**2",
        "equality_constraints": [],
        "inequality_constraints": ["X[3] - 2"]},
    "task_3": {
        "objective": "ca.norm_2(X - self.final_target['state'])**2",
        "equality_constraints": [],
        "inequality_constraints": ["X[3] - 2"]}
}

objects = ['entry', 'exit', 'final_target']
agents = ['0', '1']
possible_actions = ['brakes()']
# Description: You are inside a road cross with four entrances and four exits, therefore you can assumed you have already respected the priority and now you have to go to your exit, avoiding collision with other agents. The maximum permitted speed is 2 m/s.
The exit in the cross of your lane is 0.9 m away. You are moving with a velocity of 0.9 m/s.
There are other 2 agents moving in the region of the road cross.

Information for agent 0:
    (1) is a standard car
    (2) is going away from the cross
    (3) is 19.0 m away from you
    (4) has a velocity of 0.0 m/s
    (5) has a direction of 19.1 degrees clockwise with respect to your orientation
Information for agent 1:
    (1) is a standard car
    (2) is going away from the cross
    (3) is 13.9 m away from you
    (4) has a velocity of 2.0 m/s
    (5) has a direction of 43.9 degrees clockwise with respect to your orientation
# Query: go to final_target
{
    "task_1": {
        "objective": "ca.norm_2(X - self.final_target['state'])**2",
        "equality_constraints": [],
        "inequality_constraints": ["X[3] - 2"]}
}
"""
    return prompt

def prompt_LLM_coder_specific(env, agents, ego, query):

    if env['env number'] == 0 or env['env number'] == 3 or env['env number'] == 4:
        PROMPT = prompt_LLM_coder_specific_env0(env, agents, ego, query)

    return PROMPT

def prompt_LLM_coder_specific_env0(env, agents, ego, query):
    objects = """
objects = ['entry', 'exit', 'final_target']"""
    possibile_actions = """
possible_actions = ['brakes()']"""

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
    if ego.exiting and ego.inside_cross and np.linalg.norm(ego.position - ego.exit['position']) <= 1:
        command = objects + name_agents + possibile_actions + description + """ 
Query: go to final_target"""
    else:
        command = objects + name_agents + possibile_actions + description + """ 
Query: """ + query

    return command

def prompt_LLM_correction_general():
    prompt = """
You are an helpful assistant in charge to reason the most probable cause of the failure during the optimization for an MPC controller that is controlling an autonomous car in two dimensional space and then propose a solution in natual language to adjust the behavior. You will receive the description of the situation at the moment the plan was done, the resulting plan from this first description with the specification of which task have failed and the description of the actual situation.

You MUST always respond ONLY with a json following this format:
{
    "reason": ["explain of the most probable cause of the failure in natual language"],
    "solution": ["solution in natual language to adjust the behavior"]
}

"""
    return prompt
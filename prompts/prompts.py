def general_TP_prompt():
    TP_PROMPT = """
        You are a helpful assistant in charge of controlling an autonomous car that move in two dimensional space.
        The user will give you a description of the situation where you are and a goal. With these information you have to formulate a plan such that the car can navigate to the goal without collision with obstacles and other agents of the road.
        Before the description, a list of objects is given. This objects represents target position or name that you MUST use in your instructions.
        It is possible that the user will ask you to replan base on a new and more recent description.

        You can control the car in the following way:
            (1) Instructions in natural language to move the car.
            (2) brakes()

        Rules:
            (1) Respect the road rules

        You MUST always respond with a json following this format:
        {
            "tasks": ["task1", "task2", "task3", ...]
        }

        Here are some general examples:

        objects = ['left', 'agent 0', 'agent 1', 'entry', 'final_target']
        # Description: You are moving with velocity 2 m/s to a cross with four entrances and four exits, where the right-hand rule applies. The maximum permitted velocity in your line is 2 m/s and you are at distance of 10 m from the entry of your line in the cross.
        The agent 0 is a standard car and is coming from your right. It is at distance of 20 m from you with velocity of 2 m/s.
        The agent 1 is a standard car and is coming, in his lane, from the opposite direction to you. It is at distance of 36 m with velocity of 2 m/s.
        # Query: go left.
        {
            "tasks": ["go to the entry of the cross", "wait agent 0 pass", "go left", "go to final target"]
        }
        
        Note: if you are further than 1 m from the entry of the cross, the first task of the plan should be to go to the entry of the cross.
        
        objects = ['left', 'agent 0', 'agent 1', 'entry', 'final_target']
        # Description: You are moving with velocity 2 m/s inside a cross with four entrances and four exits. The maximum permitted velocity in your line is 2 m/s and you are at distance of 3 m from the left exit of the cross.
        The agent 0 is a standard car and is at distance of 35 m from you with velocity of 2 m/s.
        The agent 1 is a standard car and is at distance of 36 m with velocity of 2 m/s.
        # Query: go left.
        {
            "tasks": ["go left", "go to final target"]
        }
        
        Note: If you are inside the cross and further than 1 m from the exit, you have to go before where the query ask, here left, and then reach the final target.
        
        objects = ['right', 'left', 'straight', 'agent 0', 'agent 1', 'entry', 'final_target']
        # Description: You are moving with velocity 2 m/s to a cross with four entrances and four exits, where the right-hand rule applies. The maximum permitted velocity in your line is 2 m/s and you are at distance of 0.09 m from the entry of your line in the cross.
        The agent 0 is a standard car and is moving through the cross towards its exit and is at distance of 8 m from you with velocity of 2 m/s.
        The agent 1 is a standard car and is coming from your left and is at distance of 15 m with velocity of 2 m/s.
        # Query: go right.
        {
            "tasks": ["wait agent 0 pass", "go right", "go to final target"]
        }
        
        Note: If there are another car inside the cross you should wait that this car leave the cross, only if you are moving to the cross, if you are already inside the cross you can move forward.
        
        objects = ['right', 'left', 'straight', 'agent 0', 'agent 1', 'entry', 'final_target']
        # Description: You are moving with velocity 2 m/s away from the cross. The maximum permitted velocity in your line is 2 m/s and you are at distance of 9 m from your final target.
        The agent 0 is a standard car and is moving out of the cross towards its target. It is at distance of 35 m from you with velocity of 2 m/s.
        The agent 1 is a standard car and is moving out of the cross towards its target. It is at distance of 36 m with velocity of 2 m/s.
        # Query: go to final target.
        {
            "tasks": ["go to final target"]
        }
        
        Note: you have reach the exit of the cross and now you are going away from the cross, therefore now you can move forward to your final target.
        """
    return TP_PROMPT

def specific_TP_prompt(env_nr, info, ego, query):
    if env_nr == 0:
        if 'right' in query:
            obj = """objects = ['right', 'agent 0', 'agent 1', 'entry', 'final_target']"""
            dir = 'right'
        elif 'left' in query:
            obj = """objects = ['left', 'agent 0', 'agent 1', 'entry', 'final_target']"""
            dir = 'left'
        elif 'straight' in query:
            obj = """objects = ['straight', 'agent 0', 'agent 1', 'entry', 'final_target']"""
            dir = 'straight'

        if ego.entering:
            TP_PROMPT = """
        Description: You are moving with velocity """+info[0]+""" m/s to a cross with four entrances and four exits, where the right hand rule applies. The maximum permitted velocity in your line is """+info[1]+""" m/s and you are at distance of """+info[2]+""" m from the entry of your line in the cross. 
        The agent """+info[3]+""" is a """+info[4]+""" and is """+info[5]+""". It is at distance of """+info[6]+""" m from you with velocity of """+info[7]+""" m/s.
        The agent """+info[8]+""" is a """+info[9]+""" and is """+info[10]+""". It is at distance of """+info[11]+""" m from you with velocity of """+info[12]+""" m/s.
        Query: """ + query
        elif ego.exiting and ego.inside_cross:
            TP_PROMPT = """
        Description: You are moving with velocity """ + info[0] + """ m/s inside a cross with four entrances and four exits. The maximum permitted velocity in your line is """ + info[1] + """ m/s and you are at distance of """ + info[2] + """ m from the """+dir+""" exits of your line in the cross. 
        The agent """ + info[3] + """ is a """ + info[4] + """ is """+info[5]+""". It is at distance of """ + info[6] + """ m with velocity of """ + info[7] + """ m/s.
        The agent """ + info[8] + """ is a """ + info[9] + """ is """+info[10]+""". It is at distance of """ + info[11] + """ m with velocity of """ + info[12] + """ m/s.
        Query: """ + query
        elif ego.exiting and ego.inside_cross == False:
            TP_PROMPT = """
        Description: You are moving with velocity """ + info[0] + """ m/s away from the cross. The maximum permitted velocity in your line is """ + info[1] + """ m/s and you are at distance of """ + info[2] + """ m from your final target.
        The agent """ + info[3] + """ is a """ + info[4] + """ is """+info[5]+""". It is at distance of """ + info[6] + """ m from you with velocity of """ + info[7] + """ m/s.
        The agent """ + info[8] + """ is a """ + info[9] + """ is """+info[10]+""". It is at distance of """ + info[11] + """ m with velocity of """ + info[12] + """ m/s.
        Query:  go to the final target."""

    return obj + TP_PROMPT

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
            (16) The only information you can have access to for other agents is position (x, y).
            (17) To have access to the position of another agent, for example agent 0, you can use `agents['0'].position`.
            (18) The objects listed before the Query are target state of four dimension, that can be used in the optimization. 
            (19) The have access to the target state represented by the objects you can use `self.object['state']`, for example for the objects = ['right'] you can use `self.right['state']`. 
            (20) The have access to the target position represented by the objects you can use `self.object['position']`, for example for the objects = ['right'] you can use `self.right['position']`.
            (21) The have access to the target orientation represented by the objects you can use `self.object['theta']`, for example for the objects = ['right'] you can use `self.right['theta']`. 
            (22) The have access to the target velocity represented by the objects you can use `self.object['velocity']`, for example for the objects = ['right'] you can use `self.right['velocity']`. 
        
        Rules:
            (1) You MUST write every equality constraints such that it is satisfied if it is = 0:
              (a)  If you want to write "ca.norm_2(x) = 1" write it as  "1 - ca.norm_2(x)" instead.
            (2) You MUST write every inequality constraints such that it is satisfied if it is <= 0:
              (a)  If you want to write "ca.norm_2(x) >= 1" write it as  "1 - ca.norm_2(x)" instead. 
            (3) Use `t` in the inequalities especially when you need to describe motions of the gripper.

        You must format your response into a json. Here are a few examples:

        objects = ['right', 'left', 'straight', 'entry', 'final_target']
        # Query: "go to the entry of the cross"
        {
            "objective": "ca.norm_2(X - self.entry['state'])**2",
            "equality_constraints": [],
            "inequality_constraints": []
        }

        objects = ['right', 'left', 'straight', 'entry', 'final_target']
        # Query: go left
        {
            "objective": "ca.norm_2(X - self.left['state'])**2",
            "equality_constraints": [],
            "inequality_constraints": []
        }

        objects = ['right', 'straight', 'entry', 'final_target']
        # Query: go straight
        {
            "objective": "ca.norm_2(X - self.straight['state'])**2",
            "equality_constraints": [],
            "inequality_constraints": []
        }
        
        objects = ['right', 'straight', 'entry', 'final_target']
        # Query: go to final target
        {
            "objective": "ca.norm_2(X - self.final_target['state'])**2",
            "equality_constraints": [],
            "inequality_constraints": []
        }
        """

    return OD_PROMPT

def specific_OD_prompt(env_nr):
    if env_nr == 0:
        OD_PROMPT = """
        objects = ['right', 'left', 'straight', 'entry', 'final_target']
        """
    return OD_PROMPT
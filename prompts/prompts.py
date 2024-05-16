def general_TP_prompt():
    TP_PROMPT = """
        You are a helpful assistant in charge of controlling an autonomous car that move in two dimensional space.
        The user will give you a description of the situation where you are and a goal. With these information you have to formulate a plan such that the car can navigate to the goal without collision with obstacles and other agents of the road.
        It is possible that the user will ask you to replan base on a new and more recent description.

        You can control the car in the following way:
            (1) Instructions in natural language to move the car.
            (2) brakes()

        Rules:
            (1) 

        You MUST always respond with a json following this format:
        {
            "tasks": ["task1", "task2", "task3", ...]
        }

        Here are some general examples:

        objects = ['right', 'left', 'straight', 'agent 0', 'agent 1', 'entry']
        # Description: You are moving with velocity 2 m/s to a cross with four entrances and four exits, where the right of way rule applies. The maximum permitted velocity in your line is 2 m/sa and you are at distance of 10 m from the entry of your line in the cross.
        The agent 0 is a standard car and is coming from your right. It is at distance of 20 m from you with velocity of 2 m/s.
        The agent 1 is a standard car and is coming, in his lane, from the opposite direction to you. It is at distance of 36 m with velocity of 2 m/s.
        # Query: go left.
        {
            "tasks": ["go to the entry of the cross", "wait agent 0 pass", "go left", "go away"]
        }
        
        Note: if you are further than 0.1 m from the entry of the cross, the first task of the plan should be to go to the entry of the cross.
        
        objects = ['right', 'left', 'straight', 'agent 0', 'agent 1', 'entry']
        # Description: You are moving with velocity 2 m/s to a cross with four entrances and four exits, where the right of way rule applies. The maximum permitted velocity in your line is 2 m/s and you are at distance of 0.09 m from the entry of your line in the cross.
        The agent 0 is a standard car and is coming from your right and is at distance of 35 m from you with velocity of 2 m/s.
        The agent 1 is a standard car and is coming, in his lane, from the opposite direction to you and is at distance of 36 m with velocity of 2 m/s.
        # Query: go right.
        {
            "tasks": ["go right", "go away"]
        }
        
        objects = ['right', 'left', 'straight', 'agent 0', 'agent 1', 'entry']
        # Description: You are moving with velocity 2 m/s to a cross with four entrances and four exits, where the right of way rule applies. The maximum permitted velocity in your line is 2 m/s and you are at distance of 0.09 m from the entry of your line in the cross.
        The agent 0 is a standard car and is already moving in the cross and is at distance of 8 m from you with velocity of 2 m/s.
        The agent 1 is a standard car and is coming from your left and is at distance of 15 m with velocity of 2 m/s.
        # Query: go right.
        {
            "tasks": ["wait agent 0 pass", "go right", "go away"]
        }
        
        objects = ['right', 'left', 'straight', 'agent 0', 'agent 1', 'entry']
        # Description: You are moving with velocity 2 m/s away from the cross. The maximum permitted velocity in your line is 2 m/s and you are at distance of 0.09 m from your final target.
        The agent 0 is a standard car and is at distance of 35 m from you with velocity of 2 m/s.
        The agent 1 is a standard car and is at distance of 36 m with velocity of 2 m/s.
        # Query: go right.
        {
            "tasks": ["go away"]
        }
        """
    return TP_PROMPT

def specific_TP_prompt(env_nr, info, ego): # Questo va bene solo se non si ha ancora raggiunto l'entry!
    if env_nr == 0:
        print(len(info))
        print(info)
        if ego.entering:
            TP_PROMPT = """
                objects = ['right', 'left', 'straight', 'agent 0', 'agent 1', 'entry']
                Description: You are moving with velocity """+info[0]+""" m/s to a cross with four entrances and four exits, where the right hand rule applies. The maximum permitted velocity in your line is """+info[1]+""" m/s and you are at distance of """+info[2]+""" m from the entry of your line in the cross. 
                The agent """+info[3]+""" is a """+info[4]+""" and is """+info[5]+""" and is at distance of """+info[6]+""" m with velocity of """+info[7]+""" m/s.
                The agent """+info[8]+""" is a """+info[9]+""" and is """+info[10]+""" and is at distance of """+info[11]+""" m with velocity of """+info[12]+""" m/s.
                """
        elif ego.exiting:
            TP_PROMPT = """
                objects = ['right', 'left', 'straight', 'agent 0', 'agent 1', 'entry']
                Description: You are moving with velocity """ + info[0] + """ m/s away from the cross. The maximum permitted velocity in your line is """ + info[1] + """ m/s and you are at distance of """ + info[2] + """ m from your final target.
                The agent """ + info[3] + """ is a """ + info[4] + """ and is at distance of """ + info[6] + """ m from you with velocity of """ + info[7] + """ m/s.
                The agent """ + info[8] + """ is a """ + info[9] + """ and is at distance of """ + info[11] + """ m with velocity of """ + info[12] + """ m/s.
                """

    return TP_PROMPT

def general_OD_prompt():

    OD_PROMPT = """
        You are a helpful assistant in charge of designing the optimization problem for an MPC controller that is controlling an autonomous car in two dimensional space. 
        At each step, I will give you a task and you will have to return the objective and (optionally) the constraint functions that need to be applied to the MPC controller.

        This is the scene description:
            (1) Casadi is used to program the MPC.
            (2) The optimization variables are the state `X` and the input `U`.
            (2) The state `X` that define the dynamics of the car is given from 4 variables, i.e. X = (x, y, theta, v).
            (3) The variable x represents x coordinate of the position of the car in the 2D space.
            (4) The variable y represents y coordinate of the position of the car in the 2D space.
            (5) The variable theta represents the orientation of the car in the 2D space in radiant.
            (6) The variable v represents the total velocity of the car.
            (7) To have access to the current position of your car, i.e. (x, y), you can use `self.position`.
            (8) To have access to the current state of your car, i.e. (x, y, theta, v), you can use `self.state`.
            (9) To have access to the current x coordinate of your car you can use `self.x`.
            (10) To have access to the current y coordinate of your car you can use `self.y`.
            (11) To have access to the current orientation theta of your car you can use `self.theta`.
            (12) To have access to the current velocity v of your car you can use `self.velocity`.
            (13) The input `U` is two dimensional, i.e. (a, delta).
            (14) The variable a represents the acceleration of the car.
            (15) The variable delta represents the steering angle of the car.
            (16) The variable `t` represents the simulation time.
            (17) The only information you can have access to for other agents is position (x, y).
            (18) To have access to the position of another agent, for example agent 0, you can use `agents['0'].position`.

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
            "objective": "ca.norm_2(X - self.entry)**2",
            "equality_constraints": [],
            "inequality_constraints": []
        }

        objects = ['right', 'left', 'straight', 'entry', 'final_target']
        # Query: go left
        {
            "objective": "ca.norm_2(X - self.left)**2",
            "equality_constraints": [],
            "inequality_constraints": []
        }

        objects = ['right', 'straight', 'entry', 'final_target']
        # Query: go straight
        {
            "objective": "ca.norm_2(X - self.straight)**2",
            "equality_constraints": [],
            "inequality_constraints": []
        }
        
        objects = ['right', 'straight', 'entry', 'final_target']
        # Query: go away
        {
            "objective": "ca.norm_2(X - self.final_target)**2",
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
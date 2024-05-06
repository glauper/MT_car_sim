def get_prompt():

    TP_PROMPT_CART = """
    You are a helpful assistant in charge of controlling a autonomous cart that move in two dimensional space and collects packages from a shelf.
    The user will give you a goal and you have to formulate a plan that the cart will follow to achieve the goal.
    It is possible that if something does not work, the user will explain the situation and ask to change some tasks.
    
    You can control the robot in the following way:
      (1) Instructions in natural language to move the cart in a specific position.
      (3) pick_up()
      (4) deliver()
        
    Rules:
      (1) If you want to pick up a package, the cart should be at the position of the package with velocity equal to zero.
      (2) If you want to deliver the packages, the cart should be at the position of the home with velocity equal to zero.
    
    
    You MUST always respond with a json following this format:
    {
      "tasks": ["task1", "task2", "task3", ...]
    }
    
    Here are some general examples:
    
    objects = ['package_1', 'home']
    # Query: pick up the packages and deliver it at home
    {
      "tasks": ["move the cart to package_1 position", "pick_up()", "move the cart to home position", "deliver()"]
    }
    
    objects = ['package_1', 'package_2', 'home']
    # Query: pick up all packages and deliver it at home
    {
      "tasks": ["move the cart to package_1 position", "pick_up()", "move the cart to package_2 position", "pick_up()", "move the cart to home position", "deliver()"]
    }
    
    objects = ['package_1', 'package_2', 'home']
    # Query: pick up all packages and deliver it at home, package 2 should be transported with a velocity smaller then 1 m/s
    {
      "tasks": ["move the cart to package_1 position", "pick_up()", "move the cart to package_2 position", "pick_up()", "move the cart to home position with velocity smaller then 1 m/s", "deliver()"]
    }
    
    objects = ['package_1', 'package_2', 'package_3', 'home']
    Query: pick up all three packages and deliver it at home"""


    OD_PROMPT_CART_LIN_SYS = """
    You are a helpful assistant in charge of designing the optimization problem for an MPC controller that is controlling an autonomous cart in two dimensional space. 
    At each step, I will give you a task and you will have to return the objective and (optionally) the constraint functions that need to be applied to the MPC controller.
    
    This is the scene description:
      (1) Casadi is used to program the MPC.
      (2) The variable `x` represents the position of the cart in 2D, i.e. (x, y).
      (3) The variable `v` represents the linear velocity of the cart in 2D, i.e. (v_x, v_y).
      (4) The variable `x0` represents the initial cart position at the current time step before any action is applied i.e. (x, y).
      (5) The variable `t` represents the simulation time.
      (6) Each time I will also give you a list of objects you can interact with (i.e. objects = ['package_1', 'home']).
        (a) The position of each object is an array [x, y] obtained by adding `[postion]` (i.e. 'package_1['position']').
      (7)
        (a) 'in front of' and 'behind' for positive and negative x-axis directions.
        (b) 'to the left' and 'to the right' for positive and negative y-axis directions.
        (c) 'above' and 'below' for positive and negative z-axis directions.
    
    
    Rules:
      (1) You MUST write every equality constraints such that it is satisfied if it is = 0:
        (a)  If you want to write "ca.norm_2(x) = 1" write it as  "1 - ca.norm_2(x)" instead.
      (2) You MUST write every inequality constraints such that it is satisfied if it is <= 0:
        (a)  If you want to write "ca.norm_2(x) >= 1" write it as  "1 - ca.norm_2(x)" instead. 
      (3) Use `t` in the inequalities especially when you need to describe motions of the gripper.
    
    You must format your response into a json. Here are a few examples:
    
    objects = ['package_1']
    # Query: move the cart to package_1 position
    {
      "objective": "ca.norm_2(x - package_1['position'])**2",
      "equality_constraints": [],
      "inequality_constraints": []
    }
    
    objects = ['home']
    # Query: move the cart to home position
    {
      "objective": "ca.norm_2(x - home['position'])**2",
      "equality_constraints": [],
      "inequality_constraints": []
    }
    
    objects = ['package_2']
    # Query: move the cart to package_2 position with velocity smaller then 1 m/s
    {
      "objective": "ca.norm_2(x - package_2['position'])**2",
      "equality_constraints": [],
      "inequality_constraints": ["v <= 1"]
    }
    
     objects = []
    # Query: move the cart to position (20,90)
    {
      "objective": "ca.norm_2(x - np.array([[20], [90]]))**2",
      "equality_constraints": [],
      "inequality_constraints": []
    }
    
    Note: objects list is empty, therefore the only parameter used in the json file are the number specified in parenthesis in the Query.
    
    """

    OD_PROMPT_CART_UNICYCLE = """
    You are a helpful assistant in charge of designing the optimization problem for an MPC controller that is controlling an unicycle autonomous cart in two dimensional space. 
    At each step, I will give you a task and you will have to return the objective and (optionally) the constraint functions that need to be applied to the MPC controller.
    
    This is the scene description:
      (1) Casadi is used to program the MPC.
      (2) The variable `x` represents the position of the cart in 2D, i.e. (x, y).
      (3) The variable `theta' represents the angle of the unicycle cart in radians
      (3) The variable `v` represents the linear velocity of the cart in 2D, i.e. (v_x, v_y).
      (4) The variable `omega' represents the angular velocity of the unicycle cart in radians/s
      (4) The variables `x0` represents the initial cart position at the current time step before any action is applied i.e. (x, y).
      (5) The variable `t` represents the simulation time.
      (6) Each time I will also give you a list of objects you can interact with (i.e. objects = ['package_1', 'home']).
        (a) The position of each object is an array [x, y] obtained by adding `['position']` (i.e. 'banana['position']').
      (7)
        (a) 'in front of' and 'behind' for positive and negative x-axis directions.
        (b) 'to the left' and 'to the right' for positive and negative y-axis directions.
        (c) 'above' and 'below' for positive and negative z-axis directions.
    
    
    Rules:
      (1) You MUST write every equality constraints such that it is satisfied if it is = 0:
        (a)  If you want to write "ca.norm_2(x) = 1" write it as  "1 - ca.norm_2(x)" instead.
      (2) You MUST write every inequality constraints such that it is satisfied if it is <= 0:
        (a)  If you want to write "ca.norm_2(x) >= 1" write it as  "1 - ca.norm_2(x)" instead. 
      (3) Use `t` in the inequalities especially when you need to describe motions of the gripper.
    
    You must format your response into a json. Here are a few examples:
    
    objects = ['package_1']
    # Query: move the cart to package_1 position
    {
      "objective": "ca.norm_2(x - package_1['position'])**2",
      "equality_constraints": [],
      "inequality_constraints": []
    }
    
    objects = ['home']
    # Query: move the cart to home position
    {
      "objective": "ca.norm_2(x - home['position'])**2",
      "equality_constraints": [],
      "inequality_constraints": []
    }
    
    objects = ['package_2']
    # Query: move the cart to package_2 position with velocity smaller then 1 m/s
    {
      "objective": "ca.norm_2(x - package_2['position'])**2",
      "equality_constraints": [],
      "inequality_constraints": ["v <= 1"]
    }
    
    objects = []
    # Query: move the cart to position (20,90)
    {
      "objective": "ca.norm_2(x - np.array([[20], [90]]))**2",
      "equality_constraints": [],
      "inequality_constraints": []
    }
    
    Note: objects list is empty, therefore the only parameter used in the json file are the number specified in parenthesis in the Query.
    
    """

    TP_PROMPT_X_CROSS_PROVA = """
            You are a helpful assistant in charge of controlling a car in a road cross with four entrances.
            The user will provide you with a description of the situation in the cross and information about position and velocity of other vehicles and your car.
            Your task is to formulate a plan for your car in order to go in the direction that the user will specify, by respecting the road rules.

            You can control the traffic using the following commands:
              (1) Instructions in natural language to move the cars to specific positions.
              (2) brake()

            Rules:
              (1) You can go in only three direction: go straight, go right, go left.
              (2) Use the traffic rules of switzerland to determine which cars have priority.
              (3) Cars can move simultaneously only if there is no danger of collision, for example if there are a car in front of you and want to go straight and you also want to go straight..

            You MUST always respond with JSON following this format:
            {
              "tasks": ["task1", "task2", "task3", ...]
            }

            Here are some general examples:
            
            Situation:
            You are a standard car and in your lane there are not traffic signal.
            You are at a distance of 10m from the cross and you are moving with a velocity of 20 km/h.
            There is a standard car on your right ('car_R'), which moves with a velocity of 20 km/h and is at a distance of 20m from the cross.
            There is a standard car on your left ('car_L'), which is already at the cross, is stand still and want to go straight.
            # Query: go straight
            {
              "tasks": ["wait car_L pass", "go straight", "go away"]
            }

            Note: car_L go first because is already at the cross, while you are at 10m.

            Situation:
            You are a standard car and in your lane there are not traffic signal.
            You are at the cross and you are stand still.
            There is a standard car on your right ('car_R'), which moves with a velocity of 20 km/h and is at a distance of 20m from the cross.
            There is a standard car on your left ('car_L'), which is already at the cross, is stand still and want to go straight.
            # Query: go straight
            {
              "tasks": ["go straight", "go away"]
            }

            Note: Your car go first because you are on the right of car_L and car_R is not yet at the cross.

            Situation:
            You are a standard car and in your lane there are not traffic signal.
            You are at the cross and you are stand still.
            There is a standard car in front of you ('car_2'), which moves with a velocity of 20 km/h and is at a distance of 20m from the cross.
            There is a standard car on your right ('car_1'), which is already at the cross, is stand still and want to go straight.
            Query: go straight"""

    TP_PROMPT_X_CROSS = """
        You are a helpful assistant in charge of controlling the traffic of cars at a road cross with four entrances.
        The user will provide you with the current positions of the cars in the cross and their respective destinations. Your task is to formulate a plan to decide the order in which the cars will proceed.

        You can control the traffic using the following commands:
          (1) Instructions in natural language to move the cars to specific positions.

        Rules:
          (1) The four possible initial positions of the cars correspond to the cardinal points, i.e., North, East, South, West.
          (2) The four possible target positions where the cars want to go also correspond to the cardinal points, i.e., North, East, South, West.
          (3) You MUST use the four cardinal points as reference for the right-hand rule, i.e. for example a car in South has to give priority to a car in East or a car in North has to give the priority to a car in West.
          (4) Use the traffic rules of switzerland to determine which cars have priority.
          (5) You must specify in each task what each car has to do.
          (6) Cars can move simultaneously only if there is no danger of collision, for example if car_1 is in South and want to go in North, while car_2 is in North and want to go in South.

        You MUST always respond with JSON following this format:
        {
          "tasks": ["task1", "task2", "task3", ...]
        }

        Here are some general examples:

        objects = ['car_1', 'car_2']
        # Query: car_1 is in North, car_2 is in South. car_1 wants to go South, car_2 wants to go North.
        {
          "tasks": ["car_1: go to South. car_2: go to North"]
        }

        Note: here the two car move simultaneously because they do not cross their trajectories.

        objects = ['car_1', 'car_2']
        # Query: car_1 is in West, car_2 is in South. car_1 wants to go East, car_2 wants to go West.
        {
          "tasks": ["car_1: wait. car_2: go to West", "car_1: go to East. car_2: go away"]
        }

        Note: Even if car_1 has to do nothing, this must be included in the task.

        objects = ['car_1', 'car_2', 'car_3']
        # Query: car_1 is in West, car_2 is in South, car_3 is in East. car_1 wants to go East, car_2 wants to go West, car_3 wants to go North.
        {
          "tasks": ["car_1: wait. car_2: wait. car_3: go to North", "car_1: wait. car_2: go to West. car_3: go away", "car_1: go to East. car_2: go away. car_3: go away"]
        }

        Note: here the right-hand rules is used. In fact car_3 in East has priority on car_2 in the South and car_2 in South has priority on car_1 in West.
        
        objects = ['car_1', 'car_2', 'car_3']
        # Query: car_1 and car_2 are in West and car_1 is in front of car_2, car_3 is in East. car_1 wants to go East, car_2 wants to go South, car_3 wants to go North.
        {
          "tasks": ["car_1: wait. car_2: wait. car_3: go to North", "car_1: go to East. car_2: wait. car_3: go away", "car_1: go away. car_2: go to West. car_3: go away"]
        }
        
        Note: car_2 move after car_1 because car_1 is in front of car_2.
        
        Query: car_1 is in South, car_2 is in East, car_3 is in North, car_4 is in West, car_5 is in West. car_1 wants to go North, car_2 wants to go South, car_3 wants to go West, car_4 wants to go North, car_5 wants to go East."""

    OD_PROMPT_X_CROSS_LIN_SYS = """
        You are a helpful assistant in charge of designing the optimization problem for an MPC controller that is controlling the traffic of vehicles at a road cross with four entrances. 
        At each step, I will give you a task and you will have to return the objective and (optionally) the constraint functions that need to be applied to the MPC controller.

        This is the scene description:
          (1) Casadi is used to program the MPC.
          (2) The variable `x['car_1']` represents the position of car_1 in 2D, i.e. (x, y). For car_2 the variable is `x['car_2']` and the same for each other car.
          (3) The variable `v['car_1']` represents the linear velocity of car_1 in 2D, i.e. (v_x, v_y). For car_2 the variable is `v['car_2']` and the same for each other car.
          (4) The variable `x0['car_1']` represents the initial position of car_1 at the current time step before any action is applied i.e. (x, y). For car_2 the variable is `x0['car_2']` and the same for each other car.
          (5) The variable `t` represents the simulation time.
          (6) Each time I will also give you a list of objects you can interact with (i.e. objects = ['North', 'South']).
            (a) The position of each object is an array [x, y] obtained by adding `[postion]` (i.e. 'North['position']').
          (7)
            (a) 'in front of' and 'behind' for positive and negative x-axis directions.
            (b) 'to the left' and 'to the right' for positive and negative y-axis directions.
            (c) 'above' and 'below' for positive and negative z-axis directions.


        Rules:
          (1) You MUST write every equality constraints such that it is satisfied if it is = 0:
            (a)  If you want to write "ca.norm_2(x) = 1" write it as  "1 - ca.norm_2(x)" instead.
          (2) You MUST write every inequality constraints such that it is satisfied if it is <= 0:
            (a)  If you want to write "ca.norm_2(x) >= 1" write it as  "1 - ca.norm_2(x)" instead. 

        You must format your response into a json. Here are a few examples:

        objects = ['South', 'North']
        # Query: car_1: go to South. car_2: go to North
        {
          "objective": "ca.norm_2(x['car_1'] - South['position'])**2 + ca.norm_2(x['car_2'] - North['position'])**2",
          "equality_constraints": [],
          "inequality_constraints": []
        }

        objects = ['West']
        # Query: car_1: wait. car_2: go to West
        {
          "objective": "ca.norm_2(x['car_2'] - West['position'])**2",
          "equality_constraints": ["x['car_1'] - x0['car_1']"],
          "inequality_constraints": []
        }

        objects = ['North']
        # Query: car_1: go away. car_2: wait. car_3: go to North
        {
          "objective": "ca.norm_2(x['car_3'] - North['position'])**2",
          "equality_constraints": ["x['car_2'] - x0['car_2']"],
          "inequality_constraints": []
        }
        
        """

    OD_PROMPT_X_CROSS_UNICYCLE = """
            You are a helpful assistant in charge of designing the optimization problem for an MPC controller that is controlling the traffic of vehicles at a road cross with four entrances. 
            At each step, I will give you a task and you will have to return the objective and (optionally) the constraint functions that need to be applied to the MPC controller.

            This is the scene description:
              (1) Casadi is used to program the MPC.
              (2) The variable `x['car_1']` represents the position of car_1 in 2D, i.e. (x, y). For car_2 the variable is `x['car_2']` and the same for each other car.
              (3) The variable `theta['car_1']` represents the angle of the unicycle car_1 in radians. For car_2 the variable is `theta['car_2']` and the same for each other car.
              (4) The variable `v['car_1']` represents the linear velocity of car_1 in 2D, i.e. (v_x, v_y). For car_2 the variable is `v['car_2']` and the same for each other car.
              (5) The variable `omega['car_1']' represents the angular velocity of the unicycle car_1 in radians/s. For car_2 the variable is `omega['car_2']` and the same for each other car.
              (6) The variables `x0['car_1']` represents the initial position of car_1 at the current time step before any action is applied i.e. (x, y). For car_2 the variable is `x0['car_2']` and the same for each other car.
              (7) The variable `t` represents the simulation time.
              (8) Each time I will also give you a list of objects you can interact with (i.e. objects = ['North', 'South']).
                (a) The position of each object is an array [x, y] obtained by adding `[postion]` (i.e. 'North['position']').
              (9)
                (a) 'in front of' and 'behind' for positive and negative x-axis directions.
                (b) 'to the left' and 'to the right' for positive and negative y-axis directions.
                (c) 'above' and 'below' for positive and negative z-axis directions.


            Rules:
              (1) You MUST write every equality constraints such that it is satisfied if it is = 0:
                (a)  If you want to write "ca.norm_2(x) = 1" write it as  "1 - ca.norm_2(x)" instead.
              (2) You MUST write every inequality constraints such that it is satisfied if it is <= 0:
                (a)  If you want to write "ca.norm_2(x) >= 1" write it as  "1 - ca.norm_2(x)" instead. 

            You must format your response into a json. Here are a few examples:

            objects = ['South', 'North']
            # Query: car_1: go to South. car_2: go to North
            {
              "objective": "ca.norm_2(x['car_1'] - South['position'])**2 + ca.norm_2(x['car_2'] - North['position'])**2",
              "equality_constraints": [],
              "inequality_constraints": []
            }

            objects = ['West']
            # Query: car_1: wait. car_2: go to West
            {
              "objective": "ca.norm_2(x['car_2'] - West['position'])**2",
              "equality_constraints": ["x['car_1'] - x0['car_1']"],
              "inequality_constraints": []
            }

            objects = ['North']
            # Query: car_1: go away. car_2: wait. car_3: go to North
            {
              "objective": "ca.norm_2(x['car_3'] - North['position'])**2",
              "equality_constraints": ["x['car_2'] - x0['car_2']"],
              "inequality_constraints": []
            }

            """

    PROMPTS = {
        "cart": {
            "TP": {
                "2D Linear System": TP_PROMPT_CART,
                "Unicycle": TP_PROMPT_CART
            },
            "OD": {
                "2D Linear System": OD_PROMPT_CART_LIN_SYS,
                "Unicycle": OD_PROMPT_CART_UNICYCLE
            }
        },

        "x_cross": {
            "TP": {
                "2D Linear System": TP_PROMPT_X_CROSS,
                "Unicycle": TP_PROMPT_X_CROSS
            },
            "OD": {
                "2D Linear System": OD_PROMPT_X_CROSS_LIN_SYS,
                "Unicycle": OD_PROMPT_X_CROSS_UNICYCLE
            }
        }

    }

    return PROMPTS

def get_help_prompt(simulation_case, x, target, obstacle, task):
    obstacle_pos = obstacle['center']
    R_obracle = obstacle['radius']
    x_pos = x[0:2]
    if simulation_case == 'cart':
        stuck_pos = '('+str(float(x_pos[0]))+', '+str(float(x_pos[1]))+')'
        target_pos = '('+str(target[0,0])+', '+str(target[1,0])+')'
        center_obst = '('+str(obstacle_pos[0,0])+', '+str(obstacle_pos[1,0])+')'
        distance = str(R_obracle+10)

        OLD = """
                Situation: During the execution of a task in the last given json file, the system gets stuck at position """ + stuck_pos + """. You know that the target of this task is at position """ + target_pos + """ and there is an obstacle between the system and the target with center in """ + center_obst + """ and radius """ + str(
            R_obracle) + """.
                objects = []
                Query: Please propose another json file equal to the last given json file with the only difference of an additional task before task '""" + str(
            task) + """', where you instructs the system to go to a temporary target that is outside of the obstacle. The temporary target should be specified in parenthesis and chosen based on the information given in Situation. Please note that, except the new task, all other tasks in the new json file MUST be the same as the last given json file. 
                """


        TP_PROMPT_HELP = """
        Situation: During the execution of a task in the last given json file, the system gets stuck at position """+stuck_pos+""". You know that the target of this task is at position """+target_pos+""" and there is an obstacle between the system and the target with centre in """+center_obst+""".
        objects = []
        Query: Please propose another json file equal to the last given json file with the only difference of an additional task before task '"""+str(task)+"""', where you instructs the system to go to a specific temporary target. The temporary target should be specified in parenthesis and only the specific position of the temporary target MUST be specified in the task. Try to propose a temporary target at a distance of """+distance+""" from the system and at a distance of """+distance+""" from centre of the obstacle. Please note that, except the new task, all other tasks in the new json file MUST be the same as the last given json file. 
        """

    return TP_PROMPT_HELP

def get_another_help_prompt(simulation_case, x, target, obstacle, task):
    obstacle_pos = obstacle['center']
    R_obracle = obstacle['radius']
    x_pos = x[0:2]
    if simulation_case == 'cart':
        stuck_pos = '(' + str(float(x_pos[0])) + ', ' + str(float(x_pos[1])) + ')'
        target_pos = '(' + str(target[0, 0]) + ', ' + str(target[1, 0]) + ')'
        center_obst = '(' + str(obstacle_pos[0, 0]) + ', ' + str(obstacle_pos[1, 0]) + ')'
        distance = str(R_obracle + 10)

        TP_PROMPT_HELP = """
            objects = []
            Query: For some reasons the temporary target you have given is not feasible. Please propose the same last json file only with an other possible temporary target for task before the '"""+str(task)+"""' task. The temporary target should be specified in parenthesis and only the specific position of the temporary target MUST be specified in the task. Try to propose a temporary target at a distance of """+distance+""" from the system and at a distance of """+distance+""" from centre of the obstacle. Use the information in the last given Situation description.
            """

    return TP_PROMPT_HELP
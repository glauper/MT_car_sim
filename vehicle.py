import numpy as np
import casadi as ca
import random

class Vehicle:
    def __init__(self, type, info_vehicle, delta_t):
        self.type = type
        self.delta_t = delta_t
        self.length = info_vehicle['length']
        self.width = info_vehicle['width']
        self.l_r = info_vehicle['l_r']
        self.l_f = info_vehicle['l_f']
        self.security_dist = info_vehicle['security distance']
        self.vel_limits = info_vehicle['velocity limits']
        self.acc_limits = info_vehicle['acceleration limits']
        self.steering_limits = info_vehicle['steering angle limits']

    def init_system_constraints(self, state_space, street_vel_limit):
        self.n = 4
        self.m = 2
        self.A_x = np.zeros((self.n*2, self.n))
        for i in range(self.n):
            self.A_x[i*2,i] = 1
            self.A_x[i*2+1, i] = -1

        self.b_x = np.zeros((self.n*2, 1))
        self.b_x[0, 0] = state_space["x limits"][1]
        self.b_x[1, 0] = -state_space["x limits"][0]
        self.b_x[2, 0] = state_space["y limits"][1]
        self.b_x[3, 0] = -state_space["y limits"][0]
        self.b_x[4, 0] = 2 * np.pi
        self.b_x[5, 0] = 2 * np.pi
        """if street_vel_limit <= self.vel_limits[1]:
            self.b_x[6, 0] = street_vel_limit
        else:
            self.b_x[6, 0] = self.vel_limits[1]"""
        self.b_x[6, 0] = self.vel_limits[1]
        self.b_x[7, 0] = -self.vel_limits[0]

        self.A_u = np.zeros((self.m * 2, self.m))
        for i in range(self.m):
            self.A_u[i * 2, i] = 1
            self.A_u[i * 2 + 1, i] = -1

        self.b_u = np.zeros((self.m * 2, 1))
        self.b_u[0, 0] = self.acc_limits[1] * self.delta_t
        self.b_u[1, 0] = -self.acc_limits[0] * self.delta_t
        self.b_u[2, 0] = self.steering_limits[1] * (np.pi / 180)
        self.b_u[3, 0] = -self.steering_limits[0] * (np.pi / 180)

    def update_velocity_limits(self, env):

        if self.entering:
            for entrance in env['Entrances']:
                for point in range(len(env['Entrances'][entrance]['waypoints'])):
                    if (env['Entrances'][entrance]['waypoints'][point][0] == self.target[0] and
                            env['Entrances'][entrance]['waypoints'][point][1] == self.target[1]):
                        self.b_x[6, 0] = env['Entrances'][entrance]['speed limit']

        elif self.exiting:
            for exit in env['Exits']:
                if (env['Exits'][exit]['position'][0] == self.target[0] and
                        env['Exits'][exit]['position'][1] == self.target[1]):
                    self.b_x[6, 0] = env['Exits'][exit]['speed limit']
                for point in range(len(env['Exits'][exit]['waypoints'])):
                    if (env['Exits'][exit]['waypoints'][point][0] == self.target[0] and
                            env['Exits'][exit]['waypoints'][point][1] == self.target[1]):
                        self.b_x[6, 0] = env['Exits'][exit]['speed limit']

    def init_state_for_LLM(self, env, query, N):
        self.LLM_car = True
        self.N = N

        state = np.zeros((4, 1))
        state[0:2, 0] = env['Ego Entrance']['position']
        state[2, 0] = env['Ego Entrance']['orientation'] * (np.pi / 180)  # form degrees in radiants
        state[3, 0] = random.uniform(0, env['Ego Entrance']['speed limit'])

        if 'right' in query:
            key_target = '2'
        elif 'left' in query:
            key_target = '0'
        elif 'straight' in query:
            key_target = '3'

        target = np.zeros((4, 1))
        target[0:2, 0] = env['Exits'][key_target]['position']
        target[2, 0] = env['Exits'][key_target]['orientation'] * (np.pi / 180)  # form degrees in radiants
        target[3, 0] = 0

        self.waypoints_entering = []
        for i in range(len(env['Ego Entrance']['waypoints'])):
            point = np.zeros((4, 1))
            point[0, 0] = env['Ego Entrance']['waypoints'][i][0]
            point[1, 0] = env['Ego Entrance']['waypoints'][i][1]
            point[2, 0] = env['Ego Entrance']['orientation'] * (np.pi / 180)
            # point[3, 0] = env['Entrances'][key_init]['speed limit']
            point[3, 0] = 0
            self.waypoints_entering.append(point)

        self.waypoints_exiting = []
        for i in range(len(env['Exits'][key_target]['waypoints'])):
            point = np.zeros((4, 1))
            point[0, 0] = env['Exits'][key_target]['waypoints'][i][0]
            point[1, 0] = env['Exits'][key_target]['waypoints'][i][1]
            point[2, 0] = env['Exits'][key_target]['orientation'] * (np.pi / 180)
            #point[3, 0] = env['Exits'][key_target]['speed limit']
            point[3, 0] = 0
            self.waypoints_exiting.append(point)

        self.waypoints_exiting.append(target)

        self.target = self.waypoints_entering.pop(0)
        self.state = state
        self.x = state[0]
        self.y = state[1]
        self.position = np.array([self.x, self.y]).reshape(2, 1)
        self.theta = state[2]
        self.velocity = state[3]
        self.entering = True
        self.exiting = False
        self.inside_cross = False

        # Additional variables needed if you want to use the LLM with the car
        self.t_subtask = 0

        """if query == 'go right':
            self.right = {}
            self.right['state'] = self.waypoints_exiting[0]
            self.right['position'] = self.right['state'][0:2]
            self.right['theta'] = self.right['state'][2]
            self.right['velocity'] = self.right['state'][3]
        elif query == 'go left':
            self.left = {}
            self.left['state'] = self.waypoints_exiting[0]
            self.left['position'] = self.left['state'][0:2]
            self.left['theta'] = self.left['state'][2]
            self.left['velocity'] = self.left['state'][3]
        elif query == 'go straight':
            self.straight = {}
            self.straight['state'] = self.waypoints_exiting[0]
            self.straight['position'] = self.straight['state'][0:2]
            self.straight['theta'] = self.straight['state'][2]
            self.straight['velocity'] = self.straight['state'][3]"""

        self.entry = {}
        self.entry['state'] = self.target
        self.entry['position'] = self.entry['state'][0:2]
        self.entry['x'] = self.entry['state'][0]
        self.entry['y'] = self.entry['state'][1]
        self.entry['theta'] = self.entry['state'][2]
        self.entry['velocity'] = self.entry['state'][3]
        self.entry['max vel'] = env['Ego Entrance']['speed limit']

        self.exit = {}
        self.exit['state'] = self.waypoints_exiting[0]
        self.exit['position'] = self.exit['state'][0:2]
        self.exit['x'] = self.exit['state'][0]
        self.exit['y'] = self.exit['state'][1]
        self.exit['theta'] = self.exit['state'][2]
        self.exit['velocity'] = self.exit['state'][3]
        self.exit['max vel'] = env['Exits'][key_target]['speed limit']

        self.final_target = {}
        self.final_target['state'] = self.waypoints_exiting[-1]
        self.final_target['position'] = self.final_target['state'][0:2]
        self.final_target['x'] = self.final_target['state'][0]
        self.final_target['y'] = self.final_target['state'][1]
        self.final_target['theta'] = self.final_target['state'][2]
        self.final_target['velocity'] = self.final_target['state'][3]
        self.final_target['max vel'] = env['Exits'][key_target]['speed limit']

        self.previous_opt_sol_SF = {}
        self.previous_opt_sol = {}
        # This needed for the optimization of MPC other vehicle
        self.previous_opt_sol['X'] = np.tile(self.state, (1, self.N + 1)).reshape(self.n, self.N + 1)
        self.previous_opt_sol['U'] = np.zeros((self.m, self.N))

    def init_state(self, env, key_init):
        self.LLM_car = False
        state = np.zeros((4, 1))
        state[0:2, 0] = env['Entrances'][key_init]['position']
        state[2, 0] = env['Entrances'][key_init]['orientation'] * (np.pi / 180)  # form degrees in radiants
        state[3, 0] = random.uniform(0, env['Entrances'][key_init]['speed limit'])

        key_target = str(random.choice(env['Entrances'][key_init]['targets']))
        target = np.zeros((4, 1))
        target[0:2, 0] = env['Exits'][key_target]['position']
        target[2, 0] = env['Exits'][key_target]['orientation'] * (np.pi / 180)  # form degrees in radiants
        #target[3, 0] = env['Exits'][key_target]['speed limit']
        target[3, 0] = 0

        self.waypoints_entering = []
        for i in range(len(env['Entrances'][key_init]['waypoints'])):
            point = np.zeros((4,1))
            point[0, 0] = env['Entrances'][key_init]['waypoints'][i][0]
            point[1, 0] = env['Entrances'][key_init]['waypoints'][i][1]
            point[2, 0] = env['Entrances'][key_init]['orientation'] * (np.pi / 180)
            #point[3, 0] = env['Entrances'][key_init]['speed limit']
            point[3, 0] = 0
            self.waypoints_entering.append(point)

        self.waypoints_exiting = []
        for i in range(len(env['Exits'][key_target]['waypoints'])):
            point = np.zeros((4,1))
            point[0, 0] = env['Exits'][key_target]['waypoints'][i][0]
            point[1, 0] = env['Exits'][key_target]['waypoints'][i][1]
            point[2, 0] = env['Exits'][key_target]['orientation'] * (np.pi / 180)
            point[3, 0] = env['Exits'][key_target]['speed limit']
            #point[3, 0] = 0
            self.waypoints_exiting.append(point)

        self.waypoints_exiting.append(target)

        self.target = self.waypoints_entering.pop(0)
        self.state = state
        self.x = state[0]
        self.y = state[1]
        self.position = np.array([self.x, self.y]).reshape(2,1)
        self.theta = state[2]
        self.velocity = state[3]
        self.entering = True
        self.exiting = False
        self.inside_cross = False


    def dynamics_propagation(self, input):
        beta = np.arctan(self.l_r/(self.l_r + self.l_f) * np.tan(input[1]))
        self.x = self.x + self.delta_t * self.velocity * np.cos(self.theta + beta)
        self.y = self.y + self.delta_t * self.velocity * np.sin(self.theta + beta)
        self.theta = self.theta + self.delta_t * self.velocity / self.l_r * np.sin(beta)
        self.velocity = self.velocity + self.delta_t * input[0]

        self.state = np.array([self.x, self.y, self.theta, self.velocity]).reshape((4,1))
        self.position = np.array([self.x, self.y]).reshape(2,1)

    def brakes(self):
        beta = np.arctan(self.l_r / (self.l_r + self.l_f) * np.tan(0))
        input = 0 - self.velocity
        if self.acc_limits[0] > input:
            input = self.acc_limits[1]
        self.velocity = self.velocity + self.delta_t * input

        self.x = self.x + self.delta_t * self.velocity * np.cos(self.theta + beta)
        self.y = self.y + self.delta_t * self.velocity * np.sin(self.theta + beta)
        self.theta = self.theta + self.delta_t * self.velocity / self.l_r * np.sin(beta)

        self.state = np.array([self.x, self.y, self.theta, self.velocity]).reshape((4, 1))
        self.position = np.array([self.x, self.y]).reshape(2, 1)


    def dynamics_constraints(self, x_next, x_now, u_now):
        beta = np.arctan(self.l_r/(self.l_r + self.l_f) * np.tan(u_now[1]))

        constraints = [x_next[0] == x_now[0] + self.delta_t * x_now[3] * np.cos(x_now[2] + beta),
                      x_next[1] == x_now[1] + self.delta_t * x_now[3] * np.sin(x_now[2] + beta),
                      x_next[2] == x_now[2] + self.delta_t * x_now[3] / self.l_r * np.sin(beta),
                      x_next[3] == x_now[3] + self.delta_t * u_now[0]]

        return constraints

    def init_trackingMPC(self, N):
        self.N = N
        self.Q = np.eye(self.n)
        self.R = np.eye(self.m)
        self.P = 10
        self.T = 10 * self.P
        self.previous_opt_sol = {}
        self.previous_opt_sol['X'] = np.tile(self.state, (1, self.N + 1)).reshape(self.n, self.N + 1)
        self.previous_opt_sol['U'] = np.zeros((self.m, self.N))

    def trackingMPC(self, other_agents, ego, circular_obstacles, t):

        if self.state[2] > 0 and self.target[2] < 0:
            if self.target[2] < self.state[2] - np.pi:
                self.target[2] += 2 * np.pi
        elif self.state[2] < 0 and self.target[2] > 0:
            if self.target[2] > self.state[2] + np.pi:
                self.target[2] -= 2 * np.pi

        nr_agents = len(other_agents)
        opti = ca.Opti()

        X = opti.variable(self.n, self.N + 1)
        U = opti.variable(self.m, self.N)
        x_s = opti.variable(self.n, 1)
        u_s = opti.variable(self.m, 1)
        epsilon = opti.variable(1, 1)

        cost = 0
        for k in range(self.N):
            diff_X = X[:, k] - x_s
            diff_U = U[:, k] - u_s
            cost += ca.transpose(diff_X) @ self.Q @ diff_X + ca.transpose(diff_U) @ self.R @ diff_U
            # State and Input constraints
            opti.subject_to(self.A_x @ X[:, k + 1] <= self.b_x)
            opti.subject_to(self.A_u @ U[:, k] <= self.b_u)

            # System dynamics constraints
            opti.subject_to(self.dynamics_constraints(X[:, k + 1], X[:, k], U[:, k]))

        diff_X = X[:, -1] - x_s
        diff_target = self.target - x_s
        cost += ca.transpose(diff_X) @ self.P @ diff_X + ca.transpose(diff_target) @ self.T @ diff_target
        cost += epsilon**2

        # Initial state
        opti.subject_to(X[:, 0] == self.state)

        # Constraint for the steady state
        opti.subject_to(self.A_x @ x_s <= self.b_x)
        opti.subject_to(self.A_u @ u_s <= self.b_u)
        opti.subject_to(self.dynamics_constraints(x_s, x_s, u_s))
        # Terminal constraints
        opti.subject_to(X[:, -1] == x_s)  # x(N) == x_s

        # Aviodance of other agents
        if nr_agents >= 1:
            for k in range(self.N + 1):
                for other_agent in other_agents:
                    diff = X[0:2, k] - other_agents[other_agent].previous_opt_sol['X'][0:2, k]
                    # Which of the distance have to keep? mine or of the other one? Or one standard for all
                    if self.security_dist >= other_agents[other_agent].security_dist:
                        opti.subject_to(ca.transpose(diff) @ diff >= self.security_dist ** 2)
                    else:
                        opti.subject_to(ca.transpose(diff) @ diff >= other_agents[other_agent].security_dist ** 2)

        # Avoidance of ego vehicle
        if ego != []:
            for k in range(self.N + 1):
                diff = X[0:2, k] - ego.previous_opt_sol['X'][0:2, k]
                # Which of the distance have to keep? mine or of the other one? Or one standard for all
                if self.security_dist >= ego.security_dist:
                    opti.subject_to(ca.transpose(diff) @ diff >= self.security_dist ** 2 + epsilon)
                else:
                    opti.subject_to(ca.transpose(diff) @ diff >= ego.security_dist ** 2 + epsilon)

        # Obstacle aviodance
        if len(circular_obstacles) != 0:
            for id_obst in circular_obstacles:
                for k in range(self.N + 1):
                    diff_x = (X[0, k] - circular_obstacles[id_obst]['center'][0]) ** 2 / (circular_obstacles[id_obst]['r_x']) ** 2
                    diff_y = (X[1, k] - circular_obstacles[id_obst]['center'][1]) ** 2 / (circular_obstacles[id_obst]['r_y']) ** 2
                    opti.subject_to(diff_x + diff_y >= 1)

        opti.minimize(cost)
        # Solve the optimization problem
        if t == 0:  # or agents[f'{i}'].target_status
            opti.set_initial(X, np.tile(self.state, (1, self.N + 1)).reshape(self.n, self.N + 1))
            opti.set_initial(U, np.zeros((self.m, self.N)))
            opti.set_initial(x_s, self.state)
            opti.set_initial(u_s, np.zeros((self.m, 1)))

        else:
            opti.set_initial(X, self.previous_opt_sol['X'])
            opti.set_initial(U, self.previous_opt_sol['U'])
            opti.set_initial(x_s, self.previous_opt_sol['x_s'])
            opti.set_initial(u_s, self.previous_opt_sol['u_s'])

        opti.solver('ipopt')
        sol = opti.solve()

        stats = opti.stats()
        exit_flag = stats['success']

        if not exit_flag:
            print('Solution is not optimal')
            error()

        self.previous_opt_sol['X'] = sol.value(X)
        self.previous_opt_sol['U'] = sol.value(U)
        self.previous_opt_sol['x_s'] = sol.value(x_s)
        self.previous_opt_sol['u_s'] = sol.value(u_s)
        self.previous_opt_sol['Cost'] = sol.value(cost)


        input = sol.value(U)[:, 0].reshape((self.m, 1))

        if self.state[2] > 0 and self.target[2] < 0:
            if self.target[2] < self.state[2] - np.pi:
                self.target[2] -= 2 * np.pi
        elif self.state[2] < 0 and self.target[2] > 0:
            if self.target[2] > self.state[2] + np.pi:
                self.target[2] += 2 * np.pi

        return input

    def MPC_LLM(self, agents, circular_obstacles, t, llm):

        opti = ca.Opti()

        X = opti.variable(self.n, self.N + 1)
        U = opti.variable(self.m, self.N)

        cost = 0
        for k in range(self.N):
            # LLM cost and constriants
            cost, opti = self.OD_output(opti, cost, llm, X[:,k], U[:,k], agents)

            # State and Input constraints
            opti.subject_to(self.A_x @ X[:, k + 1] <= self.b_x)
            opti.subject_to(self.A_u @ U[:, k] <= self.b_u)

            # System dynamics constraints
            opti.subject_to(self.dynamics_constraints(X[:, k + 1], X[:, k], U[:, k]))

        cost, opti = self.OD_output(opti, cost, llm, X[:,-1], U[:,-1], agents) # U[-1] and X[-1] do not corresponds!

        # Initial state
        opti.subject_to(X[:, 0] == self.state)

        # Agents avoidance
        if len(agents) >= 1:
            for k in range(self.N + 1):
                for id_agent in agents:
                    diff = X[0:2, k] - agents[id_agent].position
                    # Which of the distance have to keep? mine or of the other one? Or one standard for all
                    if self.security_dist >= agents[id_agent].security_dist:
                        opti.subject_to(ca.transpose(diff) @ diff >= self.security_dist ** 2)
                    else:
                        opti.subject_to(ca.transpose(diff) @ diff >= agents[id_agent].security_dist ** 2)

        # Obstacle avoidance
        if len(circular_obstacles) != 0:
            for id_obst in circular_obstacles:
                for k in range(self.N + 1):
                    diff_x = (X[0, k] - circular_obstacles[id_obst]['center'][0]) ** 2 / (circular_obstacles[id_obst]['r_x']) ** 2
                    diff_y = (X[1, k] - circular_obstacles[id_obst]['center'][1]) ** 2 / (circular_obstacles[id_obst]['r_y']) ** 2
                    opti.subject_to(diff_x + diff_y >= 1)

        opti.minimize(cost)
        # Solve the optimization problem
        if t == 0:  # or agents[f'{i}'].target_status
            opti.set_initial(X, np.tile(self.state, (1, self.N + 1)).reshape(self.n, self.N + 1))
            opti.set_initial(U, np.zeros((self.m, self.N)))

        else:
            opti.set_initial(X, self.previous_opt_sol['X'])
            opti.set_initial(U, self.previous_opt_sol['U'])

        opti.solver('ipopt')
        sol = opti.solve()

        stats = opti.stats()
        exit_flag = stats['success']

        if not exit_flag:
            print('Solution is not optimal')
            error()

        self.previous_opt_sol['X'] = sol.value(X)
        self.previous_opt_sol['U'] = sol.value(U)
        self.previous_opt_sol['Cost'] = sol.value(cost)

        input = sol.value(U)[:, 0].reshape((self.m, 1))

        return input

    def OD_output(self, opti, obj, llm, X, U, agents):

        x = X[0]
        y = X[1]
        theta = X[2]
        v = X[3]

        delta = U[0]
        a = U[1]

        cost = llm.OD["objective"]
        ineq_constraints = llm.OD["inequality_constraints"]
        eq_constraints = llm.OD["equality_constraints"]

        nr_ineq_const = len(ineq_constraints)
        nr_eq_const = len(eq_constraints)

        obj += eval(cost)
        for i in range(nr_ineq_const):
            opti.subject_to(eval(ineq_constraints[i]) <= 0)
        for i in range(nr_eq_const):
            opti.subject_to(eval(eq_constraints[i]) == 0)

        return obj, opti

    def SF(self, u_lernt, agents, circular_obstacles, t):


        opti = ca.Opti()

        X = opti.variable(self.n, self.N + 1)
        U = opti.variable(self.m, self.N)
        x_s = opti.variable(self.n, 1)
        u_s = opti.variable(self.m, 1)

        opti.minimize(ca.norm_2(u_lernt - U[:, 0]) ** 2)

        for k in range(self.N):
            # State and Input constraints
            opti.subject_to(self.A_x @ X[:, k + 1] <= self.b_x)
            opti.subject_to(self.A_u @ U[:, k] <= self.b_u)

            # System dynamics constraints
            opti.subject_to(self.dynamics_constraints(X[:, k + 1], X[:, k], U[:, k]))

        # Initial state
        opti.subject_to(X[:, 0] == self.state)

        # Agents avoidance
        if len(agents) >= 1:
            for k in range(self.N + 1):
                for id_agent in agents:
                    if agents[id_agent].security_dist != 0:
                        diff = X[0:2, k] - agents[id_agent].position
                        # Which of the distance have to keep? mine or of the other one? Or one standard for all
                        if self.security_dist >= agents[id_agent].security_dist:
                            opti.subject_to(ca.transpose(diff) @ diff >= self.security_dist ** 2)
                        else:
                            opti.subject_to(ca.transpose(diff) @ diff >= agents[id_agent].security_dist ** 2)

        # Obstacle avoidance
        if len(circular_obstacles) != 0:
            for id_obst in circular_obstacles:
                for k in range(self.N + 1):
                    diff_x = (X[0, k] - circular_obstacles[id_obst]['center'][0]) ** 2 / (circular_obstacles[id_obst]['r_x']) ** 2
                    diff_y = (X[1, k] - circular_obstacles[id_obst]['center'][1]) ** 2 / (circular_obstacles[id_obst]['r_y']) ** 2
                    opti.subject_to(diff_x + diff_y >= 1)

        # Constraint for the steady state
        opti.subject_to(self.A_x @ x_s <= self.b_x)
        opti.subject_to(self.A_u @ u_s <= self.b_u)
        opti.subject_to(self.dynamics_constraints(x_s, x_s, u_s))
        # Terminal constraints
        opti.subject_to(X[:, -1] == x_s)  # x(N) == x_s
        # Solve the optimization problem
        if t == 0:  # or agents[f'{i}'].target_status
            opti.set_initial(X, np.tile(self.state, (1, self.N + 1)).reshape(self.n, self.N + 1))
            opti.set_initial(U, np.zeros((self.m, self.N)))
            opti.set_initial(x_s, self.state)
            opti.set_initial(u_s, np.zeros((self.m, 1)))

        else:
            opti.set_initial(X, self.previous_opt_sol_SF['X'])
            opti.set_initial(U, self.previous_opt_sol_SF['U'])
            opti.set_initial(x_s, self.previous_opt_sol_SF['x_s'])
            opti.set_initial(u_s, self.previous_opt_sol_SF['u_s'])

        opti.solver('ipopt')
        sol = opti.solve()

        stats = opti.stats()
        exit_flag = stats['success']

        if not exit_flag:
            print('Solution is not optimal')
            error()

        self.previous_opt_sol_SF['X'] = sol.value(X)
        self.previous_opt_sol_SF['U'] = sol.value(U)
        self.previous_opt_sol_SF['x_s'] = sol.value(x_s)
        self.previous_opt_sol_SF['u_s'] = sol.value(u_s)
        self.previous_opt_sol_SF['Cost'] = sol.value(np.linalg.norm(u_lernt - sol.value(U)[:, 0]) ** 2)

        input = sol.value(U)[:, 0].reshape((self.m, 1))

        return input





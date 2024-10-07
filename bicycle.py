import numpy as np
import casadi as ca
import random

class Bicycle:
    def __init__(self, type, info_vehicle, delta_t):
        self.type = type
        self.delta_t = delta_t
        self.length = info_vehicle['length']
        self.width = info_vehicle['width']
        self.l_r = info_vehicle['l_r']
        self.l_f = info_vehicle['l_f']
        self.security_dist = info_vehicle['security distance']
        self.a_security_dist = info_vehicle['a security dist']
        self.b_security_dist = info_vehicle['b security dist']
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
        self.b_u[0, 0] = self.acc_limits[1]
        self.b_u[1, 0] = -self.acc_limits[0]
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

    def init_state(self, env, key_init):
        self.LLM_car = False
        state = np.zeros((4, 1))
        state[0:2, 0] = env['Entrances'][key_init]['position']
        state[2, 0] = env['Entrances'][key_init]['orientation'] * (np.pi / 180)  # form degrees in radiants
        state[3, 0] = random.uniform(0, env['Entrances'][key_init]['speed limit'])

        if state[2, 0] == 0:
            state[0:2, 0] = state[0:2, 0] + np.array([0, -2])
        elif state[2, 0] == np.pi / 2:
            state[0:2, 0] = state[0:2, 0] + np.array([2, 0])
        elif state[2, 0] == np.pi:
            state[0:2, 0] = state[0:2, 0] + np.array([0, 2])
        elif state[2, 0] == - np.pi / 2:
            state[0:2, 0] = state[0:2, 0] + np.array([-2, 0])

        key_target = str(random.choice(env['Entrances'][key_init]['targets']))
        target = np.zeros((4, 1))
        target[0:2, 0] = env['Exits'][key_target]['position']
        target[2, 0] = env['Exits'][key_target]['orientation'] * (np.pi / 180)  # form degrees in radiants
        #target[3, 0] = env['Exits'][key_target]['speed limit']
        target[3, 0] = 0

        if target[2, 0] == 0:
            target[0:2, 0] = target[0:2, 0] + np.array([0, -2])
        elif target[2, 0] == np.pi / 2:
            target[0:2, 0] = target[0:2, 0] + np.array([2, 0])
        elif target[2, 0] == np.pi:
            target[0:2, 0] = target[0:2, 0] + np.array([0, 2])
        elif target[2, 0] == - np.pi / 2:
            target[0:2, 0] = target[0:2, 0] + np.array([-2, 0])

        self.waypoints_entering = []
        for i in range(len(env['Entrances'][key_init]['waypoints'])):
            point = np.zeros((4,1))
            point[0, 0] = env['Entrances'][key_init]['waypoints'][i][0]
            point[1, 0] = env['Entrances'][key_init]['waypoints'][i][1]
            point[2, 0] = env['Entrances'][key_init]['orientation'] * (np.pi / 180)
            #point[3, 0] = env['Entrances'][key_init]['speed limit']
            point[3, 0] = 0

            if point[2, 0] == 0:
                point[0:2, 0] = point[0:2, 0] + np.array([0, -2])
            elif point[2, 0] == np.pi / 2:
                point[0:2, 0] = point[0:2, 0] + np.array([2, 0])
            elif point[2, 0] == np.pi:
                point[0:2, 0] = point[0:2, 0] + np.array([0, 2])
            elif point[2, 0] == - np.pi / 2:
                point[0:2, 0] = point[0:2, 0] + np.array([-2, 0])

            self.waypoints_entering.append(point)

        self.waypoints_exiting = []
        for i in range(len(env['Exits'][key_target]['waypoints'])):
            point = np.zeros((4,1))
            point[0, 0] = env['Exits'][key_target]['waypoints'][i][0]
            point[1, 0] = env['Exits'][key_target]['waypoints'][i][1]
            point[2, 0] = env['Exits'][key_target]['orientation'] * (np.pi / 180)
            #point[3, 0] = env['Exits'][key_target]['speed limit']
            point[3, 0] = 0

            if point[2, 0] == 0:
                point[0:2, 0] = point[0:2, 0] + np.array([0, -2])
            elif point[2, 0] == np.pi / 2:
                point[0:2, 0] = point[0:2, 0] + np.array([2, 0])
            elif point[2, 0] == np.pi:
                point[0:2, 0] = point[0:2, 0] + np.array([0, 2])
            elif point[2, 0] == - np.pi / 2:
                point[0:2, 0] = point[0:2, 0] + np.array([-2, 0])

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
        self.input_brakes = np.zeros((self.m, 1))
        beta = np.arctan(self.l_r / (self.l_r + self.l_f) * np.tan(0))
        input = (0 - self.velocity) / self.delta_t
        if self.acc_limits[0] > input:
            input = self.acc_limits[0]
        elif self.acc_limits[1] < input:
            input = self.acc_limits[1]
        self.input_brakes[1, :] = 0
        self.input_brakes[0, :] = input
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

    def dynamics_constraints_soft(self, x_next, x_now, u_now, epsilon):
        beta = np.arctan(self.l_r/(self.l_r + self.l_f) * np.tan(u_now[1]))

        constraints = [x_next[0] == x_now[0] + self.delta_t * x_now[3] * np.cos(x_now[2] + beta) + epsilon[0],
                      x_next[1] == x_now[1] + self.delta_t * x_now[3] * np.sin(x_now[2] + beta) + epsilon[1],
                      x_next[2] == x_now[2] + self.delta_t * x_now[3] / self.l_r * np.sin(beta) + epsilon[2],
                      x_next[3] == x_now[3] + self.delta_t * u_now[0] + epsilon[3]]

        return constraints
    def agents_constraints(self, X, agent):
        R_agent = np.zeros((2,2))
        R_agent[0, 0] = np.cos(-agent.theta)
        R_agent[0, 1] = -np.sin(-agent.theta)
        R_agent[1, 0] = np.sin(-agent.theta)
        R_agent[1, 1] = np.cos(-agent.theta)

        #V_agent = R_agent @ np.array([[1.5], [0]])
        #diff = R_agent @ (X[0:2] - (agent.position + V_agent))

        diff = R_agent @ (X[0:2] - agent.position)

        constraint = (diff[0] / agent.a_security_dist) ** 2 +(diff[1] / agent.b_security_dist) ** 2 - 1

        return constraint

    def agents_constraints_traj(self, X, X_agent, agent):
        R_agent = np.zeros((2,2))
        R_agent[0, 0] = np.cos(-X_agent[2])
        R_agent[0, 1] = -np.sin(-X_agent[2])
        R_agent[1, 0] = np.sin(-X_agent[2])
        R_agent[1, 1] = np.cos(-X_agent[2])

        diff = R_agent @ (X[0:2] - X_agent[0:2])

        constraint = (diff[0] / agent.a_security_dist) ** 2 +(diff[1] / agent.b_security_dist) ** 2 - 1

        return constraint

    def agents_constraints_circle(self, X, X_agent, agent):
        diff = X[0:2] - X_agent[0:2]
        # Which of the distance have to keep? mine or of the other one? Or one standard for all
        if self.security_dist >= agent.security_dist:
            constraint = ca.transpose(diff) @ diff >= self.security_dist ** 2
        else:
            constraint = ca.transpose(diff) @ diff >= agent.security_dist ** 2

        return constraint

    def init_trackingMPC(self, N):
        self.N = N
        self.Q = np.eye(self.n)
        self.R = np.eye(self.m)
        self.P = 1
        self.T = 20 * self.P
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

        # Initial state
        opti.subject_to(X[:, 0] == self.state)

        # Constraint for the steady state
        opti.subject_to(self.A_x @ x_s <= self.b_x)
        opti.subject_to(self.A_u @ u_s <= self.b_u)
        opti.subject_to(self.dynamics_constraints(x_s, x_s, u_s))
        # Terminal constraints
        opti.subject_to(X[:, -1] == x_s)  # x(N) == x_s

        # Avoidance of other agents
        if nr_agents >= 1:
            for k in range(self.N + 1):
                for other_agent in other_agents:
                    opti.subject_to(self.agents_constraints_circle(X[:, k],
                                                                   other_agents[other_agent].previous_opt_sol['X'][:, k],
                                                                   other_agents[other_agent]))
                    #opti.subject_to(self.agents_constraints(X[:, k], other_agents[other_agent]) >= 0)

        # Avoidance of ego vehicle
        if ego != []:
            for k in range(self.N + 1):
                opti.subject_to(self.agents_constraints(X[:, k], ego) >= 0)

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
            opti.set_initial(X, self.traj_estimation)
            opti.set_initial(U, np.zeros((self.m, self.N)))
            opti.set_initial(x_s, self.traj_estimation[:, -1])
            opti.set_initial(u_s, np.array([[0-self.traj_estimation[3, -1]], [0]]))

        else:
            opti.set_initial(X, self.previous_opt_sol['X'])
            opti.set_initial(U, self.previous_opt_sol['U'])
            try:
                opti.set_initial(x_s, self.previous_opt_sol['x_s'])
                opti.set_initial(u_s, self.previous_opt_sol['u_s'])
            except Exception as e:
                opti.set_initial(x_s, self.traj_estimation[:, -1])
                opti.set_initial(u_s, np.array([[0 - self.traj_estimation[3, -1]], [0]]))

        try:
            opti.solver('ipopt')
            sol = opti.solve()
        except Exception as e:
            print(f"Optimization failed: {e}")

        if opti.stats()['success']:
            self.previous_opt_sol['X'] = sol.value(X)
            self.previous_opt_sol['U'] = sol.value(U)
            self.previous_opt_sol['x_s'] = sol.value(x_s)
            self.previous_opt_sol['u_s'] = sol.value(u_s)

            input = sol.value(U)[:, 0].reshape((self.m, 1))

        else:

            input = np.zeros((self.m, 1))

            if self.acc_limits[0] > 0 - self.velocity:
                input[0,:] = self.acc_limits[1]
            else:
                input[0, :] = self.velocity/2 - self.velocity
            input[1, :] = 0
            print('Other agents MPC solver failed. Use a default u = ', input)

        if self.state[2] > 0 and self.target[2] < 0:
            if self.target[2] < self.state[2] - np.pi:
                self.target[2] -= 2 * np.pi
        elif self.state[2] < 0 and self.target[2] > 0:
            if self.target[2] > self.state[2] + np.pi:
                self.target[2] += 2 * np.pi

        return input

    def trajecotry_estimation(self):
        input = np.zeros((2,1))
        self.traj_estimation = np.zeros((self.n, self.N+1))
        self.traj_estimation[:, 0] = self.state[:, 0]
        for k in range(self.N):
            if k <= self.N:
                beta = np.arctan(self.l_r / (self.l_r + self.l_f) * np.tan(input[1]))
                self.traj_estimation[0, k+1] = self.traj_estimation[0, k] + self.delta_t * self.traj_estimation[3, k] * np.cos(self.traj_estimation[2, k] + beta)
                self.traj_estimation[1, k+1] = self.traj_estimation[1, k] + self.delta_t * self.traj_estimation[3, k] * np.sin(self.traj_estimation[2, k] + beta)
                self.traj_estimation[2, k+1] = self.traj_estimation[2, k] + self.delta_t * self.traj_estimation[3, k] / self.l_r * np.sin(beta)
                self.traj_estimation[3, k+1] = self.traj_estimation[3, k] + self.delta_t * input[0]
            else:
                self.traj_estimation[0, k + 1] = self.traj_estimation[0, k]
                self.traj_estimation[1, k + 1] = self.traj_estimation[1, k]
                self.traj_estimation[2, k + 1] = self.traj_estimation[2, k]
                self.traj_estimation[3, k + 1] = self.traj_estimation[3, k]

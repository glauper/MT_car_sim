import numpy as np
import casadi as ca
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Point, Polygon, GeometryCollection
from shapely.ops import unary_union, triangulate
from shapely.affinity import scale
from scipy.spatial import ConvexHull, Delaunay
from shapely.geometry import Polygon, MultiPolygon
from scipy.optimize import linprog
import time

class Vehicle:
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
                    if (self.target[0] == 0 and self.target[1] == 0):
                        self.b_x[6, 0] = 2

        elif self.exiting:
            for exit in env['Exits']:
                if (env['Exits'][exit]['position'][0] == self.target[0] and
                        env['Exits'][exit]['position'][1] == self.target[1]):
                    self.b_x[6, 0] = env['Exits'][exit]['speed limit']
                for point in range(len(env['Exits'][exit]['waypoints'])):
                    if (env['Exits'][exit]['waypoints'][point][0] == self.target[0] and
                            env['Exits'][exit]['waypoints'][point][1] == self.target[1]):
                        self.b_x[6, 0] = env['Exits'][exit]['speed limit']

    def init_state_for_LLM(self, env, sim_params):
        query = sim_params['Query']
        self.LLM_car = True
        self.terminal_set = sim_params['Controller']['Ego']['LLM']['Terminal set']
        #self.use_LLM_output_in_SF = sim_params['Controller']['Ego']['SF']['Use LLM output']
        self.SF_active = sim_params['Controller']['Ego']['SF']['Active']
        #self.SF_soft_active = sim_params['Controller']['Ego']['SF']['Soft']
        self.N = sim_params['Controller']['Ego']['LLM']['Horizon']
        self.N_SF = sim_params['Controller']['Ego']['SF']['Horizon']

        state = np.zeros((4, 1))
        state[0:2, 0] = env['Ego Entrance']['position']
        y_shift = 0 + random.random() * (env['Priority space']['y limits'][0]-1-env['Ego Entrance']['position'][1])
        state[1, 0] = state[1, 0] + y_shift
        state[2, 0] = env['Ego Entrance']['orientation'] * (np.pi / 180)  # form degrees in radiants
        #state[3, 0] = random.uniform(0, env['Ego Entrance']['speed limit'])
        state[3, 0] = random.uniform(0, 2)

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
        self.previous_opt_sol_psi_opt = {}
        self.previous_opt_sol = {}
        # This needed for the optimization of MPC other vehicle
        self.previous_opt_sol['X'] = np.tile(self.state, (1, self.N + 1)).reshape(self.n, self.N + 1)
        self.previous_opt_sol['U'] = np.zeros((self.m, self.N))
        self.previous_opt_sol_SF['X'] = np.tile(self.state, (1, self.N + 1)).reshape(self.n, self.N + 1)
        self.previous_opt_sol_SF['U'] = np.zeros((self.m, self.N))

        self.fix_obstacles = {}
        for id in env["Forbidden Areas"]:
            self.fix_obstacles [id] = ConvexHull(env["Forbidden Areas"][id])

    def init_state(self, env, key_init):
        self.LLM_car = False
        state = np.zeros((4, 1))
        state[0:2, 0] = env['Entrances'][key_init]['position']
        state[2, 0] = env['Entrances'][key_init]['orientation'] * (np.pi / 180)  # form degrees in radiants
        #state[3, 0] = random.uniform(0, env['Entrances'][key_init]['speed limit'])
        state[3, 0] = random.uniform(0, 2)

        key_target = str(random.choice(env['Entrances'][key_init]['targets']))
        target = np.zeros((4, 1))
        target[0:2, 0] = env['Exits'][key_target]['position']
        target[2, 0] = env['Exits'][key_target]['orientation'] * (np.pi / 180)  # form degrees in radiants
        #target[3, 0] = env['Exits'][key_target]['speed limit']
        target[3, 0] = 0

        self.waypoints_entering = []
        for i in range(len(env['Entrances'][key_init]['waypoints'])):
            point = np.zeros((4,1))
            if self.type == 'emergency car':
                point[0, 0] = 0
                point[1, 0] = 0
                point[2, 0] = env['Entrances'][key_init]['orientation'] * (np.pi / 180)
                # point[3, 0] = env['Entrances'][key_init]['speed limit']
                point[3, 0] = 0
            else:
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
            #point[3, 0] = env['Exits'][key_target]['speed limit']
            point[3, 0] = 0
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
        self.input_brakes[0,:] = input
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

        constraint = (diff[0] / agent.a_security_dist) ** 2 + (diff[1] / agent.b_security_dist) ** 2 - 1

        return constraint

    def agents_constraints_area(self, X, X_agent, agent):

        A_hull = agent.hull.equations[:, :-1]  # Coefficients of x (A_hull)
        b_hull = agent.hull.equations[:, -1]

        return A_hull @ X[0:2] <= b_hull

    def agents_constraints_circle(self, X, X_agent, agent):

        diff = X[0:2] - X_agent[0:2]
        # Which of the distance have to keep? mine or of the other one? Or one standard for all
        if self.security_dist >= agent.security_dist:
            constraint = ca.transpose(diff) @ diff >= self.security_dist ** 2
        else:
            constraint = ca.transpose(diff) @ diff >= agent.security_dist ** 2

        return constraint

    def linear_dynamics_constraints(self, x_next, x_now, u_now):
        A = np.zeros((self.n, self.n))
        B = np.zeros((self.n, self.m))
        u_0 = np.zeros((self.m, 1))
        beta = np.arctan(self.l_r / (self.l_r + self.l_f) * np.tan(u_0[1]))

        A[0, 0] = 1
        A[0, 1] = 0
        A[0, 2] = - self.delta_t * self.velocity * np.sin(self.theta + beta)
        A[0, 3] = self.delta_t * np.cos(self.theta + beta)

        A[1, 0] = 0
        A[1, 1] = 1
        A[1, 2] = self.delta_t * self.velocity * np.cos(self.theta + beta)
        A[1, 3] = self.delta_t * np.sin(self.theta + beta)

        A[2, 0] = 0
        A[2, 1] = 0
        A[2, 2] = 1
        A[2, 3] = self.delta_t / self.l_r * np.sin(beta)

        A[3, 0] = 0
        A[3, 1] = 0
        A[3, 2] = 0
        A[3, 3] = 1

        B[0, 0] = 0
        B[1, 0] = 0
        B[2, 0] = 0
        B[3, 0] = self.delta_t

        dev_beta = (1 / (1 + (self.l_r / (self.l_r + self.l_f) * np.tan(u_0[1])) ** 2)
                    * self.l_r / (self.l_r + self.l_f) * 1/(np.cos(u_0[1]) ** 2))

        B[0, 1] = - self.delta_t * self.velocity * np.sin(self.theta + beta) * dev_beta
        B[1, 1] = self.delta_t * self.velocity * np.cos(self.theta + beta) * dev_beta
        B[2, 1] = self.delta_t * self.velocity / self.l_r * np.cos(beta) * dev_beta
        B[3, 1] = 0

        return x_next == A @ x_now + B @ u_now

    def init_trackingMPC(self, N):
        self.N = N
        self.Q = np.eye(self.n)
        self.R = np.eye(self.m)
        self.P = 1
        self.T = 20 * self.P
        self.previous_opt_sol = {}
        self.previous_opt_sol['X'] = np.tile(self.state, (1, self.N + 1)).reshape(self.n, self.N + 1)
        self.previous_opt_sol['U'] = np.zeros((self.m, self.N))
        self.previous_opt_sol['x_s'] = self.state
        self.previous_opt_sol['u_s'] = np.zeros((self.m, 1))

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
            opti.set_initial(x_s, self.previous_opt_sol['x_s'])
            opti.set_initial(u_s, self.previous_opt_sol['u_s'])

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
            input[0, :] = self.velocity / 2 - self.velocity
            if self.acc_limits[0] > input[0, :]:
                input[0, :] = self.acc_limits[0]
            elif self.acc_limits[1] < input[0, :]:
                input[0, :] = self.acc_limits[1]
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
        input = np.zeros((self.m,1))
        self.traj_estimation = np.zeros((self.n, self.N+1))
        self.traj_estimation[:, 0] = self.state[:, 0]
        for k in range(self.N):
            beta = np.arctan(self.l_r / (self.l_r + self.l_f) * np.tan(input[1]))
            self.traj_estimation[0, k+1] = self.traj_estimation[0, k] + self.delta_t * self.traj_estimation[3, k] * np.cos(self.traj_estimation[2, k] + beta)
            self.traj_estimation[1, k+1] = self.traj_estimation[1, k] + self.delta_t * self.traj_estimation[3, k] * np.sin(self.traj_estimation[2, k] + beta)
            self.traj_estimation[2, k+1] = self.traj_estimation[2, k] + self.delta_t * self.traj_estimation[3, k] / self.l_r * np.sin(beta)
            self.traj_estimation[3, k+1] = self.traj_estimation[3, k] + self.delta_t * input[0]

    def trajectory_area_estimation(self):
        input = np.zeros((self.m, 3))
        input[1, 1] = self.steering_limits[0] * 0.5 * (np.pi / 180)
        input[1, 2] = self.steering_limits[1] * 0.5 * (np.pi / 180)
        traj_estimation= np.zeros((self.n, self.N + 1, 3))
        traj_estimation[:, 0, 0] = self.state[:, 0]
        traj_estimation[:, 0, 1] = self.state[:, 0]
        traj_estimation[:, 0, 2] = self.state[:, 0]
        circles = []
        circles.append(Point(traj_estimation[0, 0, 0], traj_estimation[1, 0, 0]).buffer(self.security_dist))
        circles.append(Point(traj_estimation[0, 0, 1], traj_estimation[1, 0, 1]).buffer(self.security_dist))
        circles.append(Point(traj_estimation[0, 0, 2], traj_estimation[1, 0, 2]).buffer(self.security_dist))
        for k in range(self.N):
            if k == 3:
                input[1, 1] = 0
                input[1, 2] = 0
            for j in range(3):
                beta = np.arctan(self.l_r / (self.l_r + self.l_f) * np.tan(input[1, j]))
                traj_estimation[0, k + 1, j] = traj_estimation[0, k, j] + self.delta_t * traj_estimation[3, k, j] * np.cos(traj_estimation[2, k, j] + beta)
                traj_estimation[1, k + 1, j] = traj_estimation[1, k, j] + self.delta_t * traj_estimation[3, k, j] * np.sin(traj_estimation[2, k, j] + beta)
                traj_estimation[2, k + 1, j] = traj_estimation[2, k, j] + self.delta_t * traj_estimation[3, k, j] / self.l_r * np.sin(beta)
                traj_estimation[3, k + 1, j] = traj_estimation[3, k, j] + self.delta_t * input[0, j]

                circles.append(Point(traj_estimation[0, k + 1, j], traj_estimation[1, k + 1, j]).buffer(self.security_dist))

        union = unary_union(circles)
        convex_hull = union.convex_hull
        if isinstance(convex_hull, GeometryCollection):
            print("Convex hull resulted in a GeometryCollection")
            print(self.state)
            error()
        else:
            self.hull = ConvexHull(convex_hull.exterior.coords[:])

    def find_safe_set(self, agents, fix_obst):
        search_safe_set = True
        only_actual = False
        steps_to_steer = 3
        list_steering_angle = [0, -np.pi/6, np.pi/6, -np.pi/3, np.pi/3]
        input = np.zeros((self.m, 1))
        input[1,:] = list_steering_angle.pop(0)
        traj_estimation = np.zeros((self.n, self.N + 1))
        feasible_traj = np.zeros((self.n, self.N + 1))
        traj_estimation[:, 0] = self.state[:, 0]
        feasible_traj[:, 0] = self.state[:, 0]
        actual_pos = np.array([self.state[0, 0], self.state[1, 0]])
        trajectories = {}
        count = 0

        obstacles = {}
        for obst in fix_obst:
            circle = Point(fix_obst[obst]['center'][0], fix_obst[obst]['center'][1]).buffer(1)  # Start with a unit circle
            ellipse = scale(circle, xfact=fix_obst[obst]['r_x'], yfact=fix_obst[obst]['r_y'])
            union = unary_union(ellipse)
            hull = union.convex_hull
            obstacles[obst] = ConvexHull(hull.exterior.coords[:])

        while search_safe_set:
            # Create the trajectory
            for k in range(self.N):
                if k == steps_to_steer:
                    input[1, :] = 0
                if k == 0:
                    input[0, :] = 0
                else:
                    acc = (0 - traj_estimation[3, k]) / self.delta_t
                    if self.acc_limits[0] > acc:
                        input[0, :] = self.acc_limits[0]
                    elif self.acc_limits[1] < acc:
                        input[0, :] = self.acc_limits[1]
                    else:
                        input[0, :] = acc
                # Safe set trajectory
                beta = np.arctan(self.l_r / (self.l_r + self.l_f) * np.tan(input[1]))
                traj_estimation[0, k + 1] = traj_estimation[0, k] + self.delta_t * traj_estimation[3, k] * np.cos(traj_estimation[2, k] + beta)
                traj_estimation[1, k + 1] = traj_estimation[1, k] + self.delta_t * traj_estimation[3, k] * np.sin(traj_estimation[2, k] + beta)
                traj_estimation[2, k + 1] = traj_estimation[2, k] + self.delta_t * traj_estimation[3, k] / self.l_r * np.sin(beta)
                traj_estimation[3, k + 1] = traj_estimation[3, k] + self.delta_t * input[0]
                # Feasible set trajectory
                beta = np.arctan(self.l_r / (self.l_r + self.l_f) * np.tan(input[1]))
                feasible_traj[0, k + 1] = feasible_traj[0, k] + self.delta_t * feasible_traj[3, k] * np.cos(feasible_traj[2, k] + beta)
                feasible_traj[1, k + 1] = feasible_traj[1, k] + self.delta_t * feasible_traj[3, k] * np.sin(feasible_traj[2, k] + beta)
                feasible_traj[2, k + 1] = feasible_traj[2, k] + self.delta_t * feasible_traj[3, k] / self.l_r * np.sin(beta)
                feasible_traj[3, k + 1] = feasible_traj[3, k] + self.delta_t * 0
            trajectories['traj '+ str(count)] = {'distance': 0,
                                                'center': np.array([traj_estimation[0, -1], traj_estimation[1, -1]])}
            pred_pos = np.array([traj_estimation[0, -1], traj_estimation[1, -1]])
            radius_safe_set = 4
            #if not only_actual:
            for id_agent in agents:
                if agents[id_agent].security_dist != 0:
                    trajectories['traj ' + str(count)]['distance'] += 1 / np.linalg.norm(pred_pos - agents[id_agent].position)
                    A_hull = agents[id_agent].hull.equations[:, :-1]
                    b_hull = agents[id_agent].hull.equations[:, -1]
                    if np.all(np.dot(A_hull, pred_pos) + b_hull <= 0):  # Inside the convex hull
                        radius_safe_set = 0
                    else:
                        point = Point(traj_estimation[0, -1], traj_estimation[1, -1])
                        dist = point.distance(Polygon(agents[id_agent].hull.points))
                        if 2 <= dist:
                            if dist < radius_safe_set:
                                radius_safe_set = dist
                        else:
                            radius_safe_set = 0

            for obst in obstacles:
                A_hull = obstacles[obst].equations[:, :-1]
                b_hull = obstacles[obst].equations[:, -1]
                if np.all(np.dot(A_hull, pred_pos) + b_hull <= 0):  # Inside the convex hull
                    radius_safe_set = 0
                    #trajectories['traj ' + str(count)]['distance'] += 10
                else:
                    point = Point(pred_pos[0], pred_pos[1])
                    dist = point.distance(Polygon(obstacles[obst].points))
                    if 2 <= dist:
                        if dist < radius_safe_set:
                            radius_safe_set = dist
                    else:
                        radius_safe_set = 0

            for id in self.fix_obstacles:
                A_hull = self.fix_obstacles[id].equations[:, :-1]
                b_hull = self.fix_obstacles[id].equations[:, -1]
                if np.all(np.dot(A_hull, pred_pos) + b_hull <= 0):  # Inside the convex hull
                    radius_safe_set = 0
                    #trajectories['traj ' + str(count)]['distance'] += 10

            """else:
                for id_agent in agents:
                    if agents[id_agent].security_dist != 0:
                        trajectories['traj ' + str(count)]['distance'] += 1 / np.linalg.norm(pred_pos - agents[id_agent].position)
                        if np.linalg.norm(actual_pos - agents[id_agent].position) < agents[id_agent].security_dist:
                            radius_safe_set = 0
                        else:
                            dist = np.linalg.norm(actual_pos - agents[id_agent].position)
                            if 2 <= dist:
                                if dist < radius_safe_set:
                                    radius_safe_set = dist
                            else:
                                radius_safe_set = 0

                for obst in obstacles:
                    A_hull = obstacles[obst].equations[:, :-1]
                    b_hull = obstacles[obst].equations[:, -1]
                    if np.all(np.dot(A_hull, actual_pos) + b_hull <= 0):  # Inside the convex hull
                        radius_safe_set = 0
                        #trajectories['traj ' + str(count)]['distance'] += 10
                    else:
                        point = Point(actual_pos[0], actual_pos[1])
                        dist = point.distance(Polygon(obstacles[obst].points))
                        if 2 <= dist:
                            if dist < radius_safe_set:
                                radius_safe_set = dist
                        else:
                            radius_safe_set = 0

                for id in self.fix_obstacles:
                    A_hull = self.fix_obstacles[id].equations[:, :-1]
                    b_hull = self.fix_obstacles[id].equations[:, -1]
                    if np.all(np.dot(A_hull, actual_pos) + b_hull <= 0):  # Inside the convex hull
                        radius_safe_set = 0
                        #trajectories['traj ' + str(count)]['distance'] += 10"""
            count += 1

            if radius_safe_set == 0:
                # Do somthing with input
                if len(list_steering_angle) > 0:
                    input[1, :] = list_steering_angle.pop(0)
                else:
                    list_steering_angle = [0, -np.pi/6, np.pi/6, -np.pi/3, np.pi/3]
                    input[1, :] = list_steering_angle.pop(0)
                    if steps_to_steer <= self.N - 3:
                        steps_to_steer = steps_to_steer + 3
                    else:
                        radius_safe_set = None
                        search_safe_set = False
                    """elif not only_actual:
                        only_actual = True
                        steps_to_steer = 3"""
            if radius_safe_set is not None and 2 <= radius_safe_set <= 4:
                search_safe_set = False
                center = np.array([[traj_estimation[0, -1]], [traj_estimation[1, -1]]])

        if radius_safe_set is not None and radius_safe_set < 0:
            error()

        if radius_safe_set is None:
            """l_square = 3
            safe_zone_not_find = True
            while safe_zone_not_find:
                points = np.zeros((4, 2))
                points[0, 0] = self.x - l_square
                points[0, 1] = self.y - l_square
                points[1, 0] = self.x - l_square
                points[1, 1] = self.y + l_square
                points[2, 0] = self.x + l_square
                points[2, 1] = self.y + l_square
                points[3, 0] = self.x + l_square
                points[3, 1] = self.y - l_square
                Square = ConvexHull(points)
                A, b, find_it = self.find_feasible_convex_set(Square, agents, obstacles)
                if find_it:
                    circle, good_size, opt_var = self.largest_ellipse(A, b, agents, obstacles, True)
                    # Bisogna controllaare la grandezza del cerchio e se non interseca niente!
                    if good_size:
                        safe_zone_not_find = False
                        radius_safe_set = opt_var['a_opt']
                        center = np.array([[opt_var['x_c_opt']], [opt_var['y_c_opt']]])
                    else:
                        l_square = l_square + 0.2 * l_square
                else:
                    l_square = l_square + 0.2 * l_square"""

            """# This is not optimal, should find a better solution!
            for k in range(self.N):
                input[1, :] = 0
                acc = (0 - traj_estimation[3, k]) / self.delta_t
                if self.acc_limits[0] > acc:
                    input[0, :] = self.acc_limits[0]
                elif self.acc_limits[1] < acc:
                    input[0, :] = self.acc_limits[1]
                else:
                    input[0, :] = acc
                beta = np.arctan(self.l_r / (self.l_r + self.l_f) * np.tan(input[1]))
                traj_estimation[0, k + 1] = traj_estimation[0, k] + self.delta_t * traj_estimation[3, k] * np.cos(
                    traj_estimation[2, k] + beta)
                traj_estimation[1, k + 1] = traj_estimation[1, k] + self.delta_t * traj_estimation[3, k] * np.sin(
                    traj_estimation[2, k] + beta)
                traj_estimation[2, k + 1] = traj_estimation[2, k] + self.delta_t * traj_estimation[
                    3, k] / self.l_r * np.sin(beta)
                traj_estimation[3, k + 1] = traj_estimation[3, k] + self.delta_t * input[0]
            radius_safe_set = 2
            center = np.array([[traj_estimation[0, -1]], [traj_estimation[1, -1]]])
            """

            """dist = []
            for i in range(len(trajectories)):
                dist.append(trajectories['traj '+str(i)]['distance'])
            id = dist.index(min(dist))"""

            center = np.array([[self.x], [self.y]])

            self.safe_set = {'radius': 2,
                             'center': center,
                             'traj': traj_estimation,
                             'computed with traj': False}
        else:
            self.safe_set = {'radius': radius_safe_set,
                             'center': center,
                             'traj': traj_estimation,
                             'computed with traj': True}

        # Compute the feasible set
        num_points = 50  # Number of points to sample
        t = np.linspace(0, 2 * np.pi, num_points)
        circle_points = np.zeros((num_points * (self.N + 1), 2))
        self.trajecotry_estimation()
        for k in range(self.N + 1):
            for i in range(num_points):
                x = float(feasible_traj[0, k]) + 4 * np.cos(t[i])
                y = float(feasible_traj[1, k]) + 4 * np.sin(t[i])
                circle_points[k * num_points + i, :] = [x, y]
        hull = ConvexHull(circle_points)

        A, b, find_it = self.find_feasible_convex_set(hull, agents, obstacles) #Return the ineq A @ x + b <= 0 that describes the set
        if find_it:
            self.hull, good_size, opt_var = self.largest_ellipse(A, b, agents, obstacles, False)
            self.safe_set['feasible and safe set are the same'] = False
            if not good_size:
                print('The function largest_ellipse have not find a suitable ellipse')
                # Compute the feasible set
                num_points = 50  # Number of points to sample
                t = np.linspace(0, 2 * np.pi, num_points)
                """circle_points = np.zeros((num_points * (self.N + 1), 2))
                self.trajecotry_estimation()
                for k in range(self.N + 1):
                    for i in range(num_points):
                        x = float(self.traj_estimation[0, k]) + 2 * np.cos(t[i])
                        y = float(self.traj_estimation[1, k]) + 2 * np.sin(t[i])
                        circle_points[k * num_points + i, :] = [x, y]"""
                circle_points = np.zeros((num_points, 2))
                for i in range(num_points):
                    x = float(self.safe_set['center'][0]) + self.safe_set['radius'] * np.cos(t[i])
                    y = float(self.safe_set['center'][1]) + self.safe_set['radius'] * np.sin(t[i])
                    circle_points[i, :] = [x, y]
                """circle_points = np.zeros((num_points, 2))
                for i in range(num_points):
                    x = float(self.x) + 3 * np.cos(t[i])
                    y = float(self.y) + 3 * np.sin(t[i])
                    circle_points[i, :] = [x, y]"""
                self.hull = ConvexHull(circle_points)
                self.safe_set['feasible and safe set are the same'] = True
        else: # The traj is all in the forbidden areas, therefore we take the safe set as feasible set
            """# Generate points on the ellipse
            num_points = 50  # Number of points to sample
            t = np.linspace(0, 2 * np.pi, num_points)
            points = np.zeros((num_points, 2))
            for i in range(num_points):
                x = float(self.safe_set['center'][0]) + float(self.safe_set['radius']) * np.cos(t[i])
                y = float(self.safe_set['center'][1]) + float(self.safe_set['radius']) * np.sin(t[i])
                points[i, :] = [x, y]
            # Compute the ConvexHull
            self.hull = ConvexHull(points)"""
            # Compute the feasible set
            num_points = 50  # Number of points to sample
            t = np.linspace(0, 2 * np.pi, num_points)
            """circle_points = np.zeros((num_points * (self.N + 1), 2))
            self.trajecotry_estimation()
            for k in range(self.N + 1):
                for i in range(num_points):
                    x = float(self.traj_estimation[0, k]) + 2 * np.cos(t[i])
                    y = float(self.traj_estimation[1, k]) + 2 * np.sin(t[i])
                    circle_points[k * num_points + i, :] = [x, y]"""
            circle_points = np.zeros((num_points, 2))
            for i in range(num_points):
                x = float(self.safe_set['center'][0]) + self.safe_set['radius'] * np.cos(t[i])
                y = float(self.safe_set['center'][1]) + self.safe_set['radius'] * np.sin(t[i])
                circle_points[i, :] = [x, y]
            """circle_points = np.zeros((num_points, 2))
            for i in range(num_points):
                x = float(self.x) + 3 * np.cos(t[i])
                y = float(self.y) + 3 * np.sin(t[i])
                circle_points[i, :] = [x, y]"""
            self.hull = ConvexHull(circle_points)
            self.safe_set['feasible and safe set are the same'] = True

    def find_feasible_convex_set(self, hull, agents, obstacles):

        # Substract the intersections with trajecotry area of other agents
        poly1 = Polygon(hull.points[hull.vertices])
        for id_agent in agents:
            if agents[id_agent].security_dist != 0:
                poly2 = Polygon(agents[id_agent].hull.points[agents[id_agent].hull.vertices])
                if not poly1.intersection(poly2).is_empty:
                    poly1 = poly1.difference(poly2)
        # Substract the intersections with obstacles
        for obst in obstacles:
            poly2 = Polygon(obstacles[obst].points[obstacles[obst].vertices])
            if not poly1.intersection(poly2).is_empty:
                poly1 = poly1.difference(poly2)
        # Substract the intersections with outsides of the road
        for id in self.fix_obstacles:
            poly2 = Polygon(self.fix_obstacles[id].points[self.fix_obstacles[id].vertices])
            if not poly1.intersection(poly2).is_empty:
                poly1 = poly1.difference(poly2)
        # Check if you obtain an empty polygon
        if poly1.is_empty:
            return [], [], False

        if isinstance(poly1, Polygon):
            if poly1.area <= 10:
                return [], [], False
            else:
                hull = ConvexHull(poly1.exterior.coords[:])
        elif isinstance(poly1, MultiPolygon):
            areas = []
            for polygon in poly1.geoms:
                areas.append(polygon.area)
            polygon_1 = poly1.geoms[areas.index(max(areas))]
            if polygon_1.area <= 10:
                return [], [], False
            else:
                hull = ConvexHull(polygon_1.exterior.coords[:])
        A_hull = hull.equations[:, :-1]
        b_hull = hull.equations[:, -1]

        return A_hull, b_hull, True

    def largest_ellipse(self, A, b, agents, obstacles, search_safe_set):
        opti = ca.Opti()
        # Decision variables: ellipse parameters
        x_c = opti.variable()  # Ellipse center x
        y_c = opti.variable()  # Ellipse center y
        a_axis = opti.variable() # Semi-major axis
        b_axis = opti.variable() # Semi-minor axis
        theta = opti.variable() # Orientation angle

        # Constraints: A * ellipse_point <= b for all t
        num_t_points = 50  # Number of discretization points for the ellipse boundary
        t_values = np.linspace(0, 2 * np.pi, num_t_points)
        for t in t_values:
            ellipse_x = x_c + a_axis * ca.cos(theta) * ca.cos(t) - b_axis * ca.sin(theta) * ca.sin(t)
            ellipse_y = y_c + a_axis * ca.sin(theta) * ca.cos(t) + b_axis * ca.cos(theta) * ca.sin(t)
            opti.subject_to(A @ ca.vertcat(ellipse_x, ellipse_y) + b <= 0)

        if search_safe_set:
            opti.subject_to(a_axis == b_axis)

        # Objective: Maximize area (a * b)
        opti.minimize(-a_axis * b_axis)  # Negative for maximization in CasADi

        opti.set_initial(x_c, self.x)
        opti.set_initial(y_c, self.y)
        opti.set_initial(a_axis, 1)
        opti.set_initial(b_axis, 1)
        opti.set_initial(theta, 0)

        try:
            opti.solver('ipopt')
            sol = opti.solve()
        except Exception as e:
            return [], False, []

        x_c_opt = sol.value(x_c)
        y_c_opt = sol.value(y_c)
        a_opt = sol.value(a_axis)
        b_opt = sol.value(b_axis)
        theta_opt = sol.value(theta)

        good_size = False
        while not good_size:
            # Generate points on the ellipse
            num_points = 50  # Number of points to sample
            t = np.linspace(0, 2 * np.pi, num_points)
            ellipse_points = np.zeros((num_points, 2))
            for i in range(num_points):
                x = x_c_opt + a_opt * np.cos(t[i]) * np.cos(theta_opt) - b_opt * np.sin(t[i]) * np.sin(theta_opt)
                y = y_c_opt + a_opt * np.cos(t[i]) * np.sin(theta_opt) + b_opt * np.sin(t[i]) * np.cos(theta_opt)
                ellipse_points[i, :] = [x, y]
            # Compute the ConvexHull
            new_hull = ConvexHull(ellipse_points)
            good_size = True
            # Substract the intersections with trajecotry area of other agents
            poly1 = Polygon(new_hull.points[new_hull.vertices])
            for id_agent in agents:
                if agents[id_agent].security_dist != 0:
                    poly2 = Polygon(agents[id_agent].hull.points[agents[id_agent].hull.vertices])
                    if not poly1.intersection(poly2).is_empty:
                        good_size = False
            # Substract the intersections with obstacles
            for obst in obstacles:
                poly2 = Polygon(obstacles[obst].points[obstacles[obst].vertices])
                if not poly1.intersection(poly2).is_empty:
                    good_size = False
            # Substract the intersections with outsides of the road
            for id in self.fix_obstacles:
                poly2 = Polygon(self.fix_obstacles[id].points[self.fix_obstacles[id].vertices])
                if not poly1.intersection(poly2).is_empty:
                    good_size = False
            # Check if you obtain an empty polygon
            if poly1.is_empty:
                error()
            elif np.pi * a_opt * b_opt <= 4:
                if search_safe_set:
                    return [], False, []
                else:
                    """good_size = True
                    a_opt = float(self.safe_set['radius'])
                    b_opt = float(self.safe_set['radius'])
                    x_c_opt = float(self.safe_set['center'][0])
                    y_c_opt = float(self.safe_set['center'][1])
                    # Generate points on the ellipse
                    num_points = 50  # Number of points to sample
                    t = np.linspace(0, 2 * np.pi, num_points)
                    ellipse_points = np.zeros((num_points, 2))
                    for i in range(num_points):
                        x = x_c_opt + a_opt * np.cos(t[i]) * np.cos(theta_opt) - b_opt * np.sin(t[i]) * np.sin(theta_opt)
                        y = y_c_opt + a_opt * np.cos(t[i]) * np.sin(theta_opt) + b_opt * np.sin(t[i]) * np.cos(theta_opt)
                        ellipse_points[i, :] = [x, y]
                    # Compute the ConvexHull
                    new_hull = ConvexHull(ellipse_points)"""
                    return [], False, []
            elif not good_size:
                a_opt = 0.99 * a_opt
                b_opt = 0.99 * b_opt

        opt_varibles = {'x_c_opt': x_c_opt,
                        'y_c_opt': y_c_opt,
                        'theta_opt': theta_opt,
                        'a_opt': a_opt,
                        'b_opt': b_opt}

        return new_hull, good_size, opt_varibles

    def MPC_LLM(self, agents, circular_obstacles, t, llm, info):
        nr_ineq_const = len(llm.OD["inequality_constraints"])
        nr_eq_const = len(llm.OD["equality_constraints"])

        A_hull = self.hull.equations[:, :-1]
        b_hull = self.hull.equations[:, -1]

        A = np.zeros((np.shape(A_hull)[0] + 4, self.n))
        b = np.zeros((np.shape(A_hull)[0] + 4, 1))
        A[0:np.shape(A_hull)[0], 0:2] = A_hull
        A[-4:, :] = self.A_x[-4:, :]
        b[0:np.shape(A_hull)[0], 0] = b_hull
        b[-4:, :] = -self.b_x[-4:, :]
        b[-2, :] = -info['street speed limit']

        opti = ca.Opti()

        X = opti.variable(self.n, self.N + 1)
        U = opti.variable(self.m, self.N)
        psi_b_x = opti.variable(np.size(b), self.N_SF + 1)
        psi_v = opti.variable(1,1)
        psi_f = opti.variable(1,1)
        # LLM cost and constraints
        if nr_ineq_const + nr_eq_const >= 1:
            epsilon_LLM_constraints = opti.variable(nr_ineq_const + nr_eq_const, self.N + 1)
        else:
            epsilon_LLM_constraints = 0

        cost = 0
        for k in range(self.N):
            # LLM cost and constraints
            if nr_ineq_const + nr_eq_const >= 1:
                cost, opti = self.soft_OD_output(opti, cost, llm, X[:, k], U[:, k], agents,
                                                 epsilon_LLM_constraints[:, k], t)
            else:
                cost, opti = self.OD_output(opti, cost, llm, X[:, k], U[:, k], agents, t)

            cost += (psi_b_x[:, k].T @ psi_b_x[:, k])
            for j in range(np.size(b)):
                opti.subject_to(0 <= psi_b_x[j, k])
            # State and Input constraints
            opti.subject_to(A @ X[:, k] + b <= psi_b_x[:, k])
            opti.subject_to(self.A_u @ U[:, k] <= self.b_u)

            # System dynamics constraints
            opti.subject_to(self.dynamics_constraints(X[:, k + 1], X[:, k], U[:, k]))

        # LLM cost and constraints
        if nr_ineq_const + nr_eq_const >= 1:
            cost, opti = self.soft_OD_output(opti, cost, llm, X[:, -1], U[:, -1], agents,
                                             epsilon_LLM_constraints[:, -1], t)
        else:
            cost, opti = self.OD_output(opti, cost, llm, X[:,-1], U[:,-1], agents, t) # U[-1] and X[-1] do not corresponds!

        # Initial state
        opti.subject_to(X[:, 0] == self.state)

        # Terminal Constraints
        cost += (psi_b_x[:, -1].T @ psi_b_x[:, -1])
        for j in range(np.size(b)):
            opti.subject_to(0 <= psi_b_x[j, -1])
        opti.subject_to(A @ X[:, -1] + b <= psi_b_x[:, -1])

        # Safe terminal set
        px_S = self.safe_set['center'][0]
        py_S = self.safe_set['center'][1]
        R_S = self.safe_set['radius']
        opti.subject_to((X[0, -1] - px_S) ** 2 + (X[1, -1] - py_S) ** 2 - R_S ** 2 <= psi_f)
        opti.subject_to(X[3, -1] == psi_v)
        opti.subject_to(0 <= psi_f)
        opti.subject_to(0 <= psi_v)
        cost += psi_f ** 2 + psi_v ** 2

        """# Constraint for the steady state
        if self.terminal_set:
            opti.subject_to(X[3, -1] == 0)

        # Agents avoidance
        if len(agents) >= 1:
            for id_agent in agents:
                agents[id_agent].trajecotry_estimation()
                for k in range(self.N_SF + 1):
                    if agents[id_agent].security_dist != 0:
                        opti.subject_to(self.agents_constraints_traj(X[:, k], agents[id_agent].traj_estimation[:, k],
                                                         agents[id_agent]) >= 0)
                        #opti.subject_to(self.agents_constraints_area(X[:, k], agents[id_agent].traj_estimation[:, k],
                        #                                             agents[id_agent]))

        # Avoidance fix obstacles
        if len(circular_obstacles) != 0:
            for id_obst in circular_obstacles:
                for k in range(self.N_SF + 1):
                    diff_x = (X[0, k] - circular_obstacles[id_obst]['center'][0]) ** 2 / (circular_obstacles[id_obst]['r_x']) ** 2
                    diff_y = (X[1, k] - circular_obstacles[id_obst]['center'][1]) ** 2 / (circular_obstacles[id_obst]['r_y']) ** 2
                    opti.subject_to(diff_x + diff_y >= 1)"""

        opti.minimize(cost)
        # Solve the optimization problem
        if t == 0:
            self.trajecotry_estimation()
            opti.set_initial(X, self.traj_estimation)
            opti.set_initial(U, np.zeros((self.m, self.N)))
            opti.set_initial(psi_b_x, np.zeros((np.size(b), self.N + 1)))
            opti.set_initial(psi_v, 0)
            opti.set_initial(psi_f, 0)
        else:
            #opti.set_initial(X, self.previous_opt_sol_SF['X'])
            #opti.set_initial(U, self.previous_opt_sol_SF['U'])
            self.trajecotry_estimation()
            opti.set_initial(X, self.traj_estimation)
            opti.set_initial(U, np.zeros((self.m, self.N)))
            opti.set_initial(psi_b_x, np.zeros((np.size(b), self.N + 1)))
            opti.set_initial(psi_v, 0)
            opti.set_initial(psi_f, 0)

        try:
            opti.solver('ipopt')
            sol = opti.solve()
        except Exception as e:
            print(f"Optimization failed: {e}")
            """try:
                self.trajecotry_estimation()
                opti.set_initial(X, self.traj_estimation)
                opti.set_initial(U, np.zeros((self.m, self.N)))
                opti.solver('ipopt')
                sol = opti.solve()
            except Exception as e:
                print(f"Optimization failed: {e}")"""

        if opti.stats()['success']:
            self.previous_opt_sol['X'] = sol.value(X)
            self.previous_opt_sol['U'] = sol.value(U)
            self.previous_opt_sol['psi_b_x'] = sol.value(psi_b_x)
            self.previous_opt_sol['psi_v'] = sol.value(psi_v)
            self.previous_opt_sol['psi_f'] = sol.value(psi_f)
            if nr_ineq_const + nr_eq_const >= 1:
                self.previous_opt_sol['epsilon_LLM_constraints'] = sol.value(epsilon_LLM_constraints).reshape(nr_ineq_const + nr_eq_const, self.N + 1)
            else:
                self.previous_opt_sol['epsilon_LLM_constraints'] = epsilon_LLM_constraints
            self.previous_opt_sol['Cost'] = sol.value(cost)
            input = sol.value(U)[:, 0].reshape((self.m, 1))
            self.success_solver_MPC_LLM = True

            """for k in range(self.N_SF + 1):  # Think if +1 make sense...
                pos_slack = np.linalg.norm(self.previous_opt_sol['psi_b_x'][:, k])
                if pos_slack > 1:
                    error()
            if abs(self.previous_opt_sol['psi_v']) > 0:
                error()
            if abs(self.previous_opt_sol['psi_f']) > 0:
                error()"""
        else:
            self.previous_opt_sol['Cost'] = 10
            if not self.SF_active:
                # Brake
                input = np.zeros((self.m, 1))
                input[0, :] = (0 - self.velocity)
                if self.acc_limits[0] > input[0, :]:
                    input[0, :] = self.acc_limits[0]
                elif self.acc_limits[1] < input[0, :]:
                    input[0, :] = self.acc_limits[1]
                input[1, :] = 0
            else:
                # Previous solution
                input = self.previous_opt_sol['U'][:, 1]

            print('LLM MPC solver failed. Use a default u_L = ', input)
            self.success_solver_MPC_LLM = False

        return input

    def OD_output(self, opti, obj, llm, X, U, agents, t):

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

    def Control_Module(self, agents, circular_obstacles, t, llm):
        nr_ineq_const = len(llm.OD["inequality_constraints"])
        nr_eq_const = len(llm.OD["equality_constraints"])

        opti = ca.Opti()

        X = opti.variable(self.n, self.N + 1)
        U = opti.variable(self.m, self.N)
        #psi_b_x = opti.variable(np.size(self.b_x), self.N + 1)
        #weight_psi_b_x = 100
        if nr_ineq_const + nr_eq_const >= 1:
            epsilon_LLM_constraints = opti.variable(nr_ineq_const + nr_eq_const, self.N + 1)
        else:
            epsilon_LLM_constraints = 0

        cost = 0
        for k in range(self.N):
            # State and Input constraints
            #opti.subject_to(self.A_x @ X[:, k + 1] <= self.b_x + psi_b_x[:,k])
            opti.subject_to(self.A_u @ U[:, k] <= self.b_u)
            # LLM cost and constraints
            if nr_ineq_const + nr_eq_const >= 1:
                cost, opti = self.soft_OD_output(opti, cost, llm, X[:,k], U[:,k], agents, epsilon_LLM_constraints[:, k], t)
            else:
                cost, opti = self.OD_output(opti, cost, llm, X[:, k], U[:, k], agents, t)
            # System dynamics constraints
            opti.subject_to(self.dynamics_constraints(X[:, k + 1], X[:, k], U[:, k]))

            #cost += weight_psi_b_x * psi_b_x[:, k].T @ psi_b_x[:, k]

        # State and Input constraints
        #opti.subject_to(self.A_x @ X[:, -1] <= self.b_x + psi_b_x[:,-1])
        opti.subject_to(self.A_u @ U[:, -1] <= self.b_u)

        #cost += weight_psi_b_x * psi_b_x[:, -1].T @ psi_b_x[:, -1]

        # cost, opti = self.OD_output(opti, cost, llm, X[:, -1], U[:, -1], agents)
        if nr_ineq_const + nr_eq_const >= 1:
            cost, opti = self.soft_OD_output(opti, cost, llm, X[:, -1], U[:, -1], agents, epsilon_LLM_constraints[:, -1], t)
        else:
            cost, opti = self.OD_output(opti, cost, llm, X[:, -1], U[:, -1], agents, t)

        # Initial state
        opti.subject_to(X[:, 0] == self.state)

        opti.minimize(cost)

        # Solve the optimization problem
        if t == 0:
            self.trajecotry_estimation()
            opti.set_initial(X, self.traj_estimation)
            opti.set_initial(U, np.zeros((self.m, self.N)))
            #opti.set_initial(psi_b_x, np.zeros((np.size(self.b_x), self.N_SF + 1)))
        else:
            self.trajecotry_estimation()
            opti.set_initial(X, self.traj_estimation)
            opti.set_initial(U, np.zeros((self.m, self.N)))
            #opti.set_initial(X, self.previous_opt_sol_SF['X'])
            #opti.set_initial(U, self.previous_opt_sol_SF['U'])
            #opti.set_initial(psi_b_x, self.previous_opt_sol_SF['psi_b_x'])
        try:
            opti.solver('ipopt')
            sol = opti.solve()
        except Exception as e:
            print(f"Optimization failed: {e}")

        if opti.stats()['success']:
            self.previous_opt_sol['X'] = sol.value(X)
            self.previous_opt_sol['U'] = sol.value(U)
            #self.previous_opt_sol_SF['psi_b_x'] = sol.value(psi_b_x)
            if nr_ineq_const + nr_eq_const >= 1:
                self.previous_opt_sol['epsilon_LLM_constraints'] = sol.value(epsilon_LLM_constraints).reshape(nr_ineq_const + nr_eq_const, self.N + 1)
            else:
                self.previous_opt_sol['epsilon_LLM_constraints'] = epsilon_LLM_constraints
            self.previous_opt_sol['Cost'] = sol.value(cost)
            input = sol.value(U)[:, 0].reshape((self.m, 1))
            self.success_solver_MPC_LLM = True
        else:
            # Previous solution
            input = self.previous_opt_sol['U'][:, 1]

            print('LLM MPC solver failed. Use a default u_L = ', input)
            self.success_solver_MPC_LLM = False

        return input

    def soft_OD_output(self, opti, obj, llm, X, U, agents, epsilon, t):

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
        count = 0
        for i in range(nr_ineq_const):
            const = ineq_constraints[i] + """+ epsilon[count]"""
            opti.subject_to(eval(const) <= 0)
            obj += 10 * epsilon[count] ** 2
            count += 1
        for i in range(nr_eq_const):
            const = eq_constraints[i] + """+ epsilon[count]"""
            opti.subject_to(eval(const)== 0)
            obj += 10 * epsilon[count] ** 2
            count += 1

        return obj, opti

    def SF(self, u_lernt, agents, circular_obstacles, t, llm, info):

        opti = ca.Opti()

        X = opti.variable(self.n, self.N_SF + 1)
        U = opti.variable(self.m, self.N_SF)
        x_s = opti.variable(self.n, 1)
        u_s = opti.variable(self.m, 1)
        if self.use_LLM_output_in_SF:
            nr_ineq_const = len(llm.OD["inequality_constraints"])
            nr_eq_const = len(llm.OD["equality_constraints"])
            epsilon = opti.variable(nr_ineq_const + nr_eq_const, self.N_SF + 1)

        cost = ca.norm_2(u_lernt - U[:, 0]) ** 2 + ca.norm_2(self.previous_opt_sol['X'][:, -1] - X[:, -1]) ** 2

        for k in range(self.N_SF):
            # State and Input constraints
            opti.subject_to(self.A_x @ X[:, k + 1] <= self.b_x)
            opti.subject_to(self.A_u @ U[:, k] <= self.b_u)

            # LLM constraints
            if self.use_LLM_output_in_SF:
                cost, opti = self.soft_OD_const(opti, cost, llm, X[:,k], U[:,k], agents, epsilon[:, k])
            else:
                opti.subject_to(X[3, k + 1] <= info['street speed limit'])

            # System dynamics constraints
            opti.subject_to(self.dynamics_constraints(X[:, k + 1], X[:, k], U[:, k]))

        # Initial state
        opti.subject_to(X[:, 0] == self.state)

        # LLM constraints
        if self.use_LLM_output_in_SF:
            cost, opti = self.soft_OD_const(opti, cost, llm, X[:, -1], U[:, -1], agents, epsilon[:, -1])
        else:
            opti.subject_to(X[3, -1] <= info['street speed limit'])

        # Agents avoidance
        if len(agents) >= 1:
            for id_agent in agents:
                agents[id_agent].trajecotry_estimation()
                for k in range(self.N_SF + 1):
                    if agents[id_agent].security_dist != 0:
                        #opti.subject_to(self.agents_constraints(X[:, k], agents[id_agent]) >= 0)
                        opti.subject_to(self.agents_constraints_traj(X[:, k], agents[id_agent].traj_estimation[:, k], agents[id_agent]) >= 0)
                        #opti.subject_to(self.agents_constraints_area(X[:, k], agents[id_agent].traj_estimation[:, k],
                        #                                             agents[id_agent]))

        # Obstacle avoidance
        if len(circular_obstacles) != 0:
            for id_obst in circular_obstacles:
                for k in range(self.N_SF + 1):
                    diff_x = (X[0, k] - circular_obstacles[id_obst]['center'][0]) ** 2 / (circular_obstacles[id_obst]['r_x']) ** 2
                    diff_y = (X[1, k] - circular_obstacles[id_obst]['center'][1]) ** 2 / (circular_obstacles[id_obst]['r_y']) ** 2
                    opti.subject_to(diff_x + diff_y >= 1)

        # Constraint for the steady state
        opti.subject_to(self.A_x @ x_s <= self.b_x)
        opti.subject_to(self.A_u @ u_s <= self.b_u)
        opti.subject_to(self.dynamics_constraints(x_s, x_s, u_s))
        # Terminal constraints
        opti.subject_to(X[:, -1] == x_s)  # x(N) == x_s

        opti.minimize(cost)

        # Solve the optimization problem
        if t == 0:  # or agents[f'{i}'].target_status
            self.trajecotry_estimation()
            opti.set_initial(X, self.traj_estimation)
            opti.set_initial(U, np.zeros((self.m, self.N)))
            opti.set_initial(x_s, self.traj_estimation[:, -1])
            opti.set_initial(u_s, np.array([[0 - self.traj_estimation[3, -1]], [0]]))
        else:
            opti.set_initial(X, self.previous_opt_sol_SF['X'])
            opti.set_initial(U, self.previous_opt_sol_SF['U'])
            opti.set_initial(x_s, self.previous_opt_sol_SF['x_s'])
            opti.set_initial(u_s, self.previous_opt_sol_SF['u_s'])

        try:
            opti.solver('ipopt')
            sol = opti.solve()
        except Exception as e:
            print(f"Optimization failed: {e}")

        if opti.stats()['success']:
            self.previous_opt_sol_SF['X'] = sol.value(X)
            self.previous_opt_sol_SF['U'] = sol.value(U)
            self.previous_opt_sol_SF['x_s'] = sol.value(x_s)
            self.previous_opt_sol_SF['u_s'] = sol.value(u_s)
            self.previous_opt_sol_SF['Cost'] = sol.value(np.linalg.norm(u_lernt - sol.value(U)[:, 0]) ** 2)

            input = sol.value(U)[:, 0].reshape((self.m, 1))
            self.success_solver_SF = True

        else:
            input = np.zeros((self.m, 1))
            if self.acc_limits[0] > 0 - self.velocity:
                input[0, :] = self.acc_limits[1]
            else:
                input[0, :] = 0 - self.velocity
            input[1, :] = 0

            print('SF solver failed. Use a default input u = ', input)
            self.success_solver_SF = False

        return input

    def soft_SF(self, u_lernt, agents, circular_obstacles, t, info):

        if self.state[3] == 0:
            self.state[3] = 0.001

        opti = ca.Opti()

        X = opti.variable(self.n, self.N_SF + 1)
        U = opti.variable(self.m, self.N_SF)
        x_s = opti.variable(self.n, 1)
        u_s = opti.variable(self.m, 1)
        psi_b_x = opti.variable(np.size(self.b_x), self.N_SF + 1)
        weight_psi_b_x = 10
        psi_b_x_s = opti.variable(np.size(self.b_x), 1)
        weight_psi_b_x_s = 10
        psi_v_limit = opti.variable(1, self.N_SF + 1)
        weight_psi_v_limit = 10
        psi_agents = opti.variable(len(agents), self.N_SF + 1)
        weight_psi_agents = 100
        psi_obst = opti.variable(len(circular_obstacles), self.N_SF + 1)
        weight_psi_obst = 10
        weight_input = 1
        weight_final_state = 1

        cost = 0
        for k in range(self.N_SF):
            cost += (weight_psi_b_x * psi_b_x[:, k].T @ psi_b_x[:, k] +
                     weight_psi_agents * psi_agents[:, k].T @ psi_agents[:, k] +
                     weight_psi_obst * psi_obst[:, k].T @ psi_obst[:, k] +
                     weight_psi_v_limit * psi_v_limit[:, k].T @ psi_v_limit[:, k])

            # State and Input constraints
            opti.subject_to(self.A_x @ X[:, k + 1] <= self.b_x + psi_b_x[:, k + 1])
            opti.subject_to(self.A_u @ U[:, k] <= self.b_u)

            opti.subject_to(X[3, k + 1] <= info['street speed limit'] + psi_v_limit[:, k + 1])

            # System dynamics constraints
            opti.subject_to(self.dynamics_constraints(X[:, k + 1], X[:, k], U[:, k]))

        # Initial state
        opti.subject_to(X[:, 0] == self.state)

        # Agents avoidance
        if len(agents) >= 1:
            for nr_agent, id_agent in enumerate(agents):
                agents[id_agent].trajecotry_estimation()
                for k in range(self.N_SF + 1):
                    if agents[id_agent].security_dist != 0:
                        #opti.subject_to(self.agents_constraints(X[:, k], agents[id_agent]) >= 0)
                        opti.subject_to(self.agents_constraints_traj(X[:, k], agents[id_agent].traj_estimation[:, k],
                                                                     agents[id_agent]) >= psi_agents[nr_agent, k])
                        #opti.subject_to(self.agents_constraints_area(X[:, k], agents[id_agent].traj_estimation[:, k],
                        #                                             agents[id_agent]) + psi_agents[nr_agent, k])

        # Obstacle avoidance
        if len(circular_obstacles) != 0:
            for nr_obs, id_obst in enumerate(circular_obstacles):
                for k in range(self.N_SF + 1):
                    diff_x = (X[0, k] - circular_obstacles[id_obst]['center'][0]) ** 2 / (
                    circular_obstacles[id_obst]['r_x']) ** 2
                    diff_y = (X[1, k] - circular_obstacles[id_obst]['center'][1]) ** 2 / (
                    circular_obstacles[id_obst]['r_y']) ** 2
                    opti.subject_to(diff_x + diff_y >= 1 + psi_obst[nr_obs, k])

        #opti.subject_to(X[3, -1] <= info['street speed limit'] + psi_v_limit[:, -1])
        # Constraint for the steady state
        opti.subject_to(self.A_x @ x_s <= self.b_x + psi_b_x_s)
        opti.subject_to(self.A_u @ u_s <= self.b_u)
        opti.subject_to(self.dynamics_constraints(x_s, x_s, u_s))
        # Terminal constraints
        opti.subject_to(X[:, -1] == x_s)  # x(N) == x_s

        cost += (weight_psi_b_x * psi_b_x[:, -1].T @ psi_b_x[:, -1] +
                 weight_psi_agents * psi_agents[:, -1].T @ psi_agents[:, -1] +
                 weight_psi_obst * psi_obst[:, -1].T @ psi_obst[:, -1] +
                 weight_psi_v_limit * psi_v_limit[:, -1].T @ psi_v_limit[:, -1] +
                 weight_psi_b_x_s * psi_b_x_s.T @ psi_b_x_s)

        cost += (weight_input * ca.norm_2(u_lernt - U[:, 0]) ** 2 +
                 weight_final_state * ca.norm_2(self.previous_opt_sol['X'][:, -1] - X[:, -1]) ** 2)

        opti.minimize(cost)

        # Solve the optimization problem
        if t == 0:  # or agents[f'{i}'].target_status
            self.trajecotry_estimation()
            opti.set_initial(X, self.traj_estimation)
            opti.set_initial(U, np.zeros((self.m, self.N)))
            opti.set_initial(x_s, self.traj_estimation[:, -1])
            opti.set_initial(u_s, np.array([[0 - self.traj_estimation[3, -1]], [0]]))
            opti.set_initial(psi_b_x, np.zeros((np.size(self.b_x), self.N_SF + 1)))
            opti.set_initial(psi_v_limit, np.zeros((1, self.N_SF + 1)))
            opti.set_initial(psi_agents, np.zeros((len(agents), self.N_SF + 1)))
            opti.set_initial(psi_obst, np.zeros((len(circular_obstacles), self.N_SF + 1)))

        else:
            opti.set_initial(X, self.previous_opt_sol_SF['X'])
            opti.set_initial(U, self.previous_opt_sol_SF['U'])
            try:
                opti.set_initial(x_s, self.previous_opt_sol_SF['x_s'])
                opti.set_initial(u_s, self.previous_opt_sol_SF['u_s'])
                opti.set_initial(psi_b_x, self.previous_opt_sol_SF['psi_b_x'])
                opti.set_initial(psi_v_limit, self.previous_opt_sol_SF['psi_v_limit'])
                opti.set_initial(psi_agents, self.previous_opt_sol_SF['psi_agents'])
                opti.set_initial(psi_obst, self.previous_opt_sol_SF['psi_obst'])
                opti.set_initial(psi_b_x_s, self.previous_opt_sol_SF['psi_b_x_s'])
            except Exception as e:
                self.trajecotry_estimation()
                opti.set_initial(x_s, self.traj_estimation[:, -1])
                opti.set_initial(u_s, np.array([[0 - self.traj_estimation[3, -1]], [0]]))
                opti.set_initial(psi_b_x, np.zeros((np.size(self.b_x), self.N_SF + 1)))
                opti.set_initial(psi_v_limit, np.zeros((1, self.N_SF + 1)))
                opti.set_initial(psi_agents, np.zeros((len(agents), self.N_SF + 1)))
                opti.set_initial(psi_obst, np.zeros((len(circular_obstacles), self.N_SF + 1)))

        try:
            opti.solver('ipopt')
            sol = opti.solve()
        except Exception as e:
            print(f"Optimization failed: {e}")

        if opti.stats()['success']:
            self.previous_opt_sol_SF['X'] = sol.value(X)
            self.previous_opt_sol_SF['U'] = sol.value(U)
            self.previous_opt_sol_SF['x_s'] = sol.value(x_s)
            self.previous_opt_sol_SF['u_s'] = sol.value(u_s)
            self.previous_opt_sol_SF['psi_b_x'] = sol.value(psi_b_x)
            self.previous_opt_sol_SF['psi_v_limit'] = sol.value(psi_v_limit)
            self.previous_opt_sol_SF['psi_agents'] = sol.value(psi_agents)
            self.previous_opt_sol_SF['psi_obst'] = sol.value(psi_obst)
            self.previous_opt_sol_SF['psi_b_x_s'] = sol.value(psi_b_x_s)
            self.previous_opt_sol_SF['Cost'] = np.linalg.norm(u_lernt - sol.value(U)[:, 0]) ** 2

            input = sol.value(U)[:, 0].reshape((self.m, 1))
            self.success_solver_SF = True

        else:
            input = np.zeros((self.m, 1))
            input[0, :] = (0 - self.velocity) / self.delta_t
            if self.acc_limits[0] > input[0, :]:
                input[0, :] = self.acc_limits[0]
            elif self.acc_limits[1] < input[0, :]:
                input[0, :] = self.acc_limits[1]
            input[1, :] = 0

            print('soft SF solver failed!')
            self.success_solver_SF = False

        return input

    def psi_optimization(self, agents, circular_obstacles, t, info):

        A_hull = self.hull.equations[:, :-1]
        b_hull = self.hull.equations[:, -1]

        A = np.zeros((np.shape(A_hull)[0] + 4, self.n))
        b = np.zeros((np.shape(A_hull)[0] + 4, 1))
        A[0:np.shape(A_hull)[0], 0:2] = A_hull
        A[-4:, :] = self.A_x[-4:, :]
        b[0:np.shape(A_hull)[0], 0] = b_hull
        b[-4:, :] = -self.b_x[-4:, :]
        b[-2, :] = -info['street speed limit']

        opti = ca.Opti()

        X = opti.variable(self.n, self.N_SF + 1)
        U = opti.variable(self.m, self.N_SF)
        """#psi_b_x = opti.variable(np.size(self.b_x), self.N_SF)
        #weight_psi_b_x = 1
        #psi_v_limit = opti.variable(1, self.N_SF)
        weight_psi_v_limit = 1
        #psi_agents = opti.variable(len(agents), self.N_SF)
        #weight_psi_agents = 1
        #psi_obst = opti.variable(len(circular_obstacles), self.N_SF)
        #weight_psi_obst = 1"""
        psi_f = opti.variable(1, 1)
        psi_b_x = opti.variable(np.size(b), self.N_SF+1)
        alpha_f = 10**4
        delta_i = 0.001

        cost = 0
        for k in range(self.N_SF):
            cost += (psi_b_x[:, k].T @ psi_b_x[:, k])

            for j in range(np.size(b)):
                opti.subject_to(0 <= psi_b_x[j,k])

            # State and Input constraints
            opti.subject_to(A @ X[:, k] + b <= - k * delta_i * np.ones(np.shape(b)) + psi_b_x[:, k])
            opti.subject_to(self.A_u @ U[:, k] <= self.b_u)

            # System dynamics constraints
            opti.subject_to(self.dynamics_constraints(X[:, k + 1], X[:, k], U[:, k]))

            """
            cost += (weight_psi_b_x * psi_b_x[:, k].T @ psi_b_x[:, k] +
                     weight_psi_agents * psi_agents[:, k].T @ psi_agents[:, k] +
                     weight_psi_obst * psi_obst[:, k].T @ psi_obst[:, k] +
                     weight_psi_v_limit * psi_v_limit[:, k].T @ psi_v_limit[:, k])
            
            # State and Input constraints
            opti.subject_to(self.A_x @ X[:, k] <= self.b_x - k * delta_i * np.ones(np.shape(self.b_x)) + psi_b_x[:, k])
            opti.subject_to(self.A_u @ U[:, k] <= self.b_u)

            opti.subject_to(X[3, k] - info['street speed limit'] <= -k * delta_i + psi_v_limit[:, k])

            # System dynamics constraints
            opti.subject_to(self.dynamics_constraints(X[:, k + 1], X[:, k], U[:, k]))

            # Agents avoidance
            if len(agents) >= 1:
                for nr_agent, id_agent in enumerate(agents):
                    agents[id_agent].trajecotry_estimation()
                    if agents[id_agent].security_dist != 0:
                        opti.subject_to(-1 * (self.agents_constraints_traj(X[:, k], agents[id_agent].traj_estimation[:, k],
                                                         agents[id_agent])) <= - k * delta_i + psi_agents[nr_agent, k])

            # Obstacle avoidance
            if len(circular_obstacles) != 0:
                for nr_obs, id_obst in enumerate(circular_obstacles):
                    diff_x = ((X[0, k] - circular_obstacles[id_obst]['center'][0]) / circular_obstacles[id_obst]['r_x']) ** 2
                    diff_y = ((X[1, k] - circular_obstacles[id_obst]['center'][1]) / circular_obstacles[id_obst]['r_y']) ** 2
                    opti.subject_to(1 - diff_x - diff_y <= - k * delta_i + psi_obst[nr_obs, k])"""

        # Initial state
        opti.subject_to(X[:, 0] == self.state)

        # Terminal Constraints
        cost += (psi_b_x[:, -1].T @ psi_b_x[:, -1])
        for j in range(np.size(b)):
            opti.subject_to(0 <= psi_b_x[j, -1])
        opti.subject_to(A @ X[:, -1] + b <= - self.N_SF * delta_i * np.ones(np.shape(b)) + psi_b_x[:, -1])

        # Control Barrier Function
        px_S = self.safe_set['center'][0]
        py_S = self.safe_set['center'][1]
        R_S = self.safe_set['radius']
        opti.subject_to((X[0, -1] - px_S)**2 + (X[1, -1] - py_S)**2 - R_S**2 + 1600 * X[3, -1]**2 <= psi_f)
        opti.subject_to(0 <= psi_f)
        cost += (alpha_f * psi_f ** 2)
        opti.minimize(cost)

        # Solve the optimization problem
        if t == 0:  # or agents[f'{i}'].target_status
            self.trajecotry_estimation()
            opti.set_initial(X, self.traj_estimation)
            opti.set_initial(U, np.zeros((self.m, self.N)))
            opti.set_initial(psi_b_x, np.zeros((np.size(b), self.N_SF+1)))
            #opti.set_initial(psi_b_x, np.zeros((np.size(self.b_x), self.N_SF)))
            #opti.set_initial(psi_v_limit, np.zeros((1, self.N_SF)))
            #opti.set_initial(psi_agents, np.zeros((len(agents), self.N_SF)))
            #opti.set_initial(psi_obst, np.zeros((len(circular_obstacles), self.N_SF)))
            #opti.set_initial(psi_b_hull, np.zeros((np.size(b_hull), self.N_SF)))
            opti.set_initial(psi_f, 0)

        else:
            opti.set_initial(X, self.previous_opt_sol_psi_opt['X'])
            opti.set_initial(U, self.previous_opt_sol_psi_opt['U'])
            try:
                opti.set_initial(psi_b_x, self.previous_opt_sol_psi_opt['psi_b_x'])
                #opti.set_initial(psi_v_limit, self.previous_opt_sol_psi_opt['psi_v_limit'])
                #opti.set_initial(psi_agents, self.previous_opt_sol_psi_opt['psi_agents'])
                #opti.set_initial(psi_obst, self.previous_opt_sol_psi_opt['psi_obst'])
                #opti.set_initial(psi_b_x, self.previous_opt_sol_psi_opt['psi_b_hull'])
                opti.set_initial(psi_f, self.previous_opt_sol_psi_opt['psi_f'])
            except Exception as e:
                opti.set_initial(psi_b_x, np.zeros((np.size(b), self.N_SF+1)))
                #opti.set_initial(psi_b_x, np.zeros((np.size(self.b_x), self.N_SF)))
                #opti.set_initial(psi_v_limit, np.zeros((1, self.N_SF)))
                #opti.set_initial(psi_agents, np.zeros((len(agents), self.N_SF)))
                #opti.set_initial(psi_obst, np.zeros((len(circular_obstacles), self.N_SF)))
                #opti.set_initial(psi_b_hull, np.zeros((np.size(b_hull), self.N_SF)))
                opti.set_initial(psi_f, 0)

        try:
            opti.solver('ipopt')
            sol = opti.solve()
        except Exception as e:
            print(f"Optimization failed: {e}")
            print('A', A)
            print('b', b)
            print('State', self.state)
            print('Area of the hull', self.hull.area)
            error()

        if opti.stats()['success']:
            self.previous_opt_sol_psi_opt['X'] = sol.value(X)
            self.previous_opt_sol_psi_opt['U'] = sol.value(U)
            self.previous_opt_sol_psi_opt['psi_b_x'] = sol.value(psi_b_x)
            #self.previous_opt_sol_psi_opt['psi_v_limit'] = sol.value(psi_v_limit)
            #self.previous_opt_sol_psi_opt['psi_agents'] = sol.value(psi_agents)
            #self.previous_opt_sol_psi_opt['psi_obst'] = sol.value(psi_obst)
            #self.previous_opt_sol_psi_opt['psi_b_hull'] = sol.value(psi_b_hull)
            self.previous_opt_sol_psi_opt['psi_f'] = sol.value(psi_f)

            self.safe_set['traj'] = sol.value(X[0:2,:])

            psi = {
                "psi_b_x": sol.value(psi_b_x),
                #"psi_v_limit": sol.value(psi_v_limit),
                #"psi_agents": sol.value(psi_agents),
                #"psi_obst": sol.value(psi_obst),
                #"psi_b_hull": sol.value(psi_b_hull),
                "psi_f": sol.value(psi_f)
            }
            self.success_solver_SF = True

        else:
            psi = {
                "psi_b_x": np.zeros((np.size(b), self.N_SF)),
                #"psi_b_x": np.zeros((np.size(self.b_x), self.N_SF)),
                #"psi_v_limit": np.zeros((1, self.N_SF)),
                #"psi_agents": np.zeros((len(agents), self.N_SF)),
                #"psi_obst": np.zeros((len(circular_obstacles), self.N_SF)),
                #"psi_b_hull": np.zeros((np.size(b_hull), self.N_SF)),
                "psi_f": 0
            }

            print('soft SF solver failed!')
            self.success_solver_SF = False

        return psi

    def soft_SF_psi_opt(self, u_lernt, agents, circular_obstacles, t, info, psi):

        A_hull = self.hull.equations[:, :-1]
        b_hull = self.hull.equations[:, -1]

        A = np.zeros((np.shape(A_hull)[0] + 4, self.n))
        b = np.zeros((np.shape(A_hull)[0] + 4, 1))
        A[0:np.shape(A_hull)[0], 0:2] = A_hull
        A[-4:, :] = self.A_x[-4:, :]
        b[0:np.shape(A_hull)[0], 0] = b_hull
        b[-4:, :] = -self.b_x[-4:, :]
        b[-2, :] = -info['street speed limit']

        opti = ca.Opti()

        X = opti.variable(self.n, self.N_SF + 1)
        U = opti.variable(self.m, self.N_SF)
        delta_i = 0.001

        opti.minimize(ca.norm_2(u_lernt - U[:, 0]) ** 2)

        for k in range(self.N_SF):

            # State and Input constraints
            opti.subject_to(A @ X[:, k] + b <= - k * delta_i * np.ones(np.shape(b)) + psi['psi_b_x'][:, k].reshape(np.size(b), 1))
            opti.subject_to(self.A_u @ U[:, k] <= self.b_u)

            # System dynamics constraints
            opti.subject_to(self.dynamics_constraints(X[:, k + 1], X[:, k], U[:, k]))

            """
            cost += (weight_psi_b_x * psi_b_x[:, k].T @ psi_b_x[:, k] +
                     weight_psi_agents * psi_agents[:, k].T @ psi_agents[:, k] +
                     weight_psi_obst * psi_obst[:, k].T @ psi_obst[:, k] +
                     weight_psi_v_limit * psi_v_limit[:, k].T @ psi_v_limit[:, k])

            # State and Input constraints
            opti.subject_to(self.A_x @ X[:, k] <= self.b_x - k * delta_i * np.ones(np.shape(self.b_x)) + psi_b_x[:, k])
            opti.subject_to(self.A_u @ U[:, k] <= self.b_u)

            opti.subject_to(X[3, k] - info['street speed limit'] <= -k * delta_i + psi_v_limit[:, k])

            # System dynamics constraints
            opti.subject_to(self.dynamics_constraints(X[:, k + 1], X[:, k], U[:, k]))

            # Agents avoidance
            if len(agents) >= 1:
                for nr_agent, id_agent in enumerate(agents):
                    agents[id_agent].trajecotry_estimation()
                    if agents[id_agent].security_dist != 0:
                        opti.subject_to(-1 * (self.agents_constraints_traj(X[:, k], agents[id_agent].traj_estimation[:, k],
                                                         agents[id_agent])) <= - k * delta_i + psi_agents[nr_agent, k])

            # Obstacle avoidance
            if len(circular_obstacles) != 0:
                for nr_obs, id_obst in enumerate(circular_obstacles):
                    diff_x = ((X[0, k] - circular_obstacles[id_obst]['center'][0]) / circular_obstacles[id_obst]['r_x']) ** 2
                    diff_y = ((X[1, k] - circular_obstacles[id_obst]['center'][1]) / circular_obstacles[id_obst]['r_y']) ** 2
                    opti.subject_to(1 - diff_x - diff_y <= - k * delta_i + psi_obst[nr_obs, k])"""

        # Initial state
        opti.subject_to(X[:, 0] == self.state)

        # Terminal Constraints
        opti.subject_to(A @ X[:, -1] + b <= - self.N_SF * delta_i * np.ones(np.shape(b)) + psi['psi_b_x'][:, -1].reshape(np.size(b), 1))

        # Control Barrier Function
        px_S = self.safe_set['center'][0]
        py_S = self.safe_set['center'][1]
        R_S = self.safe_set['radius']
        opti.subject_to((X[0, -1] - px_S) ** 2 + (X[1, -1] - py_S) ** 2 - R_S ** 2 + 1600 * X[3, -1] ** 2 <= psi['psi_f'])

        # Solve the optimization problem
        if t == 0:  # or agents[f'{i}'].target_status
            self.trajecotry_estimation()
            opti.set_initial(X, self.traj_estimation)
            opti.set_initial(U, np.zeros((self.m, self.N)))
        else:
            #opti.set_initial(X, self.previous_opt_sol_SF['X'])
            #opti.set_initial(U, self.previous_opt_sol_SF['U'])
            opti.set_initial(X, self.previous_opt_sol_psi_opt['X'])
            opti.set_initial(U, self.previous_opt_sol_psi_opt['U'])

        try:
            opti.solver('ipopt')
            sol = opti.solve()
        except Exception as e:
            print('psi_b_x', psi['psi_b_x'])
            print('A', A)
            print('b', b)
            for k in range(self.N_SF):
                print('norm psi_b_x step k=', k, ': ', np.linalg.norm(psi['psi_b_x'][:,k]))
            print('psi_f', psi['psi_f'])
            print('u_lernt', u_lernt)
            print('Input of psi opt', self.previous_opt_sol_psi_opt['U'])
            print('Velocity', self.velocity)
            print('Area of the hull', self.hull.area)
            print('Inside the hull', all(A_hull @ self.position + b_hull.reshape(np.shape(A_hull @ self.position)) <= 0))
            self.plot_feasible_set(self.safe_set, self.hull, self.previous_opt_sol_psi_opt['X'], psi['psi_b_x'][0:-4, :], delta_i)
            erorr()

        if opti.stats()['success']:
            self.previous_opt_sol_SF['X'] = sol.value(X)
            self.previous_opt_sol_SF['U'] = sol.value(U)
            self.previous_opt_sol_SF['Cost'] = np.linalg.norm(u_lernt - sol.value(U)[:, 0]) ** 2

            input = sol.value(U)[:, 0].reshape((self.m, 1))
            self.success_solver_SF = True
        else:
            input = np.zeros((self.m, 1))
            input[0, :] = (0 - self.velocity) / self.delta_t
            if self.acc_limits[0] > input[0, :]:
                input[0, :] = self.acc_limits[0]
            elif self.acc_limits[1] < input[0, :]:
                input[0, :] = self.acc_limits[1]
            input[1, :] = 0

            print('soft SF solver failed!')
            self.success_solver_SF = False
            error()

        return input

    def plot_feasible_set(self, safe_set, feasible_set, traj_psi, psi, delta_i):

        for t in range(7):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for subfig in range(3):
                k = 3 * t + subfig
                ax = axes[subfig]

                # Create the ellipse
                ellipse = patches.Ellipse((safe_set['center'][0], safe_set['center'][1]), safe_set['radius'], safe_set['radius'],
                                          angle=0, edgecolor='black', linestyle='--', linewidth=1,facecolor='none')
                ax.add_patch(ellipse)

                hull = patches.Polygon(feasible_set.points,closed=True, fill=False, color='blue', alpha=0.5)
                ax.add_patch(hull)

                ax.scatter([self.x], [self.y], color='green')

                ax.plot(traj_psi[0,:], traj_psi[1,:], color='red', linestyle='--')
                ax.scatter(traj_psi[0, k], traj_psi[1, k], color='red')

                A = feasible_set.equations[:, :-1]
                b = feasible_set.equations[:, -1]

                # Create a grid of points
                x = np.linspace(-25, 25, 200)  # Adjust range as needed
                y = np.linspace(-25, 25, 200)
                X, Y = np.meshgrid(x, y)
                points = np.c_[X.ravel(), Y.ravel()]  # Flatten the grid into points
                for i in range(np.shape(points)[0]):
                    if all(A @ points[i,:] + b <= - k * delta_i * np.ones(np.shape(b)) + psi[:, k].reshape(np.shape(b))):
                        ax.scatter(points[i,0], points[i,1], s=0.5, color='blue')

                ax.set_aspect('equal')
                ax.set_title('Time step k = ' + str(k))

            plt.show()

    def soft_OD_const(self, opti, obj, llm, X, U, agents, epsilon):

        x = X[0]
        y = X[1]
        theta = X[2]
        v = X[3]

        delta = U[0]
        a = U[1]

        ineq_constraints = llm.OD["inequality_constraints"]
        eq_constraints = llm.OD["equality_constraints"]

        nr_ineq_const = len(ineq_constraints)
        nr_eq_const = len(eq_constraints)

        count = 0
        for i in range(nr_ineq_const):
            opti.subject_to(eval(ineq_constraints[i]) + epsilon[count] <= 0)
            obj += epsilon[count] ** 2
            count += 1
        for i in range(nr_eq_const):
            opti.subject_to(eval(eq_constraints[i]) + epsilon[count] == 0)
            obj += epsilon[count] ** 2
            count += 1

        return obj, opti

    def llm_conditioned_mpc(self, action, agents, circular_obstacles, t, info):

        opti = ca.Opti()

        X = opti.variable(self.n, self.N_SF + 1)
        U = opti.variable(self.m, self.N_SF)
        #x_s = opti.variable(self.n, 1)
        #u_s = opti.variable(self.m, 1)

        cost = 0
        for k in range(self.N):
            if 'go to the entry' in action:
                cost += ca.norm_2(X[:, k] - self.entry['state']) ** 2
            elif 'go to the exit' in action:
                cost += ca.norm_2(X[:, k] - self.exit['state']) ** 2
            elif 'go to the final_target' in action:
                cost += ca.norm_2(X[:, k] - self.final_target['state']) ** 2

            # State and Input constraints
            opti.subject_to(self.A_x @ X[:, k + 1] <= self.b_x)
            opti.subject_to(self.A_u @ U[:, k] <= self.b_u)

            opti.subject_to(X[3, k + 1] <= info['street speed limit'])

            # System dynamics constraints
            opti.subject_to(self.dynamics_constraints(X[:, k + 1], X[:, k], U[:, k]))

        # Initial state
        opti.subject_to(X[:, 0] == self.state)

        opti.subject_to(X[3, -1] <= info['street speed limit'])

        # Agents avoidance
        if len(agents) >= 1:
            for id_agent in agents:
                agents[id_agent].trajecotry_estimation()
                for k in range(self.N_SF + 1):
                    if agents[id_agent].security_dist != 0:
                        #opti.subject_to(self.agents_constraints(X[:, k], agents[id_agent]) >= 0)
                        opti.subject_to(self.agents_constraints_traj(X[:, k], agents[id_agent].traj_estimation[:, k], agents[id_agent]) >= 0)

        # Obstacle avoidance
        if len(circular_obstacles) != 0:
            for id_obst in circular_obstacles:
                for k in range(self.N_SF + 1):
                    diff_x = (X[0, k] - circular_obstacles[id_obst]['center'][0]) ** 2 / (circular_obstacles[id_obst]['r_x']) ** 2
                    diff_y = (X[1, k] - circular_obstacles[id_obst]['center'][1]) ** 2 / (circular_obstacles[id_obst]['r_y']) ** 2
                    opti.subject_to(diff_x + diff_y >= 1)

        # Constraint for the steady state
        #opti.subject_to(self.A_x @ x_s <= self.b_x)
        #opti.subject_to(self.A_u @ u_s <= self.b_u)
        #opti.subject_to(self.dynamics_constraints(x_s, x_s, u_s))
        # Terminal constraints
        #opti.subject_to(X[:, -1] == x_s)  # x(N) == x_s

        if 'go to the entry' in action:
            cost += ca.norm_2(X[:, -1] - self.entry['state']) ** 2
        elif 'go to the exit' in action:
            cost += ca.norm_2(X[:, -1] - self.exit['state']) ** 2
        elif 'go to the final_target' in action:
            cost += ca.norm_2(X[:, -1] - self.final_target['state']) ** 2

        opti.minimize(cost)

        # Solve the optimization problem
        if t == 0:  # or agents[f'{i}'].target_status
            self.trajecotry_estimation()
            opti.set_initial(X, self.traj_estimation)
            opti.set_initial(U, np.zeros((self.m, self.N)))
            #opti.set_initial(x_s, self.traj_estimation[:, -1])
            #opti.set_initial(u_s, np.array([[0 - self.traj_estimation[3, -1]], [0]]))
        else:
            opti.set_initial(X, self.previous_opt_sol['X'])
            opti.set_initial(U, self.previous_opt_sol['U'])
            #opti.set_initial(x_s, self.previous_opt_sol['x_s'])
            #opti.set_initial(u_s, self.previous_opt_sol['u_s'])

        try:
            opti.solver('ipopt')
            sol = opti.solve()
        except Exception as e:
            print(f"Optimization failed: {e}")

        if opti.stats()['success']:
            self.previous_opt_sol['X'] = sol.value(X)
            self.previous_opt_sol['U'] = sol.value(U)
            #self.previous_opt_sol['x_s'] = sol.value(x_s)
            #self.previous_opt_sol['u_s'] = sol.value(u_s)
            self.previous_opt_sol['Cost'] = sol.value(cost)
            #if 'go to the entry' in action:
            #    self.previous_opt_sol['Cost'] = np.linalg.norm(sol.value(X) - self.entry['state']) ** 2
            #elif 'go to the exit' in action:
            #   self.previous_opt_sol['Cost'] = np.linalg.norm(sol.value(X) - self.exit['state']) ** 2
            #elif 'go to the final_target' in action:
            #    self.previous_opt_sol['Cost'] = np.linalg.norm(sol.value(X) - self.final_target['state']) ** 2

            input = sol.value(U)[:, 0].reshape((self.m, 1))
            self.success_solver_MPC_LLM = True

        else:

            input = np.zeros((self.m, 1))
            input[0, :] = (0 - self.velocity) / self.delta_t
            if self.acc_limits[0] > input[0, :]:
                input[0, :] = self.acc_limits[0]
            elif self.acc_limits[1] < input[0, :]:
                input[0, :] = self.acc_limits[1]
            input[1, :] = 0

            print('SF solver failed. Use a default input u = ', input)
            self.success_solver_MPC_LLM = False

        return input

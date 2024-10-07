import numpy as np
import casadi as ca
import random

class Pedestrian:
    def __init__(self, type, info_pedestrian, delta_t):
        self.type = type
        self.delta_t = delta_t
        self.radius = info_pedestrian['radius']
        self.length = info_pedestrian['length']
        self.width = info_pedestrian['width']
        self.security_dist = info_pedestrian['security distance']
        self.a_security_dist = info_pedestrian['a security dist']
        self.b_security_dist = info_pedestrian['b security dist']
        self.vel_limits = info_pedestrian['velocity limits']
        self.acc_limits = info_pedestrian['acceleration limits']
        self.LLM_car = False
        self.brake_status = False

    def init(self, state_space, v_limits, N):
        self.linear_system(state_space)
        self.N = N
        self.Q = np.eye(self.n)
        self.R = np.eye(self.m)
        self.P = 10
        self.T = 100 * self.P
        landa_1 = random.random()
        landa_2 = random.random()
        landa_3 = random.random()

        self.x = state_space['x limits'][0] + landa_1 * (state_space['x limits'][1] - state_space['x limits'][0])
        self.y = state_space['y limits'][0] + landa_2 * (state_space['y limits'][1] - state_space['y limits'][0])

        shift = 10

        if self.x < -6 and self.y < 6 and self.y > -6:
            if self.y < 0:
                self.y = self.y - shift
            elif self.y >= 0:
                self.y = self.y + shift
        elif self.x > 6 and self.y < 6 and self.y > -6:
            if self.y < 0:
                self.y = self.y - shift
            elif self.y >= 0:
                self.y = self.y + shift
        elif self.y < -6 and self.x < 6 and self.x > -6:
            if self.x < 0:
                self.x = self.x - shift
            elif self.x >= 0:
                self.x = self.x + shift
        elif self.y > 6 and self.x < 6 and self.x > -6:
            if self.x < 0:
                self.x = self.x - shift
            elif self.x >= 0:
                self.x = self.x + shift
        elif self.x >= -6 and self.x <= 6 and self.y >= -6 and self.y <= 6:
            if self.x <= 0 and self.y <= 0:
                self.x = self.x - shift
                self.y = self.y - shift
            elif self.x <= 0 and self.y >= 0:
                self.x = self.x - shift
                self.y = self.y + shift
            elif self.x >= 0 and self.y >= 0:
                self.x = self.x + shift
                self.y = self.y + shift
            elif self.x >= 0 and self.y <= 0:
                self.x = self.x + shift
                self.y = self.y - shift

        v = 0 + landa_3 * v_limits[1]
        if landa_3 <= 0.5 and self.x < 0:
            self.v_x = v
            self.v_y = 0
            self.theta = 0
            self.target = np.array([50, 0, 0, 0]).reshape(self.n, 1)
        elif landa_3 <= 0.5 and self.x > 0:
            self.v_x = -v
            self.v_y = 0
            self.theta = np.pi
            self.target = np.array([-50, 0, 0, 0]).reshape(self.n,1)
        elif landa_3 >= 0.5 and self.y < 0:
            self.v_y = v
            self.v_x = 0
            self.theta = np.pi / 2
            self.target = np.array([0, 50, 0, 0]).reshape(self.n,1)
        elif landa_3 >= 0.5 and self.y > 0:
            self.v_y = -v
            self.v_x = 0
            self.theta = -np.pi / 2
            self.target = np.array([0, -50, 0, 0]).reshape(self.n,1)

        self.state = np.array([self.x, self.y, self.v_x, self.v_y]).reshape(4,1)
        self.target = self.target + self.state
        self.target[2] = 0
        self.target[3] = 0
        self.position = np.array([self.x, self.y]).reshape(2, 1)
        self.velocity = [np.sqrt(self.v_x ** 2 + self.v_y ** 2)]
        self.previous_opt_sol = {}
        self.v_x_init = self.v_x
        self.v_y_init = self.v_y


    def dynamics_propagation(self, input):
        x_new  = self.A @ self.state + self.B @ input
        self.state = x_new
        self.x = self.state[0]
        self.y = self.state[1]
        self.v_x = self.state[2]
        self.v_y = self.state[3]
        self.velocity = np.sqrt(self.v_x ** 2 + self.v_y ** 2)
        self.position = np.array([self.x, self.y]).reshape(2,1)

    def linear_system(self, state_space):

        self.A = np.array([[1, 0, 1, 0],
                      [0, 1, 0, 1],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        self.B = np.array([[0, 0],
                      [0, 0],
                      [1, 0],
                      [0, 1]])
        self.n = 4
        self.m = 2

        self.A_x = np.array([[1, 0, 0, 0],
                        [-1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1],
                        [0, 0, 0, -1]
                        ])
        self.b_x = np.array([[state_space['x limits'][1]],
                        [-state_space['x limits'][0]],
                        [state_space['y limits'][1]],
                        [-state_space['y limits'][0]],
                        [self.vel_limits[1]],
                        [-self.vel_limits[0]],
                        [self.vel_limits[1]],
                        [-self.vel_limits[0]]])
        self.A_u = np.array([[1, 0],
                        [-1, 0],
                        [0, 1],
                        [0, -1]])
        self.b_u = np.array([[self.acc_limits[1]],
                             [-self.acc_limits[0]],
                             [self.acc_limits[1]],
                             [-self.acc_limits[0]]])

    def trajecotry_estimation(self):
        input = np.zeros((2,1))
        self.traj_estimation = np.zeros((self.n, self.N + 1))
        self.traj_estimation[:, 0] = self.state[:, 0]
        for k in range(self.N):
            if k <= self.N:
                self.traj_estimation[:, k+1] = self.A @ self.traj_estimation[:, k] #+ self.B @ input
            else:
                self.traj_estimation[0, k + 1] = self.traj_estimation[0, k]
                self.traj_estimation[1, k + 1] = self.traj_estimation[1, k]
                self.traj_estimation[2, k + 1] = self.traj_estimation[2, k]
                self.traj_estimation[3, k + 1] = self.traj_estimation[3, k]
        self.previous_opt_sol['X'] = self.traj_estimation

    def trackingMPC(self, other_agents, ego, circular_obstacles, t):

        nr_agents = len(other_agents)
        opti = ca.Opti()

        X = opti.variable(self.n, self.N + 1)
        U = opti.variable(self.m, self.N)
        x_s = opti.variable(self.n, 1)
        u_s = opti.variable(self.m, 1)

        cost = 0
        for k in range(self.N):
            cost += (X[2,k] - self.v_x_init) ** 2 + (X[3,k] - self.v_y_init) ** 2
            # State and Input constraints
            opti.subject_to(self.A_x @ X[:, k + 1] <= self.b_x)
            opti.subject_to(self.A_u @ U[:, k] <= self.b_u)

            # System dynamics constraints
            opti.subject_to(self.dynamics_constraints(X[:, k + 1], X[:, k], U[:, k]))

        cost += (X[2,-1] - self.v_x_init) ** 2 + (X[3,-1] - self.v_y_init) ** 2

        # Initial state
        opti.subject_to(X[:, 0] == self.state)

        # Constraint for the steady state
        opti.subject_to(self.A_x @ x_s <= self.b_x)
        opti.subject_to(self.A_u @ u_s <= self.b_u)
        opti.subject_to(self.dynamics_constraints(x_s, x_s, u_s))
        # Terminal constraints
        opti.subject_to(X[:, -1] == x_s)  # x(N) == x_s

        # Avoidance of other agents
        #if nr_agents >= 1:
        #    for k in range(self.N + 1):
        #        for other_agent in other_agents:
        #            opti.subject_to(self.agents_constraints_circle(X[:, k],
        #                                                           other_agents[other_agent].previous_opt_sol['X'][:, k],
        #                                                           other_agents[other_agent]))
                    #opti.subject_to(self.agents_constraints(X[:, k], other_agents[other_agent]) >= 0)

        # Avoidance of ego vehicle
        if ego != []:
            for k in range(self.N + 1):
                opti.subject_to(self.agents_constraints(X[:, k], ego) >= 0)

        ## Obstacle avoidance
        #if len(circular_obstacles) != 0:
        #    for id_obst in circular_obstacles:
        #        for k in range(self.N + 1):
        #            diff_x = (X[0, k] - circular_obstacles[id_obst]['center'][0]) ** 2 / (circular_obstacles[id_obst]['r_x']) ** 2
        #            diff_y = (X[1, k] - circular_obstacles[id_obst]['center'][1]) ** 2 / (circular_obstacles[id_obst]['r_y']) ** 2
        #            opti.subject_to(diff_x + diff_y >= 1)

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

            self.previous_opt_sol['X'] = self.traj_estimation
            self.previous_opt_sol['U'] = np.zeros((self.m, self.N))
            self.previous_opt_sol['x_s'] = self.traj_estimation[:, -1]
            self.previous_opt_sol['u_s'] = np.array([[0-self.traj_estimation[3, -1]], [0]])

        return input

    def agents_constraints_circle(self, X, X_agent, agent):

        diff = X[0:2] - X_agent[0:2]
        # Which of the distance have to keep? mine or of the other one? Or one standard for all
        if self.security_dist >= agent.security_dist:
            constraint = ca.transpose(diff) @ diff >= self.security_dist ** 2
        else:
            constraint = ca.transpose(diff) @ diff >= agent.security_dist ** 2

        return constraint

    def dynamics_constraints(self, x_next, x_now, u_now):

        constraints = [x_next == self.A @ x_now + self.B @ u_now]

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

    def brakes(self):
        self.brake_status = True
        input_brakes = np.zeros((self.m, 1))
        input_brakes[0, :] = (0 - self.v_x)
        input_brakes[1, :] = (0 - self.v_y)
        # Check if acceleration is in the limis for v_x
        if input_brakes[0, :] <= self.acc_limits[0]:
            input_brakes[0, :] = self.acc_limits[0]
        elif input_brakes[0, :] >= self.acc_limits[1]:
            input_brakes[0, :] = self.acc_limits[1]
        # Check if acceleration is in the limis for v_y
        if input_brakes[1, :] <= self.acc_limits[0]:
            input_brakes[1, :] = self.acc_limits[0]
        elif input_brakes[1, :] >= self.acc_limits[1]:
            input_brakes[1, :] = self.acc_limits[1]

        """x_new = self.A @ self.state + self.B @ self.input_brakes
        self.state = x_new
        self.x = self.state[0]
        self.y = self.state[1]
        self.v_x = self.state[2]
        self.v_y = self.state[3]
        self.velocity = np.sqrt(self.v_x ** 2 + self.v_y ** 2)
        self.position = np.array([self.x, self.y]).reshape(2, 1)"""

        return input_brakes

    def move(self):
        self.brake_status = False
        input_acc = np.zeros((self.m, 1))
        if abs(self.v_x_init) == 0:

            if abs(self.v_y_init - self.v_y) >= 0.01:
                input_acc[1,:] = (self.v_y_init - self.v_y)

            if input_acc[1,:] <= self.acc_limits[0]:
                input_acc[1, :] = self.acc_limits[0]
            elif input_acc[1,:] >= self.acc_limits[1]:
                input_acc[1, :] = self.acc_limits[1]
        elif abs(self.v_y_init) == 0:

            if abs(self.v_x_init - self.v_x) >= 0.01:
                input_acc[0, :] = (self.v_x_init - self.v_x)

            if input_acc[0,:] <= self.acc_limits[0]:
                input_acc[0, :] = self.acc_limits[0]
            elif input_acc[0,:] >= self.acc_limits[1]:
                input_acc[0, :] = self.acc_limits[1]
        else:
            error()

        """x_new = self.A @ self.state + self.B @ self.input_brakes
        self.state = x_new
        self.x = self.state[0]
        self.y = self.state[1]
        self.v_x = self.state[2]
        self.v_y = self.state[3]
        self.velocity = np.sqrt(self.v_x ** 2 + self.v_y ** 2)
        self.position = np.array([self.x, self.y]).reshape(2, 1)"""

        return input_acc

    def inside_street(self):

        inside_street = False

        if self.x < -6 and self.y < 6 and self.y > -6:
            inside_street = True
        elif self.x > 6 and self.y < 6 and self.y > -6:
            inside_street = True
        elif self.y < -6 and self.x < 6 and self.x > -6:
            inside_street = True
        elif self.y > 6 and self.x < 6 and self.x > -6:
            inside_street = True
        elif self.x >= -6 and self.x <= 6 and self.y >= -6 and self.y <= 6:
            inside_street = True

        return inside_street

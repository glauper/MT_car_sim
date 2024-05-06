import numpy as np
import casadi as ca
import random
from vehicle import Vehicle

class EnvController:
    def __init__(self, horizion, previous_opt_sol, circular_obstacles):
        self.type = "tracking MPC"
        self.N = horizion
        self.previous_opt_sol = previous_opt_sol
        self.circular_obstacles = circular_obstacles

    def tracking_MPC(self, agents, t):
        nr_agents = len(agents)
        opti = ca.Opti()
        X = {}
        U = {}
        x_s = {}
        u_s = {}
        Q = {}
        R = {}
        P = {}
        T = {}

        for i in range(nr_agents):
            X[f'agent {i}'] = opti.variable(agents[f'{i}'].n, self.N + 1)
            U[f'agent {i}'] = opti.variable(agents[f'{i}'].m, self.N)
            x_s[f'agent {i}'] = opti.variable(agents[f'{i}'].n, 1)
            u_s[f'agent {i}'] = opti.variable(agents[f'{i}'].m, 1)

            Q[f'agent {i}'] = np.eye(agents[f'{i}'].n)
            R[f'agent {i}'] = np.eye(agents[f'{i}'].m)
            P[f'agent {i}'] = 10
            T[f'agent {i}'] = 10 * P[f'agent {i}']

        cost = 0
        for k in range(self.N):
            for i in range(nr_agents):
                diff_X = X[f'agent {i}'][:,k] - x_s[f'agent {i}']
                diff_U = U[f'agent {i}'][:,k] - u_s[f'agent {i}']
                #diff_X = X[f'agent {i}'][:, k] - agents[f'{i}'].target
                #diff_U = U[f'agent {i}'][:, k]
                cost += ca.transpose(diff_X) @ Q[f'agent {i}'] @ diff_X + ca.transpose(diff_U) @ R[f'agent {i}'] @ diff_U
                # State and Input constraints
                opti.subject_to(agents[f'{i}'].A_x @ X[f'agent {i}'][:, k + 1] <= agents[f'{i}'].b_x)
                opti.subject_to(agents[f'{i}'].A_u @ U[f'agent {i}'][:, k] <= agents[f'{i}'].b_u)

                # System dynamics constraints
                opti.subject_to(agents[f'{i}'].dynamics_constraints(X[f'agent {i}'][:, k + 1], X[f'agent {i}'][:, k], U[f'agent {i}'][:, k]))

                # Aviodance of other agents
                if nr_agents >= 1:
                    for other_agent in range(nr_agents):
                        if other_agent != i:
                            diff = X[f'agent {i}'][0:2, k] - X[f'agent {other_agent}'][0:2, k]
                            #Which of the ditance have to keep? mine or of the other one? Or one standrad for all
                            opti.subject_to(ca.transpose(diff) @ diff >= agents[f'{i}'].security_dist ** 2)

        for i in range(nr_agents):
            diff_X = X[f'agent {i}'][:,-1] - x_s[f'agent {i}']
            diff_target = agents[f'{i}'].target - x_s[f'agent {i}']
            cost += ca.transpose(diff_X) @ P[f'agent {i}'] @ diff_X + ca.transpose(diff_target) @ T[f'agent {i}'] @ diff_target
            #diff_X = X[f'agent {i}'][:, -1] - agents[f'{i}'].target
            #cost += ca.transpose(diff_X) @ P[f'agent {i}'] @ diff_X

            opti.subject_to(X[f'agent {i}'][:, 0] == agents[f'{i}'].state)

            # Constraint for the steady state
            opti.subject_to(agents[f'{i}'].A_x @ x_s[f'agent {i}'] <= agents[f'{i}'].b_x)
            opti.subject_to(agents[f'{i}'].A_u @ u_s[f'agent {i}'] <= agents[f'{i}'].b_u)
            opti.subject_to(agents[f'{i}'].dynamics_constraints(x_s[f'agent {i}'], x_s[f'agent {i}'], u_s[f'agent {i}']))
            #opti.subject_to(X[f'agent {i}'][:, -2] == x_s[f'agent {i}'])
            #opti.subject_to(x_s[f'agent {i}'][3] == 0)
            # Terminal constraints
            opti.subject_to(X[f'agent {i}'][:, -1] == x_s[f'agent {i}'])  # x(N) == x_s

            #opti.subject_to(X[f'agent {i}'][:, -2] == X[f'agent {i}'][:, -1])
            #opti.subject_to(X[f'agent {i}'][:, -1] == 0)

            # Aviodance of other agents
            if nr_agents >= 1:
                for other_agent in range(nr_agents):
                    if other_agent != i:
                        diff = X[f'agent {i}'][0:2, -1] - X[f'agent {other_agent}'][0:2, -1]
                        # Which of the ditance have to keep? mine or of the other one? Or one standrad for all
                        opti.subject_to(ca.transpose(diff) @ diff >= agents[f'{i}'].security_dist ** 2)

            # Obstacle aviodance
            if len(self.circular_obstacles) != 0:
                for id_obst in self.circular_obstacles:
                    for k in range(self.N + 1):
                        diff_x = (X[f'agent {i}'][0, k] - self.circular_obstacles[id_obst]['center'][0])**2 / (self.circular_obstacles[id_obst]['r_x'])**2
                        diff_y = (X[f'agent {i}'][1, k] - self.circular_obstacles[id_obst]['center'][1])**2 / (self.circular_obstacles[id_obst]['r_y'])**2
                        opti.subject_to(diff_x + diff_y >= 1)

        opti.minimize(cost)
        # Solve the optimization problem
        for i in range(nr_agents):
            if t == 0:#or agents[f'{i}'].target_status
                opti.set_initial(X[f'agent {i}'], np.tile(agents[f'{i}'].state, (1, self.N + 1)).reshape(agents[f'{i}'].n, self.N + 1))
                opti.set_initial(U[f'agent {i}'], np.zeros((agents[f'{i}'].m, self.N)))
                opti.set_initial(x_s[f'agent {i}'], agents[f'{i}'].state)
                opti.set_initial(u_s[f'agent {i}'], np.zeros((agents[f'{i}'].m, 1)))

                #agents[f'{i}'].target_status = False

            else:
                opti.set_initial(X[f'agent {i}'], self.previous_opt_sol[f'X agent {i}'])
                opti.set_initial(U[f'agent {i}'], self.previous_opt_sol[f'U agent {i}'])
                opti.set_initial(x_s[f'agent {i}'], self.previous_opt_sol[f'x_s agent {i}'])
                opti.set_initial(u_s[f'agent {i}'], self.previous_opt_sol[f'u_s agent {i}'])

        opti.solver('ipopt')
        sol = opti.solve()

        stats = opti.stats()
        exit_flag = stats['success']

        if not exit_flag:
            print('Solution is not optimal')
            error()

        for i in range(nr_agents):
            self.previous_opt_sol[f'X agent {i}'] = sol.value(X[f'agent {i}'])
            self.previous_opt_sol[f'U agent {i}'] = sol.value(U[f'agent {i}'])
            self.previous_opt_sol[f'x_s agent {i}'] = sol.value(x_s[f'agent {i}'])
            self.previous_opt_sol[f'u_s agent {i}'] = sol.value(u_s[f'agent {i}'])
        self.previous_opt_sol['Cost'] = sol.value(cost)


        input = {}
        for i in range(nr_agents):
            input[f'agent {i}'] = sol.value(U[f'agent {i}'])[:, 0].reshape((agents[f'{i}'].m, 1))

        return input
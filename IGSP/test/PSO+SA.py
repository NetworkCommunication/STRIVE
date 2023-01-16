import math

import numpy as np
import random
from matplotlib import pyplot as plt


class PSO():

    def __init__(self, pN, dim, max_iter):
        self.ws = 0.9
        self.we = 0.4
        self.c1 = 2
        self.c2 = 2
        self.r2 = 0.3
        self.N = 10
        self.M = 10
        self.P_u = 0.1
        self.P_d = 0.1
        self.D_n = 1.0  # MB Data size of tasks
        self.R = 100
        self.C_n = 0.7  #
        self.lamb_e = 0.5
        self.lamb_t = 0.5
        self.T_max = 0.05
        self.x_max = 1

        # SA
        self.T = 100
        self.T_MIN = 1
        self.K = 100
        self.DELTA = 0.9
        self.DISTURBANCE = (-0.1, 0.1)
        self.k = 0.95
        self.X_new = []
        self.X1_new = np.zeros((self.N, self.M))
        self.X2_new = np.zeros((self.N, self.M))


        self.pN = pN
        self.dim = dim
        self.max_iter = max_iter
        self.X1 = np.zeros((self.pN, 2))
        self.X2 = np.zeros((self.pN, 2))
        self.V1 = np.zeros((self.pN, 2))
        self.V2 = np.zeros((self.pN, 2))

        self.Xmax = 0.5
        self.Xmin = 0.014
        self.V1max = 0.5
        self.V1min = -0.5
        self.V2max = 0.01
        self.V2min = -0.01
        self.p_best = []
        self.g_best = np.zeros((1, self.dim))
        self.p_fit = np.zeros(self.pN)
        self.g_fit = 10000000
        self.swarm = np.zeros((self.pN, 4, self.N, self.M))
        self.V_new = []

    def fitness_function(self, A, x):
        fitness = 0
        for i in range(self.N):
            h1 = 0
            h2 = 0

            for j in range(self.M):
                h2 += A[i, j]
                h1 += A[i, j] * x[i, j]
                c = A[i, j] * (self.lamb_t * ((1 + self.P_u) * self.D_n / self.R + self.C_n / x[i, j]) + self.lamb_e * (
                            x[i, j] ** 2) * self.C_n) \
                    + (1 - A[i, j]) * (self.lamb_e * (x[i, j] ** 2) * self.C_n + self.lamb_t * self.C_n / x[i, j])
                fitness += c
            p1 = 100 * max(0., h1 - self.x_max) ** 2
            # p1 = 10000 * (h1 - self.x_max) ** 2

            p2 = 100 * (h2 - 1) ** 2

            fitness += p1
            fitness += p2

        # print(fitness)
        return fitness

    def init_Pop(self):
        for p in range(self.pN):
            for i in range(self.dim):
                for j in range(self.N):
                    for k in range(self.M):

                        if i == 0:
                            self.swarm[p, i, j, k] = random.uniform(0, 1)
                        elif i == 1:
                            self.swarm[p, i, j, k] = random.uniform(self.Xmin, self.Xmax)
                        else:
                            self.swarm[p, i, j, k] = random.uniform(-1, 1)
            self.p_best.append([self.swarm[p][0], self.swarm[p][1]])
            # print(self.p_best)
            fitness = self.fitness_function(self.p_best[p][0], self.p_best[p][1])  # 计算这个粒子的适应度值
            if self.g_fit > fitness:
                self.g_fit = fitness
                self.g_best = []
                # print(fitness, self.p_best[p])
                self.g_best.append(self.p_best[p])
        return self.swarm



    def anneal(self, X, X1, X2):
        # print(X)
        t = self.T
        cnt = 0
        while t > self.T_MIN:
            # for _ in range(self.K):

            y = self.fitness_function(X1, X2)

            for i in range(self.N):
                for j in range(self.M):
                    # print(X1)
                    # print(X2)

                    self.X1_new[i][j] = X1[i][j] + random.uniform(self.DISTURBANCE[0], self.DISTURBANCE[1]) * t
                    self.X2_new[i][j] = X2[i][j] + random.uniform(self.DISTURBANCE[0], self.DISTURBANCE[1]) * t

                    if self.X1_new[i, j] > 1:
                        self.X1_new[i, j] = 1
                    elif self.X1_new[i, j] < 0:
                        self.X1_new[i, j] = 0

                    if self.X2_new[i, j] > self.Xmax:
                        self.X2_new[i, j] = self.Xmax
                    elif self.X2_new[i, j] < self.Xmin:
                        self.X2_new[i, j] = self.Xmin

            # print(self.X1_new)
            # print(self.X2_new)

            y_new = self.fitness_function(self.X1_new, self.X2_new)
            # print(y_new)

            X_new = np.array([self.X1_new, self.X2_new])
            # print(X_new)

            if y_new < y:
                X = X_new
            else:
                p = math.exp(-(y_new - y) / t)
                # print(p)
                if random.random() > p:
                    X = X_new
            t = self.DELTA * t
            cnt += 1
        # print(X_new)
        return X

    def iterator(self):
        def clip(x, low, high):
            if x <= low:
                return low
            elif x >= high:
                return high
            return x

        x_plt, y_plt = [], []
        for t in range(self.max_iter):
            w = self.ws - (self.ws - self.we) * (t / self.max_iter)
            for p in range(self.pN):
                V1 = self.swarm[p][2]
                V2 = self.swarm[p][3]
                X1 = self.swarm[p][0]
                X2 = self.swarm[p][1]
                for i in range(self.N):
                    for j in range(self.M):
                        V1[i][j] = clip(
                            w * V1[i][j] + self.c1 * random.random() * (self.p_best[p][0][i][j] - X1[i][j]) + \
                            self.c2 * random.random() * (self.g_best[0][0][i][j] - X1[i][j]), self.V1min, self.V1max)
                        V2[i][j] = clip(
                            w * V2[i][j] + self.c1 * random.random() * (self.p_best[p][1][i][j] - X2[i][j]) + \
                            self.c2 * random.random() * (self.g_best[0][1][i][j] - X2[i][j]), self.V2min, self.V2max)
                        X1[i][j] = clip(X1[i][j] + V1[i][j], 0, 1)
                        X2[i][j] = clip(X2[i][j] + V2[i][j], self.Xmin, self.Xmax)
                X = [np.array([X1, X2])]
                if self.fitness_function(X1, X2) < self.fitness_function(self.p_best[p][0], self.p_best[p][1]):
                    self.p_best[p][0] = np.copy(X1)
                    self.p_best[p][1] = np.copy(X2)
                else:
                    self.anneal(X, X1, X2)
                    self.p_best[p][0] = np.copy(X[0][0])
                    self.p_best[p][1] = np.copy(X[0][1])

                if self.fitness_function(self.p_best[p][0], self.p_best[p][1]) < self.fitness_function(self.g_best[0][0], self.g_best[0][1]):
                    self.g_best[0][0] = np.copy(self.p_best[p][0])
                    self.g_best[0][1] = np.copy(self.p_best[p][1])

            x_plt.append(t)
            y_plt.append(self.fitness_function(self.p_best[p][0], self.p_best[p][1]))
            # print(self.fitness_function(self.p_best[p][0], self.p_best[p][1]))
        print(self.g_best)
        plt.plot(x_plt, y_plt)
        plt.show()



if __name__ == '__main__':
    # for _ in range(10):
        pso = PSO(pN=50, dim=4, max_iter=50)
        pso.init_Pop()
        pso.iterator()

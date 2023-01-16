import math

import numpy as np
import random


class PSO():

    def __init__(self, pN, dim, max_iter): # 初始化类  设置粒子数量  位置信息维度  最大迭代次数
        self.ws = 0.9
        self.we = 0.4
        self.c1 = 1.49445
        self.c2 = 1.49445
        self.r1 = 0.6
        self.r2 = 0.3
        self.N = 10
        self.M = 5
        self.P_u = 0.1
        self.P_d = 0.1
        self.D_n = 1.0
        self.R = 100
        self.C_n = 0.7
        self.lamb_e = 0.5
        self.lamb_t = 0.5

        self.pN = pN
        self.dim = dim
        self.max_iter = max_iter
        self.X1 = np.zeros((self.pN, 2))
        self.X2 = np.zeros((self.pN, 2))
        self.V1 = np.zeros((self.pN, 2))
        self.V2 = np.zeros((self.pN, 2))

        self.Xmax = 0.5
        self.Xmin = 0.01
        self.Vmax = 0.01
        self.Vmin = -0.01
        self.p_best = []
        self.g_best = np.zeros((1, self.dim))
        self.p_fit = np.zeros(self.pN)
        self.g_fit = 10000000
        self.swarm = np.zeros((self.pN, 4, self.N, self.M))
        self.V_new = []

    def fitness_function(self, A, x):
        fitness = 0
        for i in range(self.N):
            for j in range(self.M):
                if A[i, j] == 0:

                    c = self.lamb_e * (x[i, j] ** 2) * self.C_n + self.lamb_t * self.C_n / x[i, j]
                else:
                    c = self.lamb_t *( (1 + self.P_u) * self.D_n / self.R + self.C_n / x[i, j]) + self.lamb_e * (x[i, j] ** 2) * self.C_n

                fitness += c
        return fitness


    def init_Pop(self):
        for p in range(self.pN):
            for i in range(self.dim):
                for j in range(self.N):
                    for k in range(self.M):

                        if i == 0:
                            self.swarm[p, i, j, k] = random.uniform (self.Xmin, self.Xmax)
                        elif i == 1:
                            self.swarm[p, i, j, k] = random.uniform(self.Xmin, self.Xmax)
                        else:
                            self.swarm[p, i, j, k] = random.uniform(-1, 1)
            self.p_best.append([self.swarm[p][0], self.swarm[p][1]])
            # print(self.p_best)
            fitness = self.fitness_function(self.p_best[p][0], self.p_best[p][1]) # 计算这个粒子的适应度值
            if self.g_fit > fitness:
                self.g_fit = fitness
                self.g_best = []
                # print(fitness, self.p_best[p])
                self.g_best.append(self.p_best[p])
        return self.swarm


    def iterator(self):
        # fitness = []
        for t in range (self.max_iter):
            w = self.ws - (self.ws - self.we) * (t / self.max_iter)

            for p in range(self.pN):

                V1 = self.swarm[p][2]
                V2 = self.swarm[p][3]

                # V_new = np.array([V1, V2])
                X1 = self.swarm[p][0]
                X2 = self.swarm[p][1]

                # V3 = np.append([V1], [V2])
                # X3 = np.append([X1], [X2])
                # V3_new = V3.flatten()
                # X3_new = X3.flatten()
                # p_best_2dim = np.append([self.p_best[0],self.p_best[1]])
                # p_best_1dim = p_best_2dim.flatten()
                # g_best_1dim = self.g_best.flatten()



                # for i in range (self.N * 2 * self.N * 2):
                #     V3[i] = w * V3[i] + self.c1 * self.r1 * (self.p_best[i] - X1[i]) + self.c2 * self.r2 * (
                #                  g_best_1dim[i] - X1[i])
                #
                #             if V3[i] > self.Vmax:
                #                 V3[i] = self.Vmax
                #             elif V3[i] < self.Vmin:
                #                  V3[i] = self.Vmin
                #
                #             # 更新位置X1
                #             X3[i] = X3[i] + V3[i]
                #             if X3[i] > self.Xmax:
                #                 X3[i] = self.Xmax
                #             elif X3[i] < self.Xmin:
                #                 X3[i] = self.Xmin


                for i in range (self.N ):
                    for j in range (self.M):

                        # 更新速度V1
                        V1[i, j] = w * V1[i, j] + self.c1 * self.r1 * (self.p_best[p][0][i, j] - X1[i, j]) + self.c2 * self.r2 * (
                                self.g_best[0][0][i, j] - X1[i, j])
                        if V1[i, j] > self.Vmax:
                            V1[i, j] = self.Vmax
                        elif V1[i, j] < self.Vmin:
                             V1[i, j] = self.Vmin

                        # 更新位置X1
                        X1[i, j] = X1[i, j] + V1[i, j]
                        if X1[i, j] > 1:
                            X1[i, j] = 1
                        elif X1[i, j] < 0:
                            X1[i, j] = 0

                        # 更新速度V2
                        V2[i, j] = w * V2[i, j] + self.c1 * self.r1 * (
                                    self.p_best[p][1][i, j] - X2[i, j]) + self.c2 * self.r2 * (
                                           self.g_best[0][0][i, j] - X2[i, j])
                        if V2[i, j] > self.Vmax:
                            V2[i, j] = self.Vmax
                        elif V2[i, j] < self.Vmin:
                            V2[i, j] = self.Vmin

                        # 更新位置X2
                        X2[i, j] = X2[i, j] + V2[i, j]
                        if X2[i, j] > self.Xmax:


                            X2[i, j]= self.Xmax
                        elif X2[i, j] < self.Xmin:
                            X2[i, j] = self.Xmin

            for i in range(self.pN):
                X = [np.array([X1, X2])]
                # self.p_best.append([X1, X2])
                fitness_tmp = self.fitness_function(X1, X2)

                if(fitness_tmp < self.p_fit[i]): #更新个体最优
                    # self.p_best[i][0] = self.X1
                    # self.p_best[i][1] = self.X2
                    self.p_best[i] = X
                    self.p_fit[i] = fitness_tmp

                #使用模拟退火算法更新p_fit
                # else:
                #      self.p_fit[i].anneal(X)

                if (fitness_tmp < self.g_fit):  # 更新全局最优
                    self.g_best = X
                    print(self.g_fit)
                    self.g_fit = fitness_tmp

            # fitness.append(self.g_fit)
            print('最小值为：\n', self.g_fit)
            z1 = self.g_fit

            # for i in range(self.N):
            #     for j in range(self.M):
            #         if X1[i][j] < 0.5:
            #             X1[i][j] = 0
            #         else:
            #             X1[i][j] = 1
            print('最佳卸载策略和分配资源：\n', X1, '\n', X2)
            x1 = X1
            y1 = X2
        return z1, x1, y1

if __name__ == '__main__':
    pso = PSO(pN = 10, dim= 4, max_iter= 100)
    pso.init_Pop()
    z1, x1, y1 = pso.iterator()
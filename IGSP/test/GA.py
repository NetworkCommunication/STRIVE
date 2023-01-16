import math

import numpy as np
import random
class GA():

    def __init__(self, pN, dim, max_iter):
        # self.w = 0.8
        self.ws = 0.9
        self.we = 0.4
        self.c1 = 1.49445
        self.c2 = 1.49445
        self.r1 = 0.6
        self.r2 = 0.3
        self.N = 5
        self.M = 5
        self.P_u = 0.1  # W
        self.P_d = 0.1
        self.D_n = 1.0  # MB Data size of tasks
        self.R = 100
        self.C_n = 0.7  # cycles/s\


        # GA
        self.c3 = 0.5
        self.c4 = 0.5
        self.pc = 0.1
        self.pm = 0.1




        self.pN = pN
        self.dim = dim
        self.max_iter = max_iter
        # self.X1 = np.zeros((self.pN, 2, self.N, self.M))
        # self.X2 = np.zeros((self.pN, 2, self.N, self.M))
        self.X1 = np.zeros((self.pN, 2))
        self.X2 = np.zeros((self.pN, 2))
        # self.X1 = []
        # self.X2 = []
        self.Xmax = 0.5
        self.Xmin = 0.01
        self.V1 = np.zeros((self.pN, 2))
        self.V2 = np.zeros((self.pN, 2))
        # self.V1 = []
        # self.V2 = []
        self.Vmax = 0.01
        self.Vmin = -0.01
        self.p_best = []
        self.g_best = np.zeros((1, self.dim))
        self.p_fit = np.zeros(self.pN)
        self.g_fit = 10000000
        self.swarm = np.zeros((self.pN, 4, self.N, self.M))
        self.V_new = []

        self.X1_new = np.zeros((5, 5))
        self.X2_new = np.zeros((5, 5))

        self.X_superior = np.zeros((self.pN, 2, self.N, self.M))
        # self.X_superior1 = np.zeros((self.pN, 1, self.N, self.M))
        # self.X_superior2 = np.zeros((self.pN, 1, self.N, self.M))
        self.pop_p = np.zeros((self.pN, 2, self.N, self.M))
        # self.pop_g = np.zeros((self.pN, 2, self.N, self.M))

    def fitness_function(self, A, x):
        fitness = 0
        for i in range(self.N):
            for j in range(self.M):
                if A[i, j] == 0:

                    c = 0.5 * ((1 + self.P_u) * self.D_n / self.R + self.C_n / x[i, j] + (x[i, j] ** 2) * self.C_n)
                else:
                    c = 0.5 * ((x[i, j] ** 2) * self.C_n + self.C_n / x[i, j])
                fitness += c
        return fitness

    def cal_fitness_function_pop(self, pop):

        pop_value = np.zeros((self.pN, 1), dtype=float)

        for p in range(0, self.pN):

            pop_value[p, 0] = self.fitness_function(pop[p][0], pop[p][1])

        # print(pop_value)
        return pop_value

    def cal_GA_fitness_function_pop(self, pop_value):
        value_pop = []
        # GA_fitness = []
        for i in range(self.pN):

            value = 1 / (pop_value[i][0] + 0.001)
            value_pop.append(value)
            # print(value_pop)
        max_value = max(value_pop)
        min_value = min(value_pop)

        GA_fitness = (value-min_value)/(max_value-min_value+0.001)
        # print(GA_fitness)

        return GA_fitness

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
            fitness = self.fitness_function(self.p_best[p][0], self.p_best[p][1]) # 计算这个粒子的适应度值
            if self.g_fit > fitness:
                self.g_fit = fitness
                self.g_best = []
                # print(fitness, self.p_best[p])
                self.g_best.append(self.p_best[p])
            # print('P_bestp', self.p_best[p])
            # print('G_best', self.g_best[0])
        return self.swarm

    def init_X_superior(self):

        cnt = 0
        for p in range(self.pN):
            # print(self.X_superior[p])
            for k in range(2):
                # print(self.X_superior[p][k])
                for i in range(self.N):
                    for j in range(self.M):

                        self.X_superior[p][k][i, j] = ((self.c3 * self.r1 * self.p_best[p][k][i, j]) + (
                                    self.c4 * self.r2 * self.g_best[0][0][i, j])) / \
                                                       (self.c3 * self.r1 + self.c4 * self.r2)


                        if k == 0:
                            if self.X_superior[p][k][i, j] > 1:
                                self.X_superior[p][k][i, j] = 1
                            elif self.X_superior[p][k][i, j] < 0:
                                self.X_superior[p][k][i, j] = 0
                            cnt+=1
                        else:
                            if self.X_superior[p][k][i, j] > 0.5:
                                self.X_superior[p][k][i, j] = 0.5
                            elif self.X_superior[p][k][i, j] < 0.01:
                                self.X_superior[p][k][i, j] = 0.01
        # print(cnt, self.X_superior[p])

                        # self.X_superior2[p][k][i, j] = ((self.c3 * self.r1 * self.p_best[p][k][i, j]) + (
                        #             self.c4 * self.r2 * self.g_best[p][0][i, j])) / \
                        #                                (self.c3 * self.r1 + self.c4 * self.r2)
                        # print("22222222222222222", self.X_superior2[p][k][i, j])
                        # if self.X_superior2[p][k][i, j] > 0.5:
                        #     self.X_superior2[p][k][i, j] = 0.5
                        # elif self.X_superior2[p][k][i, j] < 0.01:
                        #     self.X_superior2[p][k][i, j] = 0.01

        # self.X_superior = np.array([self.X_superior1, self.X_superior2])

        # print(self.X_superior)

        return self.X_superior



    def crossover(self, pop):
        for p in range (self.pN):
            for k in range (1):
                for i in range(self.N):
                        for j in range(self.M):

                            parent_pop1 = self.p_best[p][k][i, j]
                            parent_pop2 = self.g_best[0][k][i, j]
                            # print(parent_pop2)
                            n = np.random.rand()
                            if np.random.rand() < self.pc:
                                parent_pop1 = parent_pop2
                                self.p_best[p][k][i, j] = parent_pop1

        pop = self.p_best

        # print(self.p_best)

        return pop

    def mutation (self, pop):
        for p in range(self.pN):
            for k in range(1):
                for i in range(self.N):
                    for j in range(self.M):
                        if np.random.rand() < self.pc:

                            if k == 0:
                                self.p_best[p][k][i, j] = random.uniform(0, 1)
                            else:
                                self.p_best[p][k][i, j] = random.uniform(0.01, 0.5)

        pop = self.p_best
        return pop


    def select(self, pop1, pop2):
        new_pop = np.zeros((self.pN, 4, self.N, self.M))

        x1_value = self.cal_fitness_function_pop(pop1)
        x2_value = self.cal_fitness_function_pop(pop2)
        x1 = self.cal_GA_fitness_function_pop(x1_value)
        x2 = self.cal_GA_fitness_function_pop(x2_value)

        if (x1<x2):
            new_pop = pop1
        else:
            new_pop = pop2

        return new_pop



        # for p in range(self.pN):
        #     fitness_pop= self.fitness_function(self.p_best[p][0], self.p_best[p][1])
        #     fitness_X_superior =self.fitness_function(self.X_superior[p][0], self.X_superior[p][1])
        #     if (fitness_X_superior > fitness_pop):
        #         new_pop = np.append(pop1[p])
        #     else:
        #         new_pop = np.append(pop2[p])



    # # 二元锦标赛选择
    # def select (pop, self):
    #     new_pop = []
    #     for p in range(self.pN):
    #         x_idx1, x_idx2 = random.randint(0, M - 1), random.randint(0, M - 1)
    #         x = pop[x_idx1] if F(Pop[x_idx1]) < F(Pop[x_idx2]) else Pop[x_idx2]
    #         new_Pop.append(x)



if __name__ == '__main__':
    ga = GA(pN = 10, dim= 4, max_iter= 10)
    pop1 = ga.init_Pop()
    pop2 = ga.init_X_superior()
    print(pop2)
    # print(pop)
    # pop_value = ga.cal_fitness_function_pop(pop)
    #
    # GA_fitness = ga.cal_GA_fitness_function_pop(pop_value)
    # print(GA_fitness)
    fitvalue = []
    pop_sum = []
    for i in range(0, 10):
        pop_c = ga.crossover(pop1)
        # new_pop = np.concatenate((pop_c, pop))
        # print(pop_c)
        pop_m = ga.mutation(pop1)
        # print(pop_m)
        new_pop = ga.select(pop1, pop2)
        new_pop_value = ga.cal_fitness_function_pop(new_pop)
        # print(new_pop)





import math
import random

import numpy as np
from task import Task

Dv = 50
Fv = 4000
MAX_TASK = 10
TASK_SOLT = 20
alpha = 0.25

class Vehicle(object):

    def __init__(self, id, loc_x, loc_y, direction, velocity):
        self.loc_x = loc_x
        self.loc_y = loc_y
        self.loc = [loc_x, loc_y]
        self.velocity = velocity
        self.direction = direction
        self.id = id
        self.alpha = alpha
        self.range = Dv
        self.resources = round((1 - np.random.randint(1, 5) / 10) * Fv, 2)  # MHz

        self.trans_task = 0
        self.mec_lest = None
        self.neighbor = []
        self.cur_task = None
        self.total_task = []
        self.len_task = len(self.total_task)
        self.accept_task = []
        self.overflow = 0
        self.hold_on = 0
        self.lastCreatWorkTime = 0
        self.cur_frame = 0
        self.len_action = 0

        self.create_work()

    def create_work(self):

        if (self.cur_frame - self.lastCreatWorkTime) % TASK_SOLT == 0:

            if random.random() < 0.6:
                if self.len_task < MAX_TASK:
                    task = Task(self, self.cur_frame)
                    self.lastCreatWorkTime = self.cur_frame
                    self.total_task.append(task)
                    self.len_task += 1
                    print("第{}辆车产生了任务".format(self.id))
                    self.overflow = 0
                else:
                    print("第{}辆车任务队列已满".format(self.id))
                    self.overflow += 1


    @property
    def get_location(self):
        return self.loc

    def set_location(self, loc_x, loc_y):
        self.loc_x = loc_x
        self.loc_y = loc_y
        self.loc = [self.loc_x, self.loc_y]

    @property
    def get_x(self):
        return self.loc_x

    @property
    def get_y(self):
        return self.loc_y






    def get_state(self):
        self.otherState = []
        self.excludeNeighbor_state = []
        self.taskState = []

        self.otherState.extend(self.loc)
        self.otherState.append(self.velocity)
        self.otherState.append(self.direction)
        self.excludeNeighbor_state.extend(self.loc)
        self.excludeNeighbor_state.append(self.velocity)
        self.excludeNeighbor_state.append(self.direction)

        self.otherState.append(self.resources)
        self.excludeNeighbor_state.append(self.resources)

        self.excludeNeighbor_state.append(self.trans_task)
        self.otherState.append(self.trans_task)

        # if self.trans_task is not None:
        #     self.otherState.append(self.trans_task.need_trans_size)
        #     self.excludeNeighbor_state.append(self.trans_task.need_trans_size)
        # else:
        #     self.otherState.append(0)
        #     self.excludeNeighbor_state.append(0)
        self.otherState.append(self.len_task)  # 当前队列长度
        self.excludeNeighbor_state.append(self.len_task)
        for neighbor in self.neighbor:
            self.otherState.extend(neighbor.loc)
            self.otherState.append(neighbor.velocity)
            self.otherState.append(neighbor.direction)
            self.otherState.append(neighbor.resources)

        for i in range(MAX_TASK):
            if i < self.len_task:
                task = self.total_task[i]
                self.taskState.append([task.create_time, task.need_trans_size, task.need_precess_cycle, task.max_time])
            else:
                self.taskState.append([0, 0, 0, 0])

        return self.excludeNeighbor_state











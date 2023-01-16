import numpy as np
from task import Task
Dv = 100
Fv = 4000
MAX_TASK = 10
TASK_SOLT = 20
alpha = 0.25
class Vehicle(object):
    # 位置：x，y
    def __init__(self, id, loc_x, loc_y, direction, velocity):
        # 车的位置信息
        self.loc_x = loc_x
        self.loc_y = loc_y
        self.loc = [loc_x, loc_y]
        self.velocity = velocity  # m/s
        self.direction = direction
        self.id = id
        self.alpha = alpha
        self.range = Dv
        self.resources = round((1 - np.random.randint(1, 5) / 10) * Fv, 2)  # MHz
        self.trans_task = 0
        self.total_task = []
        self.len_task = len(self.total_task)
        self.accept_task = []
        self.overflow = 0
        self.hold_on = 0
        self.lastCreatWorkTime = 0
        self.cur_frame = 0
        self.cur_task = None
        # self.create_work()

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

    def create_work(self):
        for i in range (MAX_TASK):
            if (self.cur_frame - self.lastCreatWorkTime) % TASK_SOLT == 0:

                if self.len_task < MAX_TASK:  # 队列不满
                    task = Task(self, self.cur_frame)
                    self.lastCreatWorkTime = self.cur_frame
                    self.total_task.append(task)
                    self.len_task += 1
                    print("任务车辆产生了任务{}".format(i+1))
                    self.overflow = 0
                else:
                    print("任务车辆任务队列已满")
                    self.overflow += 1
            self.total_task.append(task)
        # print(self.total_task)
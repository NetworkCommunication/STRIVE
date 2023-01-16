import numpy as np

class Task:

    def __init__(self, vehicle, createTime):

        self.vehicle = vehicle
        self.aim = None

        self.size = np.random.uniform(0.2, 1)
        self.cycle = np.random.randint(20, 50)
        self.need_trans_size = self.size * np.power(2, 10)
        self.need_precess_cycle = self.cycle * self.size * 1000
        self.rate = 0
        self.compute_resource = 0
        self.hold_on_time = 0
        self.create_time = createTime
        self.pick_time = 0

        self.energy = 0
        self.trans_time = 0
        self.precess_time = 0
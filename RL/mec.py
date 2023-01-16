# -*- coding: utf-8 -*-
import random

from experiment.vehicle import Vehicle

RANGE_MEC = 200
RESOURCE = 20000  #



class MEC(object):
    def __init__(self, id, loc_x, loc_y, resources=RESOURCE):
        self.loc_x = loc_x
        self.loc_y = loc_y
        self.loc = [self.loc_x, self.loc_y]
        self.id = id
        self.resources = resources
        self.state = []
        self.range = RANGE_MEC
        self.accept_task = []
        self.sum_needDeal_task = 0
        self.len_action = 0
        self.cur_frame = 0
        self.get_state()

    @property
    def get_x(self):
        return self.loc_x

    @property
    def get_y(self):
        return self.loc_y

    @property
    def get_location(self):
        return self.loc


    def get_state(self):

        self.state = []
        self.state.extend(self.loc)
        self.state.append(self.resources)
        return self.state



if __name__ == '__main__':
    mec = MEC(10, 10, 1)
    # vehicles = []
    # for i in range(40):
    #     vehicle = Vehicle(i, random.randint(1, 5), random.randint(1, 5), random.randint(0, 4))
    #     vehicle.creat_work()
    #     vehicles.append(vehicle)
    # for i, vehicle in enumerate(vehicles):
    #     print("v{}.get_state():{}".format(i, vehicle.get_state()))
    # print("mec.get_state():", mec.get_state(), mec.cur_frame)
    # mec.get_task([2] * 40, vehicles)
    # print("mec.received_task:", mec.received_task)
    # print("resources:", mec.resources)
    # mec.renew_resources(1)
    # print("after received_task:", mec.received_task)
    # print("after resources:", mec.resources)
    # print("renew_state", mec.renew_state(1, [1, 2, 2], vehicles), mec.cur_frame)
    print(mec.get_location)

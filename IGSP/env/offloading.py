import random
import sys
from random import randint
import numpy as np
from vehicle import Vehicle
from task import Task

MAX_TASK = 10
TASK_SOLT = 20
y = [0,1]
directions = [1]
HOLD_TIME = [5, 10, 20, 30]
N = 10
sigma = -114
POWER = 23
BrandWidth_Mec = 100
gama = 1.25 * (10 ** -11)

class Offloading(object):
    def __init__(self, num_Vehicles=N):
        # 车辆天线高度
        self.vehHeight = 1.5
        self.stdV2V = 3
        self.freq = 2
        self.vehAntGain = 3
        self.vehNoiseFigure = 9
        self.vehicles = []
        self.Service_vehicles = []
        self.num_Vehicles = num_Vehicles
        self.need_trans_task = []
        self.cur_frame = 0
        self.aims = []
        self.M = 10
        self.A = np.zeros((num_Vehicles, MAX_TASK))
        self.x = np.zeros((self.num_Vehicles, MAX_TASK))


    def add_new_vehicles(self, id, loc_x, loc_y, direction, velocity):
        vehicle = Vehicle(id=id, loc_x=loc_x, loc_y=loc_y, direction=direction, velocity=velocity)
        self.vehicles.append(vehicle)


    def reset(self):
        self.vehicles = []
        self.Service_vehicles = []
        self.cur_frame = 0
        # self.otherState = []
        # self.offloadingActions = [0] * self.num_Vehicles
        # self.taskActions = [0] * self.num_Vehicles

        lastpath1 = r''
        file1 = open(lastpath1, 'r')
        line = file1.readline()
        data_list1 = []
        while line:
            num = list(map(float, line.split()))
            data_list1.append(num)
            line = file1.readline()
        file1.close()
        data_array1 = np.array(data_list1)

        lastpath2 = r''
        file2 = open(lastpath2, 'r')

        line = file2.readline()
        data_list2 = []
        while line:
            num = list(map(float, line.split()))
            data_list2.append(num)
            line = file2.readline()
        file2.close()
        data_array2 = np.array(data_list2)

        lastpath3 = r''
        file3 = open(lastpath3, 'r')

        line = file3.readline()
        data_list3 = []
        while line:
            num = list(map(float, line.split()))
            data_list3.append(num)
            line = file3.readline()
        file3.close()
        data_array3 = np.array(data_list3)

        for i in range(data_array2.size):
            self.add_new_vehicles(id=i+1, loc_x=data_array2[0][i], loc_y=data_array3[0][i], direction=1,
                                  velocity=data_array1[0][i])


    def distribute_task(self):

        def get_aim(A):
            for i in range(self.num_Vehicles):
                for j in range(MAX_TASK):
                    if A[i][j] == 1:
                        return self.vehicles[i]
                    else:
                        return self.Service_vehicles[0]
        # print(len(self.vehicles))

        for i, vehicle in enumerate(self.vehicles):
            task = vehicle.total_task[i]
            aim = get_aim(self.A)
            task.aim = aim
            task.rate = self.compute_rate(vehicle, aim)
            if aim == self.Service_vehicles[0]:
                task.pick_time = self.cur_frame
                vehicle.accept_task.append(task)
                vehicle.total_task.remove(task)
                vehicle.cur_task = task
                vehicle.len_task -= 1
                continue
            else:
                task.pick_time = self.cur_frame
                task.aim = aim
                task.rate = self.compute_rate(task.vehicle, task.aim)
                vehicle.len_task -= 1
                vehicle.cur_task = task
                vehicle.trans_task = 1
                vehicle.total_task.remove(task)
                self.need_trans_task.append(task)


    def distribute_resource(self, x):

        for i, vehicle in enumerate(self.vehicles):
            for j in range(MAX_TASK):
                task = vehicle.cur_task
                if task is not None:
                    if x[i][j] == 0:
                        task.vehicle.cur_task = None
                        task.vehicle.total_task.append(task)
                        if task.aim == task.vehicle:
                            vehicle.accept_task.remove(task)
                        else:
                            self.need_trans_task.remove(task)
                    task.compute_resource = x[i][j]



    @staticmethod
    def compute_distance(taskVehicle: Vehicle, aim):
        return round(np.sqrt(np.abs(taskVehicle.get_x - aim.get_x) ** 2 + np.abs(taskVehicle.get_y - aim.get_y) ** 2),
                     2)
    def generate_fading_V2V(self, dist_DuePair):
        pathLoss = 32.4 + 20 * np.log10(dist_DuePair) + 20 * np.log10(self.freq)
        combinedPL = -(np.random.randn() * self.stdV2V + pathLoss)
        return combinedPL + self.vehAntGain * 2 - self.vehNoiseFigure

    def compute_rate(self, vehicle: Vehicle, aim):
        print("aim:{} ".format(aim.id))
        if aim == vehicle:
            return 0

        else:
            distance = self.compute_distance(vehicle, aim)
            fade = self.generate_fading_V2V(distance)
        power = np.power(10, (POWER + fade) / 10)
        sigma_w = np.power(10, sigma / 10)
        sign = power / sigma_w
        SNR = round((BrandWidth_Mec / self.num_Vehicles) * np.log2(1 + sign), 2)
        print("第{}辆车计算速率:{} kb/ms".format(vehicle.id, SNR))
        return SNR  # kb/ms

    def compute_persist(self, vehicle: Vehicle, aim):
        distance = self.compute_distance(vehicle, aim)
        if distance > aim.range:
            return 0
        if vehicle.velocity == aim.velocity and vehicle.direction == aim.direction:
            return sys.maxsize * 1000
        else:
            return (np.sqrt(vehicle.range ** 2 - (aim.get_y - vehicle.get_y) ** 2) + aim.get_x - vehicle.get_x) / \
                   np.abs(vehicle.velocity * vehicle.direction - aim.velocity * aim.direction) * 1000

    @staticmethod
    def compute_energy(trans_time):
        return np.power(10, POWER / 10) / 1000 * trans_time

    def get_sum_time(self, task):
        vehicle = task.vehicle
        aim = task.aim
        # print('xxxxxxx')
        if vehicle == aim:
            trans_time = 0
        else:
            cur_rate = task.rate
            trans_time = task.need_trans_size / cur_rate
        cur_compute = task.compute_resource
        compute_time = task.need_precess_cycle / cur_compute
        communication_time = self.compute_persist(vehicle, aim)
        sum_time = trans_time + compute_time + task.pick_time - task.create_time
        if sum_time > communication_time:
            print("该卸载方案不可行")
        else:
            if task.aim != vehicle:
                energy = self.compute_energy(trans_time)
                print("传输需要{}ms".format(trans_time))
                print("传输消耗{} J".format(energy))

            else:
                energy = round(gama * np.power(cur_compute, 3) * compute_time, 2)
                print("本地计算消耗{} J".format(energy))

    def compute_sum_time(self):
        for i, vehicle in enumerate(self.vehicles):

            if vehicle.cur_task is not None:
                self.get_sum_time(vehicle.cur_task)
                vehicle.cur_task = None











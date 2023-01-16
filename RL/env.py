import random
import sys
from random import randint
import numpy as np
from mec import MEC
from vehicle import Vehicle

MAX_TASK = 5
y = [0,1]
directions = [1, 1]
MEC_loc = [ [0, 0], [200,0], [200,1], [400, 0], [800, 1]]
HOLD_TIME = [5, 10, 20, 30]

N = 6
K = 5
MAX_NEIGHBOR = 5
# CAPACITY = 20000

sigma = -114
POWER = 23
BrandWidth_Mec = 100

gama = 1.25 * (10 ** -11)
# a = 0.6
# b = 0.4
# T1 = -0.5
# T2 = -0.2
# T3 = 0.05


# MEC_Price = 0.6
# VEC_Price = 0.4
# LOC_Price = 0.3



class Env(object):
    def __init__(self, num_Vehicles=N, num_MECs=K):
        self.bsHeight = 25
        self.vehHeight = 1.5
        self.stdV2I = 8
        self.stdV2V = 3
        self.freq = 2
        self.vehAntGain = 3
        self.bsAntGain = 8
        self.bsNoiseFigure = 5
        self.vehNoiseFigure = 9
        self.vehicles = []
        self.Vehicles_service = []
        self.MECs = []
        self.num_Vehicles = num_Vehicles
        self.num_MECs = num_MECs
        self.need_trans_task = []
        self.cur_frame = 0
        self.offloadingActions = [0] * num_Vehicles
        self.taskActions = [0] * num_Vehicles
        self.holdActions = [0] * num_Vehicles
        self.computeRatioActions = [0] * num_Vehicles
        self.aims = []
        self.otherState = []
        self.taskState = []
        self.vehicles_state = []
        self.reward = [0] * self.num_Vehicles


    def add_new_vehicles(self, id, loc_x, loc_y, direction, velocity):
        vehicle = Vehicle(id=id, loc_x=loc_x, loc_y=loc_y, direction=direction, velocity=velocity)
        self.vehicles.append(vehicle)



    def reset(self):
        self.vehicles = []
        self.MECs = []

        self.otherState = []
        self.cur_frame = 0
        self.offloadingActions = [0] * self.num_Vehicles
        self.taskActions = [0] * self.num_Vehicles

        for i in range(0, self.num_MECs):  # 初始化mec
            cur_mec = MEC(id=i, loc_x=MEC_loc[i][0], loc_y=MEC_loc[i][1])
            self.MECs.append(cur_mec)

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
            self.add_new_vehicles(id=i, loc_x=data_array2[0][i], loc_y=data_array3[0][i], direction=1,
                                  velocity=data_array1[0][i])


        self.renew_neighbor()
        self.renew_neighbor_MEC()

        for vehicle in self.vehicles:
            self.otherState.extend(vehicle.get_state())
            self.vehicles_state.append(vehicle.otherState)
            self.taskState.append(vehicle.taskState)


        for m in self.MECs:
            self.otherState.extend(m.state)

    def renew_neighbor(self):
        for i in range(len(self.vehicles)):
            self.vehicles[i].neighbor = []
        z = np.array([[complex(vehicle.get_x, vehicle.get_y) for vehicle in self.vehicles]])
        Distance = abs(z.T - z)

        for i in range(len(self.vehicles)):
            sort_idx = np.argsort(Distance[:, i])
            for j in range(MAX_NEIGHBOR):
                self.vehicles[i].neighbor.append(self.vehicles[sort_idx[j + 1]])

    def renew_neighbor_MEC(self):
        for vehicle in self.vehicles:
            distance = []
            for mec in self.MECs:
                distance.append(self.compute_distance(vehicle, mec))
            vehicle.mec_lest = self.MECs[distance.index(min(distance))]

    @staticmethod
    def get_aim(vehicle: Vehicle, action):
        if action == 0:
            return vehicle
        elif action == 1:
            return vehicle.mec_lest
        else:
            return vehicle.neighbor[action - 2]


    def process_taskActions(self):
        # numOffloading = 1 + 1 + MAX_NEIGHBOR
        for i, vehicle in enumerate(self.vehicles):
            if vehicle.len_task <= 0 or vehicle.hold_on > 0:
                vehicle.cur_task = None
                continue
            action = self.taskActions[i]
            offloadingAction = self.offloadingActions[i]

            if action >= vehicle.len_task or action < 0:
                # self.reward[i] += Ki - Kq * vehicle.len_task - vehicle.overflow
                vehicle.cur_task = None
                continue
            elif action >= MAX_TASK:
                task = vehicle.total_task[0]
            else:
                task = vehicle.total_task[action]

            aim = self.get_aim(vehicle, offloadingAction)
            task.aim = aim
            task.rate = self.compute_rate(vehicle, aim)
            if vehicle == aim:
                task.pick_time = self.cur_frame
                vehicle.accept_task.append(task)
                vehicle.total_task.remove(task)
                vehicle.cur_task = task
                vehicle.len_task -= 1
                continue

            if vehicle.trans_task == 1:
                vehicle.cur_task = None

            else:
                task.pick_time = self.cur_frame
                task.aim = aim
                task.rate = self.compute_rate(task.vehicle, task.aim)

                aim.len_action += 1
                vehicle.len_task -= 1
                vehicle.cur_task = task
                vehicle.trans_task = 1
                vehicle.total_task.remove(task)
                self.need_trans_task.append(task)


    def process_holdActions(self):
        for i, vehicle in enumerate(self.vehicles):
            if vehicle.hold_on > 0:
                vehicle.hold_on -= 1
                for task in vehicle.total_task:
                    task.hold_on_time += 1

    @staticmethod
    def compute_distance(taskVehicle: Vehicle, aim):
        return round(np.sqrt(np.abs(taskVehicle.get_x - aim.get_x) ** 2 + np.abs(taskVehicle.get_y - aim.get_y) ** 2),
                     2)
    def generate_fading_V2I(self, dist_veh2bs):
        dist2 = (self.vehHeight - self.bsHeight) ** 2 + dist_veh2bs ** 2
        pathLoss = 128.1 + 37.6 * np.log10(np.sqrt(dist2) / 1000)  # 路损公式中距离使用km计算
        combinedPL = -(np.random.randn() * self.stdV2I + pathLoss)
        return combinedPL + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure

    def generate_fading_V2V(self, dist_DuePair):
        pathLoss = 32.4 + 20 * np.log10(dist_DuePair) + 20 * np.log10(self.freq)
        combinedPL = -(np.random.randn() * self.stdV2V + pathLoss)
        return combinedPL + self.vehAntGain * 2 - self.vehNoiseFigure

    def compute_rate(self, vehicle: Vehicle, aim):
        print("vehicle:{} aim:{} ".format(vehicle.id, aim.id))
        if aim == vehicle:
            return 0

        distance = self.compute_distance(vehicle, aim)
        if type(aim) == MEC:
            fade = self.generate_fading_V2I(distance)
        else:
            fade = self.generate_fading_V2V(distance)
        power = np.power(10, (POWER + fade) / 10)
        sigma_w = np.power(10, sigma / 10)
        sign = power / sigma_w
        SNR = round((BrandWidth_Mec / self.num_Vehicles) * np.log2(1 + sign), 2)
        print("第{}辆车计算速率:{} kb/ms".format(vehicle.id, SNR))
        return SNR  # kb/ms
    def compute_persist(self, vehicle: Vehicle, aim):
        distance = self.compute_distance(vehicle, aim)
        print("aim:{} vehicle:{}".format(aim.id, vehicle.id))
        if distance > aim.range:
            return 0

        if type(aim) is Vehicle:
            if vehicle.velocity == aim.velocity and vehicle.direction == aim.direction:
                return sys.maxsize * 1000
                # return np.abs(vehicle.direction * 500 - np.max(np.abs(vehicle.get_x),
                #        np.abs(aim.get_x))) / vehicle.velocity
            else:
                return (np.sqrt(vehicle.range ** 2 - (aim.get_y - vehicle.get_y) ** 2) + aim.get_x - vehicle.get_x) / \
                       np.abs(vehicle.velocity * vehicle.direction - aim.velocity * aim.direction) * 1000

        else:
            return (np.sqrt(aim.range ** 2 - (aim.get_y - vehicle.get_y) ** 2) + aim.get_x - vehicle.get_x) / np.abs(
                vehicle.velocity * vehicle.direction) * 1000
    @staticmethod
    def compute_energy(trans_time):
        return np.power(10, POWER / 10) / 1000 * trans_time


    def distribute_resource(self, ratio: list):
        sum_ratio_matrix = np.zeros((self.num_Vehicles, self.num_Vehicles + self.num_MECs), dtype=float)
        for i, vehicle in enumerate(self.vehicles):
            task = vehicle.cur_task
            if task is not None:
                if ratio[i] == 0:

                    # self.reward[i] += Ki - Kq * vehicle.len_task - vehicle.overflow
                    task.vehicle.cur_task = None
                    # 重新入队列
                    task.vehicle.total_task.append(task)
                    if task.aim == task.vehicle:
                        vehicle.accept_task.remove(task)
                    else:
                        self.need_trans_task.remove(task)
                resources = task.aim.resources
                task.compute_resource = ratio[i] * resources
                task.aim.resources -= task.compute_resource
                j = task.aim.id + self.num_Vehicles if type(task.aim) == MEC else task.aim.id
                sum_ratio_matrix[i][j] += ratio[i]
        sum_ratio = np.sum(sum_ratio_matrix, axis=0)
        for j, cur_ratio in enumerate(sum_ratio):

            aim = self.vehicles[j] if j < self.num_Vehicles else self.MECs[j - self.num_Vehicles]
            if cur_ratio > 1 or aim.resources <= 0:
                print("第{}ms分配资源非法".format(self.cur_frame))
                for i in range(sum_ratio_matrix.shape[0]):
                    if sum_ratio_matrix[i][j] > 0:
                        task = self.vehicles[i].cur_task
                        task.vehicle.cur_task = None
                        # self.reward[i] += Ki - Kq * self.vehicles[i].len_task - self.vehicles[i].overflow
                        if i == j:
                            self.vehicles[i].accept_task.remove(task)
                        else:
                            self.need_trans_task.remove(task)
                        self.vehicles[i].total_task.append(task)
                        aim.resources += task.compute_resource

    def get_sum_time(self, task):
        vehicle = task.vehicle
        aim = task.aim
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

            if sum_time > task.max_time:
                deltaTime = sum_time - task.max_time
                if deltaTime < 20:
                    print("任务{}超时{}ms".format(vehicle.id, sum_time - task.max_time))

        return sum_time

    def compute_sum_time(self):
        for i, vehicle in enumerate(self.vehicles):
            if vehicle.cur_task is not None:


                self.get_sum_time(vehicle.cur_task)
                vehicle.cur_task = None
        self.Reward = np.mean(self.reward)





    def distribute_task(self, cur_frame):

        need_task = []
        time = cur_frame - self.cur_frame
        for task in self.need_trans_task:
            aim = task.aim
            vehicle = task.vehicle
            distance = self.compute_distance(vehicle, aim)
            if distance > aim.range:
                continue
            task.trans_time += time
            rate = self.compute_rate(vehicle, aim)
            if task.need_trans_size <= time * rate:
                self.vehicles[vehicle.id].trans_task = 0
                print("第{}车任务传输完成，真实花费{}ms".format(vehicle.id, task.trans_time))
                aim.accept_task.append(task)
                aim.len_action -= 1
            else:
                task.need_trans_size -= time * rate
                need_task.append(task)
        print("forward", self.need_trans_task)
        self.need_trans_task = need_task
        print("after", self.need_trans_task)
        for vehicle in self.vehicles:
            vehicle.sum_needDeal_task = len(vehicle.accept_task)
        for mec in self.MECs:
            mec.sum_needDeal_task = len(mec.accept_task)

    def renew_resources(self, cur_frame):
        time = cur_frame - self.cur_frame
        for i, vehicle in enumerate(self.vehicles):
            total_task = vehicle.accept_task
            size = len(total_task)

            if size > 0:
                retain_task = []
                for task in total_task:
                    f = task.compute_resource
                    precessed_time = task.need_precess_cycle / f
                    task.precess_time += time
                    task.energy = gama * np.power(f / 1000, 3) * time
                    if precessed_time > time:
                        print(f"-----------------{precessed_time},{time}")
                        task.need_precess_cycle -= f * time
                        retain_task.append(task)
                    else:
                        if task.aim == task.vehicle:
                            print("任务{}卸载给自己".format(task.vehicle.id))
                        print("任务{}已完成，实际传输花费{}ms，实际计算花费{}ms".format(task.vehicle.id,
                                                                                      task.trans_time,
                                                                                      task.precess_time))
                        task.aim.resources += task.compute_resource
                vehicle.accept_task = retain_task

        for i, mec in enumerate(self.MECs):
            total_task = mec.accept_task
            size = len(total_task)
            if size > 0:
                retain_task = []
                for task in total_task:
                    f = task.compute_resource
                    precessed_time = task.need_precess_cycle / f
                    task.precess_time += time
                    if precessed_time > time:
                        task.need_precess_cycle -= f * time
                        retain_task.append(task)
                    else:
                        print("任务{}已完成，传输花费{}ms，计算花费{}ms".format(task.vehicle.id, task.trans_time,
                                                                              task.precess_time))
                        # 收回计算资源
                        task.aim.resources += task.compute_resource
                mec.accept_task = retain_task
        self.distribute_task(cur_frame=cur_frame)


    def renew_locs(self, cur_frame):
        time = cur_frame - self.cur_frame
        for vehicle in self.vehicles:
            loc_x = round(vehicle.get_x + vehicle.direction * vehicle.velocity * time, 2)
            if loc_x > 1000:
                vehicle.set_location(-1000, vehicle.get_y)
            elif loc_x < -1000:
                vehicle.set_location(1000, vehicle.get_y)
            else:
                vehicle.set_location(loc_x, vehicle.get_y)




    def renew_state(self, cur_frame):
        self.otherState = []
        self.taskState = []
        self.vehicles_state = []

        self.renew_locs(cur_frame)
        self.renew_neighbor()
        self.renew_neighbor_MEC()




        for vehicle in self.vehicles:
            vehicle.cur_frame = cur_frame
            vehicle.create_work()
            self.otherState.extend(vehicle.get_state())
            self.taskState.append(vehicle.taskState)
            self.vehicles_state.append(vehicle.otherState)
            vehicle.cur_task = None

        for mec in self.MECs:
            mec.cur_frame = cur_frame
            self.otherState.extend(mec.get_state())

    def step(self, taskActions, offloadingActions, computeRatioActions):
        cur_frame = self.cur_frame + 1  # ms
        self.offloadingActions = offloadingActions
        self.taskActions = taskActions
        self.computeRatioActions = computeRatioActions
        self.process_holdActions()

        self.process_taskActions()

        self.distribute_resource(self.computeRatioActions)


        other_state = self.otherState
        task_state = self.taskState
        vehicle_state = self.vehicles_state
        self.renew_resources(cur_frame)

        self.renew_state(cur_frame)

        self.cur_frame = cur_frame
        print("当前有{}个任务没传输完成".format(len(self.need_trans_task)))


        return other_state, task_state, vehicle_state, self.vehicles_state, self.otherState, self.taskState,
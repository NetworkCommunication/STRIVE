from env import Env
import numpy as np
if __name__ == '__main__':
    print()
    env = Env()
    #
    env.reset()




    for vehicle in env.vehicles:
        print("vehicle{} location:".format(vehicle.id), vehicle.get_location)
    # for mec in env.MECs:
    #     print("mec{} location:".format(mec.id), mec.get_location)
    # for vehicle in env.vehicles:
    #     print(vehicle.mec_lest.get_location, end="  ")



    # task = np.array(env.taskState)
    # print(task.size)
    # print(task.shape)



    vehicles = env.vehicles

    # for vehicle in vehicles:
    #     print(vehicle.get_location)
    # print("-----------------------------------")
    # for vehicle in vehicles:
    #     for i in vehicle.neighbor:
    #         print(i.id, end=" ")
    #     print()


    # list = [vehicles[0],vehicles[1],vehicles[2],vehicles[3],Vehicles_service[0]]
    # print(list)


    # for vehicle in vehicles:
    #     print("第{}车状态：{}".format(vehicle.id+1, vehicle.otherState))
    #     print("该车邻居:")
    #     for i in vehicle.neighbor:
    #         print(i.id+1, end="  ")
    #     print()



    # for i in range(1000):
        # taskActions, offloadingActions, computeRatioActions

    # for i in range(10):
    #     action1 = []
    #     action2 = []
    #     action3 = []
    #     for j in range(20):
    #         action1.append(0)
    #         action2.append(np.random.randint(0, 7))
    #         # action2.append(0)
    #         # action3.append(round(np.random.random(), 2))
    #         action3.append(0.8)
    #     env.step(action1, action2, action3)
    #
    #
    # # def step(self, taskActions, offloadingActions, computeRatioActions):
    # taskActions = []
    # offloadingActions = []
    # computeRatioActions = []
    #
    # env.reset()
    # for i in range(env.num_Vehicles):




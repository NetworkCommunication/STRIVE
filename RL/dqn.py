# -*- coding: utf-8 -*-
import os
import time
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pylab import mpl
import netron

from RL.env import Env
from RL.model import DQN


mpl.rcParams["font.sans-serif"] = ["SimHei"]

Experience = namedtuple('Transition',
                        field_names=['cur_otherState', 'cur_TaskState',
                                     'taskAction', 'aimAction', 'resourceAction',
                                     'reward',
                                     'next_otherState', 'next_TaskState'])
GAMMA = 0.99
BATCH_SIZE = 64
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000

EPSILON_DECAY_LAST_FRAME = 150000
EPSILON_START = 0.6
EPSILON_FINAL = 0.01

RESET = 10000

MAX_TASK = 10

momentum = 0.005

RESOURCE = [0.2, 0.4, 0.6, 0.8]


@torch.no_grad()
def play_step(env, epsilon, models, device="cpu"):
    vehicles = env.vehicles
    old_otherState = []
    old_taskState = []

    actionTask = []
    actionAim = []
    actionResource = []

    for i, model in enumerate(models):
        old_otherState.append(vehicles[i].otherState)
        old_taskState.append(vehicles[i].taskState)
        if np.random.random() < epsilon:

            actionTask.append(np.random.randint(0, 10))
            actionAim.append(np.random.randint(0, 7))  # local+mec+neighbor
            actionResource.append(round(np.random.random(), 1))
        else:
            state_v = torch.tensor([vehicles[i].otherState], dtype=torch.float32)
            taskState_v = torch.tensor([[vehicles[i].taskState]], dtype=torch.float32)
            taskAction, aimAction, resourceAction = model(state_v, taskState_v)

            taskAction = np.array(taskAction, dtype=np.float32).reshape(-1)
            aimAction = np.array(aimAction, dtype=np.float32).reshape(-1)
            resourceAction = np.array(resourceAction, dtype=np.float32).reshape(-1)

            actionAim.append(np.argmax(aimAction))
            actionTask.append(np.argmax(taskAction))
            actionResource.append(RESOURCE[np.argmax(resourceAction)])
    # print("action:", action)
    _, _, _, otherState, _, taskState, Reward, reward = env.step(actionTask, actionAim, actionResource)
    # print("reward:", reward)

    for i, vehicle in enumerate(vehicles):
        exp = Experience(old_otherState[i], [old_taskState[i]],
                         actionTask[i], actionAim[i], actionResource[i],
                         reward[i],
                         otherState[i], [taskState[i]])
        vehicle.buffer.append(exp)
    return round(Reward, 2)



def calc_loss(batch, net: DQN, tgt_net: DQN, device="cpu"):
    cur_otherState, cur_TaskState, taskAction, aimAction, resourceAction, rewards, next_otherState, next_TaskState = batch  #

    otherStates_v = torch.tensor(np.array(cur_otherState, copy=False), dtype=torch.float32).to(device)
    taskStates_v = torch.tensor(np.array(cur_TaskState, copy=False), dtype=torch.float32).to(device)
    # print("states_v:", states_v)  # batch状态
    taskActions_v = torch.tensor(np.array(taskAction), dtype=torch.int64).to(device)
    aimActions_v = torch.tensor(np.array(aimAction), dtype=torch.int64).to(device)
    resourceAction_v = torch.tensor(np.array(resourceAction), dtype=torch.int64).to(device)
    # print("actions_v", actions_v)  # batch动作
    rewards_v = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
    # print("rewards_v", rewards_v)  # batch奖励
    next_otherStates_v = torch.tensor(np.array(next_otherState, copy=False), dtype=torch.float32).to(device)
    next_taskStates_v = torch.tensor(np.array(next_TaskState, copy=False), dtype=torch.float32).to(device)
    # print("next_states_v", next_states_v)  # batch下一个状态


    taskActionValues, aimActionValues, resourceActionValues = net(otherStates_v,
                                                                  taskStates_v)  # .gather(1, aimActions_v.unsqueeze(-1)).squeeze(-1)
    taskActionValues = taskActionValues.gather(1, taskActions_v.unsqueeze(-1)).squeeze(-1)
    aimActionValues = aimActionValues.gather(1, aimActions_v.unsqueeze(-1)).squeeze(-1)
    resourceActionValues = resourceActionValues.gather(1, resourceAction_v.unsqueeze(-1)).squeeze(-1)


    next_taskActionValues, next_aimActionValues, next_resourceActionValues = tgt_net(next_otherStates_v,
                                                                                     next_taskStates_v)  # .max(1)[0]  # 得到最大的q值

    next_taskActionValues = next_taskActionValues.max(1)[0].detach()
    next_aimActionValues = next_aimActionValues.max(1)[0].detach()
    next_resourceActionValues = next_resourceActionValues.max(1)[0].detach()


    # next_states_values = next_aimActionValues.detach()
    # print("next_states_values", next_states_values)
    expected_aim_values = next_aimActionValues * GAMMA + rewards_v
    expected_task_values = next_taskActionValues * GAMMA + rewards_v
    expected_resource_values = next_resourceActionValues * GAMMA + rewards_v
    # print(" expected_state_values", expected_state_values)

    return nn.MSELoss()(taskActionValues, expected_task_values) + \
           nn.MSELoss()(aimActionValues, expected_aim_values) + \
           nn.MSELoss()(resourceActionValues, expected_resource_values)


if __name__ == '__main__':
    env = Env()
    env.reset()

    frame_idx = 0
    # writer = SummaryWriter(comment="-" + env.__doc__)
    agents = env.vehicles
    models = []
    tgt_models = []
    optimizers = []
    for agent in agents:
        # print(agent.get_location, agent.velocity)
        task_shape = np.array([agent.taskState]).shape
        # print(task_shape)
        model = DQN(len(agent.otherState), task_shape, MAX_TASK, len(agent.neighbor) + 2, len(RESOURCE))
        models.append(model)
        optimer = optim.RMSprop(params=model.parameters(), lr=LEARNING_RATE, momentum=momentum)
        optimizers.append(optimer)
    for agent in agents:
        # print(agent.get_location, agent.velocity)
        task_shape = np.array([agent.taskState]).shape
        # print(task_shape)
        model = DQN(len(agent.otherState), task_shape, MAX_TASK, len(agent.neighbor) + 2, len(RESOURCE))
        model.load_state_dict(models[agent.id].state_dict())
        tgt_models.append(model)

    total_reward = []
    recent_reward = []
    loss_1 = []
    reward_1 = []

    epsilon = EPSILON_START
    eliposde = 150000
    while eliposde > 0:

        if frame_idx % RESET == 0:
            print("游戏重置")
            env.reset()

        frame_idx += 1
        print("the {} steps".format(frame_idx))
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
        reward = play_step(env, epsilon, models)
        total_reward.append(reward)
        print("current reward:", reward)
        print("current 100 times total rewards:", np.mean(total_reward[-100:]))
        recent_reward.append(np.mean(total_reward[-100:]))
        if np.mean(total_reward[-100:]) > 0.5:
            break

        for i, agent in enumerate(agents):
            # print("length of {} buffer".format(agent.id), len(agent.buffer))
            if len(agent.buffer) < REPLAY_SIZE:
                continue
            if frame_idx % SYNC_TARGET_FRAMES == 0:
                tgt_models[i].load_state_dict(models[i].state_dict())
            optimizers[i].zero_grad()
            batch = agent.buffer.sample(BATCH_SIZE)
            loss_t = calc_loss(batch, models[i], tgt_models[i])
            # print("loss:", loss_t)
            loss_t.backward()
            optimizers[i].step()
            if agent.id == 0:
                print("cur_loss:", loss_t.item())
                loss_1.append(loss_t.item())
                reward_1.append(env.reward[0])
        eliposde -= 1

    cur_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime(time.time()))

    os.makedirs("D:/pycharm/Project/VML/MyErion/experiment/result/" + cur_time)
    for i, vehicle in enumerate(env.vehicles):
        torch.save(models[i].state_dict(),
                   "D:/pycharm/Project/VML/MyErion/experiment/result/" + cur_time + "/vehicle" + str(i) + ".pkl")

    plt.plot(range(len(recent_reward)), recent_reward)
    plt.title("奖励曲线")
    plt.show()

    plt.plot(range(len(loss_1)), loss_1)
    plt.title("损失曲线")
    plt.show()

    plt.plot(range(1000), reward_1[-1000:])
    plt.title("车辆一奖励曲线")
    plt.show()

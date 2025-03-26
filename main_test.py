# debug field
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

<<<<<<< HEAD
import numpy as np
import torch
import random

import matplotlib.pyplot as plt
from env import UAV_RIS_UE_EAV
from ddpg import Agent

import numpy as np
import math
import time




def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(123)

system = UAV_RIS_UE_EAV(
    UE_num=3,
    RIS_ant_num=8,
    UAV_ant_num=2,
    if_move_UEs=True,
    if_movements=True,
    )



if_robust = True


episode_num = 80
episode_cnt = 0
step_num = 50

agent_1_param_dic = {}
agent_1_param_dic["alpha"] = 0.0001
agent_1_param_dic["beta"] = 0.0001
agent_1_param_dic["input_dims"] = system.get_system_state_dim()
agent_1_param_dic["tau"] = 0.001
agent_1_param_dic["batch_size"] = 32
agent_1_param_dic["n_actions"] = system.get_system_action_dim()
agent_1_param_dic["action_noise_factor"] = 0.1
agent_1_param_dic["memory_max_size"] = int(step_num * episode_num)
=======
from env import MiniSystem
from ddpy import Agent
import ddpy_2
import numpy as np
import math
import time
# 1 init system model
system = MiniSystem(
    user_num=2,
    RIS_ant_num=4,
    UAV_ant_num=4,
    if_dir_link=1,
    if_with_RIS=True,
    if_move_users=True,
    if_movements=True,
    reverse_x_y=(False, False),
    if_UAV_pos_state = True
    )
if_Theta_fixed = False
if_G_fixed = False
if_BS = False
if_robust = True
# 2 init RL Agent

episode_num = 100
episode_cnt = 0
step_num = 100

agent_1_param_dic = {}
agent_1_param_dic["alpha"] = 0.0001
agent_1_param_dic["beta"] = 0.001
agent_1_param_dic["input_dims"] = system.get_system_state_dim()
agent_1_param_dic["tau"] = 0.001
agent_1_param_dic["batch_size"] = 64
agent_1_param_dic["n_actions"] = system.get_system_action_dim() - 2
agent_1_param_dic["action_noise_factor"] = 0.1
agent_1_param_dic["memory_max_size"] = int(5/5 * episode_num * step_num)
>>>>>>> ffe98bc78fcde95c811f10b8c9220cac7d875392
agent_1_param_dic["agent_name"] = "G_and_Phi"
agent_1_param_dic["layer1_size"] = 800
agent_1_param_dic["layer2_size"] = 600
agent_1_param_dic["layer3_size"] = 512
agent_1_param_dic["layer4_size"] = 256

<<<<<<< HEAD

=======
agent_2_param_dic = {}
agent_2_param_dic["alpha"] = 0.0001
agent_2_param_dic["beta"] = 0.001
agent_2_param_dic["input_dims"] = 3
agent_2_param_dic["tau"] = 0.001
agent_2_param_dic["batch_size"] = 64
agent_2_param_dic["n_actions"] = 2
agent_2_param_dic["action_noise_factor"] = 0.5
agent_2_param_dic["memory_max_size"] = int(5/5 * episode_num * step_num)
agent_2_param_dic["agent_name"] = "UAV"
agent_2_param_dic["layer1_size"] = 400
agent_2_param_dic["layer2_size"] = 300
agent_2_param_dic["layer3_size"] = 256
agent_2_param_dic["layer4_size"] = 128
>>>>>>> ffe98bc78fcde95c811f10b8c9220cac7d875392

agent_1 = Agent(
    alpha       = agent_1_param_dic["alpha"],
    beta        = agent_1_param_dic["beta"],
    input_dims  = [agent_1_param_dic["input_dims"]],
    tau         = agent_1_param_dic["tau"],
    env         = system,
    batch_size  = agent_1_param_dic["batch_size"],
    layer1_size=agent_1_param_dic["layer1_size"],
    layer2_size=agent_1_param_dic["layer2_size"], 
    layer3_size=agent_1_param_dic["layer3_size"],
    layer4_size=agent_1_param_dic["layer4_size"],
    n_actions   = agent_1_param_dic["n_actions"],
    max_size = agent_1_param_dic["memory_max_size"],
    agent_name= agent_1_param_dic["agent_name"]
    ) 

<<<<<<< HEAD



=======
agent_2 = Agent(
    alpha       = agent_2_param_dic["alpha"],
    beta        = agent_2_param_dic["beta"],
    input_dims  = [agent_2_param_dic["input_dims"]],
    tau         = agent_2_param_dic["tau"],
    env         = system,
    batch_size  = agent_2_param_dic["batch_size"],
    layer1_size=agent_2_param_dic["layer1_size"],
    layer2_size=agent_2_param_dic["layer2_size"], 
    layer3_size=agent_2_param_dic["layer3_size"],
    layer4_size=agent_2_param_dic["layer4_size"],
    n_actions   = agent_2_param_dic["n_actions"],
    max_size = agent_2_param_dic["memory_max_size"],
    agent_name= agent_2_param_dic["agent_name"]
    ) 
"""
agent_1.load_models(
    load_file_actor = 'C:\\demo\\other_branch\\Learning-based_Secure_Transmission_for_RIS_Aided_mmWave-UAV_Communications_with_Imperfect_CSI\\data\\mannal_store\\models\\Actor_G_and_Phi_ddpg',
    load_file_critic ='C:\\demo\\other_branch\\Learning-based_Secure_Transmission_for_RIS_Aided_mmWave-UAV_Communications_with_Imperfect_CSI\\data\\mannal_store\\models\\Critic_G_and_Phi_ddpg'
    )
"""
agent_2.load_models(
    load_file_actor = 'C:\\demo\\other_branch\\Learning-based_Secure_Transmission_for_RIS_Aided_mmWave-UAV_Communications_with_Imperfect_CSI\\data\\mannal_store\\models\\Actor_UAV_ddpg',
    load_file_critic ='C:\\demo\\other_branch\\Learning-based_Secure_Transmission_for_RIS_Aided_mmWave-UAV_Communications_with_Imperfect_CSI\\data\\mannal_store\\models\\Critic_UAV_ddpg'
    )
>>>>>>> ffe98bc78fcde95c811f10b8c9220cac7d875392
meta_dic = {}
print("***********************system information******************************")
print("folder_name:     "+str(system.data_manager.time_stemp))
meta_dic['folder_name'] = system.data_manager.time_stemp
<<<<<<< HEAD
print("user_num:        "+str(system.UE_num))
meta_dic['user_num'] = system.UE_num
=======
print("user_num:        "+str(system.user_num))
meta_dic['user_num'] = system.user_num
print("if_dir:          "+str(system.if_dir_link))
meta_dic['if_dir_link'] = system.if_dir_link
print("if_with_RIS:     "+str(system.if_with_RIS))
meta_dic['if_with_RIS'] = system.if_with_RIS
print("if_user_m:       "+str(system.if_move_users))
meta_dic['if_move_users'] = system.if_move_users
>>>>>>> ffe98bc78fcde95c811f10b8c9220cac7d875392
print("RIS_ant_num:     "+str(system.RIS.ant_num))
meta_dic['system_RIS_ant_num'] = system.RIS.ant_num
print("UAV_ant_num:     "+str(system.UAV.ant_num))
meta_dic['system_UAV_ant_num'] = system.UAV.ant_num
print("if_movements:    "+str(system.if_movements))
meta_dic['system_if_movements'] = system.if_movements
<<<<<<< HEAD
=======
print("reverse_x_y:     "+str(system.reverse_x_y))
meta_dic['system_reverse_x_y'] = system.reverse_x_y
print("if_UAV_pos_state:"+str(system.if_UAV_pos_state))
meta_dic['if_UAV_pos_state'] = system.if_UAV_pos_state
>>>>>>> ffe98bc78fcde95c811f10b8c9220cac7d875392

print("ep_num:          "+str(episode_num))
meta_dic['episode_num'] = episode_num
print("step_num:        "+str(step_num))
meta_dic['step_num'] = step_num
print("***********************agent_1 information******************************")
tplt = "{0:{2}^20}\t{1:{2}^20}"
for i in agent_1_param_dic:
    parm = agent_1_param_dic[i]
    print(tplt.format(i, parm, chr(12288)))
meta_dic["agent_1"] = agent_1_param_dic

<<<<<<< HEAD



data_rate_all = []
print("***********************traning information******************************")
while episode_cnt < episode_num:
    UAV_pos_x = []
    UAV_pos_y = []

    UE1_pos_x = []
    UE1_pos_y = []

    UE2_pos_x = []
    UE2_pos_y = []

    UE3_pos_x = []
    UE3_pos_y = []

    RIS_pos_x = [system.RIS.coordinate[0]]
    RIS_pos_y = [system.RIS.coordinate[1]]


=======
print("***********************agent_2 information******************************")
for i in agent_2_param_dic:
    parm = agent_2_param_dic[i]
    print(tplt.format(i, parm, chr(12288)))
meta_dic["agent_2"] = agent_2_param_dic

system.data_manager.save_meta_data(meta_dic)

print("***********************traning information******************************")
while episode_cnt < episode_num:
    # 1 reset the whole system
>>>>>>> ffe98bc78fcde95c811f10b8c9220cac7d875392
    system.reset()
    step_cnt = 0
    score_per_ep = 0
    # 2 get the initial state
    if if_robust:
        tmp = system.observe()
<<<<<<< HEAD

        z = np.random.normal(size=len(tmp))
        observersion_1 = list(
            np.array(tmp) + 0.6 *1e-8* z
            )
        observersion_1[0:3] = observersion_1[0:3] + 0.6 * z[0:3]
    else:
        observersion_1 = system.observe()


    data_rate_per = []
    while step_cnt < step_num:

        UAV_pos_x.append(system.UAV.coordinate[0])
        UAV_pos_y.append(system.UAV.coordinate[1])

        for index in range(system.UE_num):
            if(index==0):
                UE1_pos_x.append(system.UE_list[index].coordinate[0])
                UE1_pos_y.append(system.UE_list[index].coordinate[1])
            elif(index==1):
                UE2_pos_x.append(system.UE_list[index].coordinate[0])
                UE2_pos_y.append(system.UE_list[index].coordinate[1])
            elif(index==2):
                UE3_pos_x.append(system.UE_list[index].coordinate[0])
                UE3_pos_y.append(system.UE_list[index].coordinate[1])



        step_cnt += 1

        if not system.render_obj.pause:

            action_1 = agent_1.choose_action(observersion_1, greedy=agent_1_param_dic["action_noise_factor"] * math.pow((1-episode_cnt / 70), 2))




  
            new_state_1, reward, done, info, data_rate = system.step(
                    action_0=action_1[-2],
                    action_1=action_1[-1],
                    G=action_1[0:0+2 * system.UAV.ant_num * system.UE_num],
                    Phi=action_1[0+2 * system.UAV.ant_num * system.UE_num:-2]
                )

            score_per_ep += reward
            if(data_rate>0):
                data_rate_per.append(data_rate)
            print(step_cnt, reward)

            agent_1.remember(observersion_1, action_1, reward, new_state_1, int(done))

            agent_1.learn()

            observersion_1 = new_state_1

            if done == True:
                break
    if(len(data_rate_per)):
        data_rate_all.append(np.mean(data_rate_per))
    else:
        data_rate_all.append(data_rate_all[-1])
    system.reset()
    print("ep_num: "+str(episode_cnt)+"   ep_score:  "+str(score_per_ep))
    episode_cnt +=1

    if((episode_cnt)%10 == 0):
        plt.figure('episode_cnt: '+str(episode_cnt))
        plt.scatter(RIS_pos_x, RIS_pos_y)
        plt.text(RIS_pos_x[0], RIS_pos_y[0], 'RIS', fontsize=15, verticalalignment="top", horizontalalignment="right")
        plt.text(UAV_pos_x[0], UAV_pos_y[0], 'UAV-start', fontsize=10, verticalalignment="top", horizontalalignment="right")
        plt.scatter(UAV_pos_x, UAV_pos_y)  # scatter绘制散点图
        plt.text(UAV_pos_x[-1], UAV_pos_y[-1], 'UAV-end', fontsize=10, verticalalignment="top", horizontalalignment="right")
        plt.xlim((-30, 30))
        plt.ylim((-5, 55))

        for index in range(system.UE_num):
            if(index==0):
                plt.text(UE1_pos_x[0], UE1_pos_y[0], 'UE1-start', fontsize=10, verticalalignment="top", horizontalalignment="right")
                plt.scatter(UE1_pos_x, UE1_pos_y)
                plt.text(UE1_pos_x[-1], UE1_pos_y[-1], 'UE1-end', fontsize=10, verticalalignment="top", horizontalalignment="right")
            if(index==1):
                plt.text(UE2_pos_x[0], UE2_pos_y[0], 'UE2-start', fontsize=10, verticalalignment="top", horizontalalignment="right")
                plt.scatter(UE2_pos_x, UE2_pos_y)
                plt.text(UE2_pos_x[-1], UE2_pos_y[-1], 'UE2-end', fontsize=10, verticalalignment="top", horizontalalignment="right")
            if(index==2):
                plt.text(UE3_pos_x[0], UE3_pos_y[0], 'UE3-start', fontsize=10, verticalalignment="top", horizontalalignment="right")
                plt.scatter(UE3_pos_x, UE3_pos_y)
                plt.text(UE3_pos_x[-1], UE3_pos_y[-1], 'UE3-end', fontsize=10, verticalalignment="top", horizontalalignment="right")

        plt.draw()  # 显示绘图
    
        plt.pause(1)
        plt.close()




    

    



plt.figure('episode_cnt: '+str(episode_cnt))
plt.scatter(RIS_pos_x, RIS_pos_y)
plt.text(RIS_pos_x[0], RIS_pos_y[0], 'RIS', fontsize=15, verticalalignment="top", horizontalalignment="right")
plt.text(UAV_pos_x[0], UAV_pos_y[0], 'UAV-start', fontsize=10, verticalalignment="top", horizontalalignment="right")
plt.scatter(UAV_pos_x, UAV_pos_y)  # scatter绘制散点图
plt.text(UAV_pos_x[-1], UAV_pos_y[-1], 'UAV-end', fontsize=10, verticalalignment="top", horizontalalignment="right")
plt.xlim((-30, 30))
plt.ylim((-5, 55))

for index in range(system.UE_num):
    if(index==0):
            plt.text(UE1_pos_x[0], UE1_pos_y[0], 'UE1-start', fontsize=10, verticalalignment="top", horizontalalignment="right")
            plt.scatter(UE1_pos_x, UE1_pos_y)
            plt.text(UE1_pos_x[-1], UE1_pos_y[-1], 'UE1-end', fontsize=10, verticalalignment="top", horizontalalignment="right")
    if(index==1):
        plt.text(UE2_pos_x[0], UE2_pos_y[0], 'UE2-start', fontsize=10, verticalalignment="top", horizontalalignment="right")
        plt.scatter(UE2_pos_x, UE2_pos_y)
        plt.text(UE2_pos_x[-1], UE2_pos_y[-1], 'UE2-end', fontsize=10, verticalalignment="top", horizontalalignment="right")
    if(index==2):
        plt.text(UE3_pos_x[0], UE3_pos_y[0], 'UE3-start', fontsize=10, verticalalignment="top", horizontalalignment="right")
        plt.scatter(UE3_pos_x, UE3_pos_y)
        plt.text(UE3_pos_x[-1], UE3_pos_y[-1], 'UE3-end', fontsize=10, verticalalignment="top", horizontalalignment="right")

plt.draw()  # 显示绘图
plt.close()
print(data_rate_all)
plt.plot(data_rate_all)
plt.show()
plt.pause(1)
plt.savefig('rate.png')
=======
        #z = np.random.multivariate_normal(np.zeros(2), 0.5*np.eye(2), size=len(tmp)).view(np.complex128)
        z = np.random.normal(size=len(tmp))
        observersion_1 = list(
            np.array(tmp) + 0.6 *1e-7* z
            )
    else:
        observersion_1 = system.observe()
    observersion_2 = list(system.UAV.coordinate)
    if episode_cnt == 80:
        print("break point")
    while step_cnt < step_num:
        # 1 count num of step in one episode
        step_cnt += 1
        # judge if pause the whole system
        if not system.render_obj.pause:
            # 2 choose action acoording to current state
            """
            epsilon = 0.5
            epsilon = epsilon + epsilon * (episode_cnt*100 + step_cnt)/(step_num*episode_num)
            ran = np.random.uniform()
            if ran < epsilon:
                action = agent.choose_action(observersion, greedy=0)
            else:
                action = list(np.random.uniform(-1,1,n_actions))
            """
            action_1 = agent_1.choose_action(observersion_1, greedy=agent_1_param_dic["action_noise_factor"] * math.pow((1-episode_cnt / episode_num), 2))
            action_2 = agent_2.choose_action(observersion_2, greedy=agent_2_param_dic["action_noise_factor"]* math.pow((1-episode_cnt / episode_num), 2))
            if if_BS:
                action_2[0]=0
                action_2[1]=0

            if if_Theta_fixed:
                action_1[0+2 * system.UAV.ant_num * system.user_num:] = len(action_1[0+2 * system.UAV.ant_num * system.user_num:])*[0]
                
            if if_G_fixed:
                action_1[0:0+2 * system.UAV.ant_num * system.user_num]=np.array([-0.0313, -0.9838, 0.3210, 1.0, -0.9786, -0.1448, 0.3518, 0.5813, -1.0, -0.2803, -0.4616, -0.6352, -0.1449, 0.7040, 0.4090, -0.8521]) * math.pow(episode_cnt / episode_num, 2) * 0.7
                #action_1[0:0+2 * system.UAV.ant_num * system.user_num]=len(action_1[0:0+2 * system.UAV.ant_num * system.user_num])*[0.5]
            # 3 get newstate, reward
            if system.if_with_RIS:
                new_state_1, reward, done, info = system.step(
                    action_0=action_2[0],
                    action_1=action_2[1],
                    G=action_1[0:0+2 * system.UAV.ant_num * system.user_num],
                    Phi=action_1[0+2 * system.UAV.ant_num * system.user_num:],
                    set_pos_x=action_2[0],
                    set_pos_y=action_2[1]
                )
                new_state_2 = list(system.UAV.coordinate)
            else:
                new_state_1, reward, done, info = system.step(
                    action_0=action_2[0],
                    action_1=action_2[1],
                    G=action_1[0:0+2 * system.UAV.ant_num * system.user_num],
                    set_pos_x=action_2[0],
                    set_pos_y=action_2[1]
                )
                new_state_2 = list(system.UAV.coordinate)

            score_per_ep += reward
            # 4 store state pair into mem pool
            agent_1.remember(observersion_1, action_1, reward, new_state_1, int(done))
            agent_2.remember(observersion_2, action_2, reward, new_state_2, int(done))
            # 5 update DDPG net
            agent_1.learn()
            #agent_2.learn()

            system.render_obj.render(0.001)
            observersion_1 = new_state_1
            observersion_2 = new_state_2
            if done == True:
                break
            
        else:
            system.render_obj.render_pause() 
            time.sleep(1)
    system.data_manager.save_file(episode_cnt=episode_cnt)
    system.reset()
    print("ep_num: "+str(episode_cnt)+"   ep_score:  "+str(score_per_ep))
    episode_cnt +=1
    if episode_cnt % 10 == 0:
        agent_1.save_models()
        agent_2.save_models()
>>>>>>> ffe98bc78fcde95c811f10b8c9220cac7d875392

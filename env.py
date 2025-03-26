#%matplotlib inline
import numpy as np
from RIS import *
from UAV import *
from UE import *
from channel import *
from math_tool import *
import torch

import matplotlib.pyplot as plt
from render import Render
from data_manager import DataManager



class UAV_RIS_UE_EAV(object):

    def __init__(self, UE_num = 1, fre = 28e9, RIS_ant_num = 16, UAV_ant_num=8, if_move_UEs = True, if_movements = True):

        self.if_move_UEs = if_move_UEs
        self.if_movements = if_movements
        self.UE_num = UE_num
        self.border = [(-25,25), (0, 50)]

        self.data_manager = DataManager(file_path='./data', store_list = ['beamforming_matrix', 'reflecting_coefficient', 'UAV_state', 'user_capacity', 'secure_capacity', 'attaker_capacity','G_power', 'reward','UAV_movement'])

        self.UAV = UAV(
            coordinate=self.data_manager.read_init_location('UAV', 0), 
            ant_num= UAV_ant_num, 
            max_movement_per_time_slot=0.3
        )
        self.UAV.G = np.mat(np.ones((self.UAV.ant_num, UE_num), dtype=complex), dtype=complex)
        self.power_factor = 1
        self.UAV.G_Pmax = abs(np.trace(self.UAV.G * self.UAV.G.H)) 
        self.UAV.G = np.sqrt(self.power_factor/self.UAV.G_Pmax)*self.UAV.G

        self.RIS = RIS(
                coordinate=self.data_manager.read_init_location('RIS', 0), 
                coor_sys_z=self.data_manager.read_init_location('RIS_norm_vec', 0), 
                ant_num=RIS_ant_num
            )

        self.UE_list = []
        
        for i in range(UE_num):
            UE_coordinate = self.data_manager.read_init_location('user', i)
            ue = UE(coordinate=UE_coordinate, index=i)
            ue.noise_power = -145
            self.UE_list.append(ue)

        self.H_UR = mmWave_channel(self.UAV, self.RIS, fre)
        self.h_U_k = []
        self.h_R_k = []


        for UE_k in self.UE_list:
            self.h_U_k.append(mmWave_channel(UE_k, self.UAV, fre))
            self.h_R_k.append(mmWave_channel(UE_k, self.RIS, fre))



        self.update_channel_capacity()


        self.render_obj = Render(self)      


    def reset(self):

        self.UAV.reset(coordinate=self.data_manager.read_init_location('UAV', 0))

        for i in range(self.UE_num):
            user_coordinate = self.data_manager.read_init_location('user', i)
            self.UE_list[i].reset(coordinate=user_coordinate)


        self.UAV.G = np.mat(np.ones((self.UAV.ant_num, self.UE_num), dtype=complex), dtype=complex)

        self.UAV.G_Pmax = abs(np.trace(self.UAV.G * self.UAV.G.H)) 

        self.RIS.Phi = np.mat(np.diag(np.ones(self.RIS.ant_num, dtype=complex)), dtype = complex)

        self.render_obj.t_index = 0

        self.H_UR.update_CSI()
        for h in self.h_U_k + self.h_R_k:
            h.update_CSI()

        self.update_channel_capacity()



    def observe(self):

        UAV_position_list = []
        UAV_position_list = list(self.UAV.coordinate)
        for en in range(len(UAV_position_list)):
            UAV_position_list[en] = UAV_position_list[en]/10


        comprehensive_channel_elements_list1 = []
        tmp_list0 = list(np.array(np.reshape(self.H_UR.channel_matrix, (1,-1)))[0])
        comprehensive_channel_elements_list1 += list(np.real(tmp_list0)) + list(np.imag(tmp_list0)) 
        for index in range(self.UE_num):
            tmp_list1 = list(np.array(np.reshape(self.h_U_k[index].channel_matrix, (1,-1)))[0])
            tmp_list2 = list(np.array(np.reshape(self.h_R_k[index].channel_matrix, (1,-1)))[0])
            comprehensive_channel_elements_list1 += list(np.real(tmp_list1)) + list(np.imag(tmp_list1)) 
            comprehensive_channel_elements_list1 += list(np.real(tmp_list2)) + list(np.imag(tmp_list2)) 


        return comprehensive_channel_elements_list1 + UAV_position_list
    

    def step(self, action_0 = 0, action_1 = 0, G = 0, Phi = 0):

        self.render_obj.t_index += 1

        
        if self.if_move_UEs:
            for i in range(self.UE_num):
                
                if(i==0):
                    self.UE_list[i].update_coordinate(0.6, -1/2 * math.pi)
                if(i==1):
                    self.UE_list[i].update_coordinate(0.6, -1/2 * math.pi)
                if(i==2):
                    self.UE_list[i].update_coordinate(0.6, 0*math.pi)

        if self.if_movements:
            move_x = action_0 * self.UAV.max_movement_per_time_slot
            move_y = action_1 * self.UAV.max_movement_per_time_slot
                
            self.UAV.coordinate[0] +=move_x
            self.UAV.coordinate[1] +=move_y
            self.data_manager.store_data([move_x, move_y], 'UAV_movement')

        

        
        # 2 update channel CSI
        
        for h in self.h_U_k  + self.h_R_k:
            h.update_CSI()


        self.H_UR.update_CSI()

        self.UAV.G = np.sqrt(self.power_factor/self.UAV.G_Pmax/1.8)*convert_list_to_complex_matrix(G, (self.UAV.ant_num, self.UE_num))
        

        self.RIS.Phi = convert_list_to_complex_diag(Phi, self.RIS.ant_num)

        self.update_channel_capacity()

        new_state = self.observe()

        reward, data_rate = self.reward()

        #reward = math.tanh(reward)
        done = False
        x, y = self.UAV.coordinate[0:2]
        if x < self.border[0][0] or x > self.border[0][1]:
            done = True
            reward = -100
        if y < self.border[1][0] or y > self.border[1][1]:
            done = True
            reward = -100
        self.data_manager.store_data([reward],'reward')
        return new_state, reward, done, [], data_rate

    def reward(self):
  
        reward = 0
        reward_ = 0
        P = np.trace(self.UAV.G * self.UAV.G.H)
        if abs(P) > abs(self.power_factor) :
            reward = (abs(self.power_factor) - abs(P))*100
            reward /= self.power_factor 
            #print('----------------', P)
            ss = [0]*self.UE_num
        else:
            ss = []
            for user in self.UE_list:
                r = user.capacity 
                if r < user.QoS_constrain:
                    reward_ += (r - user.QoS_constrain)/(self.UE_num)
                elif(user.index == 2):
                    reward += (r/(self.UE_num))*1-24/(self.UE_num)
                    reward = reward
                elif(user.index == 1):
                    reward += (r/(self.UE_num))*1-24/(self.UE_num)
                    
                elif(user.index == 0):
                    reward += (r/(self.UE_num))*1-24/(self.UE_num)
                ss.append(r)
            if reward_ < 0:
                reward = reward_ * self.UE_num * 10

            #print(P, ss)
        
        return reward, np.mean(ss)


    def update_channel_capacity(self):

        for ue in self.UE_list:
            ue.capacity = self.calculate_capacity_of_user_k(ue.index)





    def calculate_capacity_of_user_k(self, k):

        noise_power = self.UE_list[k].noise_power
        h_U_k = self.h_U_k[k].channel_matrix
        h_R_k = self.h_R_k[k].channel_matrix
        Psi = diag_to_vector(self.RIS.Phi)
        H_c = vector_to_diag(h_R_k).H * self.H_UR.channel_matrix
        G_k = self.UAV.G[:, k]
        G_k_ = 0
        if len(self.UE_list) == 1:
            G_k_ = np.mat(np.zeros((self.UAV.ant_num, 1), dtype=complex), dtype=complex)
        else:
            G_k_1 = self.UAV.G[:, 0:k]
            G_k_2 = self.UAV.G[:, k+1:]
            G_k_ = np.hstack((G_k_1, G_k_2))
        alpha_k = math.pow(abs((h_U_k.H + Psi.H * H_c) * G_k), 2)

        beta_k = dB_to_normal(noise_power) #+ math.pow(np.linalg.norm((h_U_k.H + Psi.H * H_c)*G_k_), 2)

        return math.log2(1 + abs(alpha_k / beta_k))


    def get_system_action_dim(self):
        """
        function used in main function to get the dimention of actions
        """
        result = 0
        # 0 UAV movement
        result += 2
        # 1 beamforming matrix dimention

        result += 2 * self.UAV.ant_num * self.UE_num
 
        result += self.RIS.ant_num
        return result

    def get_system_state_dim(self):
        """
        function used in main function to get the dimention of states
        """
        result = 0
        # users' and attackers' comprehensive channel
        #result += 2 * (self.UE_num) * self.UAV.ant_num
        result += 2 * (self.UE_num) * self.UAV.ant_num

        result += 2 * (self.UE_num) * self.RIS.ant_num

        result += 2 * (self.UAV.ant_num) * self.RIS.ant_num
        # UAV position
        result += 3

        return result
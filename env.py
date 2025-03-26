#%matplotlib inline
import numpy as np
<<<<<<< HEAD
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
=======
from entity import *
from channel import *
from math_tool import *

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from render import Render
from data_manager import DataManager
# s.t every simulition is the same model
np.random.seed(2)

class MiniSystem(object):
    """
    define mini RIS communication system with one UAV
        and one RIS and one user, one attacker
    """
    def __init__(self, UAV_num = 1, RIS_num = 1, user_num = 1, attacker_num = 1, fre = 28e9, RIS_ant_num = 16, UAV_ant_num=8, if_dir_link = 1, if_with_RIS = True, if_move_users = True, if_movements = True, reverse_x_y = (True, True), if_UAV_pos_state = True):
        self.if_dir_link = if_dir_link
        self.if_with_RIS = if_with_RIS
        self.if_move_users = if_move_users
        self.if_movements = if_movements
        self.if_UAV_pos_state = if_UAV_pos_state
        self.reverse_x_y = reverse_x_y
        self.user_num = user_num
        self.attacker_num = attacker_num
        self.border = [(-25,25), (0, 50)]
        # 1.init entities: 1 UAV, 1 RIS, many users and attackers
        self.data_manager = DataManager(file_path='./data', \
        store_list = ['beamforming_matrix', 'reflecting_coefficient', 'UAV_state', 'user_capacity', 'secure_capacity', 'attaker_capacity','G_power', 'reward','UAV_movement'])
        # 1.1 init UAV position and beamforming matrix
        self.UAV = UAV(
            coordinate=self.data_manager.read_init_location('UAV', 0), 
            ant_num= UAV_ant_num, 
            max_movement_per_time_slot=0.25)
        self.UAV.G = np.mat(np.ones((self.UAV.ant_num, user_num), dtype=complex), dtype=complex)
        self.power_factor = 100
        self.UAV.G_Pmax =  np.trace(self.UAV.G * self.UAV.G.H) * self.power_factor
        # 1.2 init RIS
        self.RIS = RIS(\
        coordinate=self.data_manager.read_init_location('RIS', 0), \
        coor_sys_z=self.data_manager.read_init_location('RIS_norm_vec', 0), \
        ant_num=RIS_ant_num)
        # 1.3 init users
        self.user_list = []
        
        for i in range(user_num):
            user_coordinate = self.data_manager.read_init_location('user', i)
            user = User(coordinate=user_coordinate, index=i)
            user.noise_power = -114
            self.user_list.append(user)

        # 1.4 init attackers
        self.attacker_list = []
        
        for i in range(attacker_num):
            attacker_coordinate = self.data_manager.read_init_location('attacker', i)
            attacker = Attacker(coordinate=attacker_coordinate, index=i)
            attacker.capacity = np.zeros((user_num))
            attacker.noise_power = -114
            self.attacker_list.append(attacker)
        # 1.5 generate the eavesdrop capacity array , shape: P X K
        self.eavesdrop_capacity_array= np.zeros((attacker_num, user_num))
        # 2.init channel
        self.H_UR = mmWave_channel(self.UAV, self.RIS, fre)
        self.h_U_k = []
        self.h_R_k = []
        self.h_U_p = []
        self.h_R_p = []
        for user_k in self.user_list:
            self.h_U_k.append(mmWave_channel(user_k, self.UAV, fre))
            self.h_R_k.append(mmWave_channel(user_k, self.RIS, fre))
        for attacker_p in self.attacker_list:
            self.h_U_p.append(mmWave_channel(attacker_p, self.UAV, fre))
            self.h_R_p.append(mmWave_channel(attacker_p, self.RIS, fre))

        # 3 update user and attaker channel capacity
        self.update_channel_capacity()

        # 4 draw system
        self.render_obj = Render(self)      
        
    def reset(self):
        """
        reset UAV, users, attackers, beamforming matrix, reflecting coefficient
        """
        # 1 reset UAV
        self.UAV.reset(coordinate=self.data_manager.read_init_location('UAV', 0))
        # 2 reset users
        for i in range(self.user_num):
            user_coordinate = self.data_manager.read_init_location('user', i)
            self.user_list[i].reset(coordinate=user_coordinate)
        # 3 reset attackers
        for i in range(self.attacker_num):
            attacker_coordinate = self.data_manager.read_init_location('attacker', i)
            self.attacker_list[i].reset(coordinate=attacker_coordinate)
        # 4 reset beamforming matrix
        self.UAV.G = np.mat(np.ones((self.UAV.ant_num, self.user_num), dtype=complex), dtype=complex)
        self.UAV.G_Pmax = np.trace(self.UAV.G * self.UAV.G.H) * self.power_factor
        # 5 reset reflecting coefficient
        """self.RIS = RIS(\
        coordinate=self.data_manager.read_init_location('RIS', 0), \
        coor_sys_z=self.data_manager.read_init_location('RIS_norm_vec', 0), \
        ant_num=16)"""
        self.RIS.Phi = np.mat(np.diag(np.ones(self.RIS.ant_num, dtype=complex)), dtype = complex)
        # 6 reset time
        self.render_obj.t_index = 0
        # 7 reset CSI
        self.H_UR.update_CSI()
        for h in self.h_U_k + self.h_U_p + self.h_R_k + self.h_R_p:
            h.update_CSI()
        # 8 reset capcaity
        self.update_channel_capacity()

    def step(self, action_0 = 0, action_1 = 0, G = 0, Phi = 0, set_pos_x = 0, set_pos_y = 0):
        """
        test step only move UAV and update channel
        """
        # 0 update render
        
        self.render_obj.t_index += 1
        # 1 update entities
        
        if self.if_move_users:
            self.user_list[0].update_coordinate(0.2, -1/2 * math.pi)
            self.user_list[1].update_coordinate(0.2, -1/2 * math.pi)
>>>>>>> ffe98bc78fcde95c811f10b8c9220cac7d875392

        if self.if_movements:
            move_x = action_0 * self.UAV.max_movement_per_time_slot
            move_y = action_1 * self.UAV.max_movement_per_time_slot
<<<<<<< HEAD
=======
            if self.reverse_x_y[0]:
                move_x = -move_x
            
            if self.reverse_x_y[1]:
                move_y = -move_y
>>>>>>> ffe98bc78fcde95c811f10b8c9220cac7d875392
                
            self.UAV.coordinate[0] +=move_x
            self.UAV.coordinate[1] +=move_y
            self.data_manager.store_data([move_x, move_y], 'UAV_movement')
<<<<<<< HEAD

        

        
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
=======
        else:
            set_pos_x = map_to(set_pos_x, (-1, 1), self.border[0])
            set_pos_y = map_to(set_pos_y, (-1, 1), self.border[1])
            self.UAV.coordinate[0] = set_pos_x
            self.UAV.coordinate[1] = set_pos_y

        # 2 update channel CSI
        
        for h in self.h_U_k + self.h_U_p + self.h_R_k + self.h_R_p:
            h.update_CSI()
        # !!! test to make direct link zero
        if self.if_dir_link == 0:
            for h in self.h_U_k + self.h_U_p:
                h.channel_matrix = np.mat(np.zeros(shape = np.shape(h.channel_matrix)), dtype=complex)
        if self.if_with_RIS == False:
            self.H_UR.channel_matrix = np.mat(np.zeros((self.RIS.ant_num, self.UAV.ant_num)), dtype=complex)
        else:
            self.H_UR.update_CSI()
        # 3 update beamforming matrix & reflecting phase shift
        """
        self.UAV.G = G
        self.RIS.Phi = Phi
        """
        self.UAV.G = convert_list_to_complex_matrix(G, (self.UAV.ant_num, self.user_num)) * math.pow(self.power_factor, 0.5)
        
        # fix beamforming matrix
        #self.UAV.G = np.mat(np.ones((self.UAV.ant_num, self.user_num), dtype=complex), dtype=complex) * math.pow(self.power_factor, 0.5)
        if self.if_with_RIS:
            self.RIS.Phi = convert_list_to_complex_diag(Phi, self.RIS.ant_num)
        # 4 update channel capacity in every user and attacker
        self.update_channel_capacity()
        # 5 store current system state to .mat
        self.store_current_system_sate()
        # 6 get new state
        new_state = self.observe()
        # 7 get reward
        reward = self.reward()
        # 8 calculate if UAV is cross the bourder
        reward = math.tanh(reward)
>>>>>>> ffe98bc78fcde95c811f10b8c9220cac7d875392
        done = False
        x, y = self.UAV.coordinate[0:2]
        if x < self.border[0][0] or x > self.border[0][1]:
            done = True
<<<<<<< HEAD
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
=======
            reward = -10
        if y < self.border[1][0] or y > self.border[1][1]:
            done = True
            reward = -10
        self.data_manager.store_data([reward],'reward')
        return new_state, reward, done, []

    def reward(self):
        """
        used in function step to get the reward of current step
        """
        reward = 0
        reward_ = 0
        P = np.trace(self.UAV.G * self.UAV.G.H)
        if abs(P) > abs(self.UAV.G_Pmax) :
            reward = abs(self.UAV.G_Pmax) - abs(P)
            reward /= self.power_factor 
        else:
            for user in self.user_list:
                r = user.capacity - max(self.eavesdrop_capacity_array[:, user.index])
                if r < user.QoS_constrain:
                    reward_ += r - user.QoS_constrain
                else:
                    reward += r/(self.user_num*2)
            if reward_ < 0:
                reward = reward_ * self.user_num * 10
        return reward
    
    def observe(self):
        """
        used in function main to get current state
        the state is a list with 
        """
        # users' and attackers' comprehensive channel
        comprehensive_channel_elements_list = []
        for entity in self.user_list + self.attacker_list:
            tmp_list = list(np.array(np.reshape(entity.comprehensive_channel, (1,-1)))[0])
            comprehensive_channel_elements_list += list(np.real(tmp_list)) + list(np.imag(tmp_list)) 
        UAV_position_list = []
        if self.if_UAV_pos_state:
            UAV_position_list = list(self.UAV.coordinate)

        return comprehensive_channel_elements_list + UAV_position_list

    def store_current_system_sate(self):
        """
        function used in step() to store system state
        """
        # 1 store beamforming matrix
        row_data = list(np.array(np.reshape(self.UAV.G, (1, -1)))[0,:])
        self.data_manager.store_data(row_data, 'beamforming_matrix')
        # 2 store reflecting coefficient matrix
        row_data = list(np.array(np.reshape(diag(self.RIS.Phi), (1,-1)))[0,:])      
        self.data_manager.store_data(row_data, 'reflecting_coefficient')
        # 3 store UAV state
        row_data = list(self.UAV.coordinate)
        self.data_manager.store_data(row_data, 'UAV_state')
        # 4 store user_capicity
        row_data = [user.secure_capacity for user in self.user_list] \
        + [user.capacity for user in self.user_list]
        # 5 store G_power
        row_data = [np.trace(self.UAV.G*self.UAV.G.H), self.UAV.G_Pmax]
        self.data_manager.store_data(row_data, 'G_power')
        row_data = []
        for user in self.user_list:
            row_data.append(user.capacity)
        self.data_manager.store_data(row_data, 'user_capacity')

        row_data = []
        for attacker in self.attacker_list:
            row_data.append(attacker.capacity)
        self.data_manager.store_data(row_data, 'attaker_capacity')

        row_data = []
        for user in self.user_list:
            row_data.append(user.secure_capacity)
        self.data_manager.store_data(row_data, 'secure_capacity')


    def update_channel_capacity(self):
        """
        function used in step to calculate user and attackers' capacity 
        """
        # 1 calculate eavesdrop rate
        for attacker in self.attacker_list:
            attacker.capacity = self.calculate_capacity_array_of_attacker_p(attacker.index)
            self.eavesdrop_capacity_array[attacker.index, :] = attacker.capacity
            # remmeber to update comprehensive_channel
            attacker.comprehensive_channel = self.calculate_comprehensive_channel_of_attacker_p(attacker.index)
        # 2 calculate unsecure rate
        for user in self.user_list:
            user.capacity = self.calculate_capacity_of_user_k(user.index)
            # 3 calculate secure rate
            user.secure_capacity = self.calculate_secure_capacity_of_user_k(user.index)
            # remmeber to update comprehensive_channel
            user.comprehensive_channel = self.calculate_comprehensive_channel_of_user_k(user.index)

    def calculate_comprehensive_channel_of_attacker_p(self, p):
        """
        used in update_channel_capacity to calculate the comprehensive_channel of attacker p
        """
        h_U_p = self.h_U_p[p].channel_matrix
        h_R_p = self.h_R_p[p].channel_matrix
        Psi = diag_to_vector(self.RIS.Phi)
        H_c = vector_to_diag(h_R_p).H * self.H_UR.channel_matrix
        return h_U_p.H + Psi.H * H_c

    def calculate_comprehensive_channel_of_user_k(self, k):
        """
        used in update_channel_capacity to calculate the comprehensive_channel of user k
        """
        h_U_k = self.h_U_k[k].channel_matrix
        h_R_k = self.h_R_k[k].channel_matrix
        Psi = diag_to_vector(self.RIS.Phi)
        H_c = vector_to_diag(h_R_k).H * self.H_UR.channel_matrix
        return h_U_k.H + Psi.H * H_c

    def calculate_capacity_of_user_k(self, k):
        """
        function used in update_channel_capacity to calculate one user
        """     
        noise_power = self.user_list[k].noise_power
>>>>>>> ffe98bc78fcde95c811f10b8c9220cac7d875392
        h_U_k = self.h_U_k[k].channel_matrix
        h_R_k = self.h_R_k[k].channel_matrix
        Psi = diag_to_vector(self.RIS.Phi)
        H_c = vector_to_diag(h_R_k).H * self.H_UR.channel_matrix
        G_k = self.UAV.G[:, k]
        G_k_ = 0
<<<<<<< HEAD
        if len(self.UE_list) == 1:
=======
        if len(self.user_list) == 1:
>>>>>>> ffe98bc78fcde95c811f10b8c9220cac7d875392
            G_k_ = np.mat(np.zeros((self.UAV.ant_num, 1), dtype=complex), dtype=complex)
        else:
            G_k_1 = self.UAV.G[:, 0:k]
            G_k_2 = self.UAV.G[:, k+1:]
            G_k_ = np.hstack((G_k_1, G_k_2))
        alpha_k = math.pow(abs((h_U_k.H + Psi.H * H_c) * G_k), 2)
<<<<<<< HEAD

        beta_k = dB_to_normal(noise_power) #+ math.pow(np.linalg.norm((h_U_k.H + Psi.H * H_c)*G_k_), 2)

        return math.log2(1 + abs(alpha_k / beta_k))

=======
        beta_k = math.pow(np.linalg.norm((h_U_k.H + Psi.H * H_c)*G_k_), 2) + dB_to_normal(noise_power) * 1e-3
        return math.log10(1 + abs(alpha_k / beta_k))

    def calculate_capacity_array_of_attacker_p(self, p):
        """
        function used in update_channel_capacity to calculate one attacker capacities to K users
        output is a K length np.array ,shape: (K,)
        """
        K = len(self.user_list)
        noise_power = self.attacker_list[p].noise_power
        h_U_p = self.h_U_p[p].channel_matrix
        h_R_p = self.h_R_p[p].channel_matrix
        Psi = diag_to_vector(self.RIS.Phi)
        H_c = vector_to_diag(h_R_p).H * self.H_UR.channel_matrix
        if K == 1:
            G_k = self.UAV.G
            G_k_ = np.mat(np.zeros((self.UAV.ant_num, 1), dtype=complex), dtype=complex)
            alpha_p = math.pow(abs((h_U_p.H + Psi.H * H_c) * G_k), 2)
            beta_p = math.pow(np.linalg.norm((h_U_p.H + Psi.H * H_c)*G_k_), 2) + dB_to_normal(noise_power) * 1e-3
            return np.array([math.log10(1 + abs(alpha_p / beta_p))])
        else:
            result = np.zeros(K)
            for k in range(K):
                G_k = G_k = self.UAV.G[:, k]
                G_k_1 = self.UAV.G[:, 0:k]
                G_k_2 = self.UAV.G[:, k+1:]
                G_k_ = np.hstack((G_k_1, G_k_2))
                alpha_p = math.pow(abs((h_U_p.H + Psi.H * H_c) * G_k), 2)
                beta_p = math.pow(np.linalg.norm((h_U_p.H + Psi.H * H_c)*G_k_), 2) + dB_to_normal(noise_power) * 1e-3
                result[k] = math.log10(1 + abs(alpha_p / beta_p))
            return result

    def calculate_secure_capacity_of_user_k(self, k):
        """
        function used in update_channel_capacity to calculate the secure rate of user k
        """
        user = self.user_list[k]
        R_k_unsecure = user.capacity
        R_k_maxeavesdrop = max(self.eavesdrop_capacity_array[:, k])
        return max(0, R_k_unsecure - R_k_maxeavesdrop)
>>>>>>> ffe98bc78fcde95c811f10b8c9220cac7d875392

    def get_system_action_dim(self):
        """
        function used in main function to get the dimention of actions
        """
        result = 0
        # 0 UAV movement
        result += 2
        # 1 beamforming matrix dimention
<<<<<<< HEAD

        result += 2 * self.UAV.ant_num * self.UE_num
 
=======
        if self.if_with_RIS:
            result += 2 * self.UAV.ant_num * self.user_num    
        else:
            result += 0
        # 2 RIS reflecting elements
>>>>>>> ffe98bc78fcde95c811f10b8c9220cac7d875392
        result += self.RIS.ant_num
        return result

    def get_system_state_dim(self):
        """
        function used in main function to get the dimention of states
        """
        result = 0
        # users' and attackers' comprehensive channel
<<<<<<< HEAD
        #result += 2 * (self.UE_num) * self.UAV.ant_num
        result += 2 * (self.UE_num) * self.UAV.ant_num

        result += 2 * (self.UE_num) * self.RIS.ant_num

        result += 2 * (self.UAV.ant_num) * self.RIS.ant_num
        # UAV position
        result += 3

        return result
=======
        result += 2 * (self.user_num + self.attacker_num) * self.UAV.ant_num
        # UAV position
        if self.if_UAV_pos_state:
            result += 3
        return result
>>>>>>> ffe98bc78fcde95c811f10b8c9220cac7d875392

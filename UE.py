import numpy as np

class UE(object):

    def __init__(self, coordinate, index, ant_num = 1, ant_type = 'single'):

        self.type = 'UE'
        self.coordinate = coordinate
        self.ant_num = ant_num
        self.ant_type = ant_type
        self.index = index
        self.coor_sys = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]

        # init the capacity
        self.capacity = 0
        self.QoS_constrain = 0.01
        # init the comprehensive_channel, (must used in env.py to init)
        self.comprehensive_channel = 0
        # init receive noise sigma in dB
        self.noise_power = -145

    def reset(self, coordinate):
        self.coordinate = coordinate
        
    def update_coordinate(self, distance_delta_d, direction_fai):

        delta_x = distance_delta_d * np.cos(direction_fai)
        delta_y = distance_delta_d * np.sin(direction_fai)
        self.coordinate[0] += delta_x
        self.coordinate[1] += delta_y



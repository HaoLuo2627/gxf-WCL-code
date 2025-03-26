import numpy as np


class UAV(object):

    def __init__(self, coordinate, index = 0, rotation = 0, ant_num=16, ant_type = 'ULA', max_movement_per_time_slot = 0.5):

        self.max_movement_per_time_slot = max_movement_per_time_slot
        self.type = 'UAV'
        self.coordinate = coordinate
        self.rotation = rotation
        self.ant_num = ant_num
        self.ant_type = ant_type
        self.coor_sys = [np.array([1, 0, 0]), np.array([0, -1, 0]), np.array([0, 0, -1])]
        self.index = index
        self.G = np.mat(np.zeros((ant_num, 1)))
        self.G_Pmax = 0

    def reset(self, coordinate):

        self.coordinate = coordinate






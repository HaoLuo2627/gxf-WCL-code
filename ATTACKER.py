
import numpy as np

class Attacker(object):
    """
    Attacker with single antenas
    """
    def __init__(self, coordinate, index, ant_num = 1, ant_type= 'single'):
        """
        coordinate is the init coordinate of Attacker, meters, np.array
        ant_num is the antenas number of Attacker
        """
        self.type = 'attacker'
        self.coordinate = coordinate
        self.ant_num = ant_num
        self.ant_type = ant_type
        self.index = index
        self.coor_sys = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]

        self.capacity = 0
        self.comprehensive_channel = 0
        self.noise_power = -114

    def reset(self, coordinate):

        self.coordinate = coordinate




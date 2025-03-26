import numpy as np


class RIS(object):

    def __init__(self, coordinate, coor_sys_z, ant_num=36, ant_type = 'UPA'):

        self.type = 'RIS'
        self.coordinate = coordinate
        self.ant_num = ant_num
        self.ant_type = ant_type
        coor_sys_z = coor_sys_z / np.linalg.norm(coor_sys_z)
        coor_sys_x = np.cross(coor_sys_z, np.array([0,0,1]))
        coor_sys_x = coor_sys_x / np.linalg.norm(coor_sys_x)
        coor_sys_y = np.cross(coor_sys_z, coor_sys_x)
        self.coor_sys = [coor_sys_x,coor_sys_y,coor_sys_z]


        self.Phi = np.mat(np.diag(np.ones(self.ant_num, dtype=complex)), dtype = complex)
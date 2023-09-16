import numpy as np
import mujoco
import mujoco_viewer
from control import Controller
from matplotlib import pyplot as plt

class Simulation():

    def __init__(self,xml,n,dt):
        self.model = mujoco.MjModel.from_xml_path(xml)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco_viewer.MujocoViewer(self.model,self.data)
        self.controller = Controller(6,self.model,self.data,dt)
        self.n = n
        self.dt = dt
        self.links = [str(i+1) for i in range(self.n)]

        self.gyrometer  = np.zeros((self.n,3))
        self.angular_acceleration  = np.zeros((self.n,3))
        self.accelerometer  = np.zeros((self.n,3))
        self.magnetometer  = np.zeros((self.n,3))

    def step(self):
        self.controller.input()
        self.data.qfrc_applied = self.controller.cmd_tau
        self.measurement()
        mujoco.mj_step(self.model, self.data)
        self.viewer.render()
        for i in range(6):
            if self.controller.t[i] > self.controller.T[i]:
                self.controller.randomize(i)

    def measurement(self):
        for i in range(self.n):
            self.angular_acceleration[i,:] = (self.data.sensor('gyro_'+self.links[i]).data - self.gyrometer[i,:])/self.dt
            self.gyrometer[i,:] = self.data.sensor('gyro_'+self.links[i]).data
            self.accelerometer[i,:] = self.data.sensor('acc_'+self.links[i]).data
            self.magnetometer[i,:] = self.data.sensor('mag_'+self.links[i]).data




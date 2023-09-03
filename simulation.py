import numpy as np
import mujoco
import mujoco_viewer
from matplotlib import pyplot as plt

class Simulation():

    def __init__(self,xml,m_a,m_g,m_m,cov_a,cov_g,cov_m,b1,b2):
        self.model = mujoco.MjModel.from_xml_path(xml)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco_viewer.MujocoViewer(self.model,self.data)
        self.dt = 0.01
        self.d1 = np.array([0, 0, -0.2])
        self.d2 = np.array([0, 0, 0.2])

        self.a1 = np.zeros((3,1))
        self.a2 = np.zeros((3,1))
        self.g1 = np.zeros((3,1))
        self.g2 = np.zeros((3,1))
        self.m1 = np.zeros((3,1))
        self.m2 = np.zeros((3,1))
        self.g1_d = np.zeros((3,1))
        self.g2_d = np.zeros((3,1))
        self.a1_corrected = np.zeros((3,1))
        self.a2_corrected = np.zeros((3,1))

        self.a1_log = np.zeros((3,1))
        self.a2_log = np.zeros((3,1))
        self.g1_log = np.zeros((3,1))
        self.g2_log = np.zeros((3,1))
        self.m1_log = np.zeros((3,1))
        self.m2_log = np.zeros((3,1))
        self.g1_d_log = np.zeros((3,1))
        self.g2_d_log = np.zeros((3,1))
        self.a1_corrected_log = np.zeros((3,1))
        self.a2_corrected_log = np.zeros((3,1))

        self.mean_accelerometer = m_a
        self.mean_gyrometer = m_g
        self.mean_magnetometer = m_m
        self.covariance_accelerometer = cov_a
        self.covariance_gyrometer = cov_g
        self.covariance_magnetometer = cov_m
        self.gyrometer_bias_1 = b1
        self.gyrometer_bias_2 = b2

        

    def step(self):
        if self.viewer.is_alive:
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_kinematics(self.model, self.data)
            self.viewer.render()
            self.observe()
            self.noise()
            self.register_log()
        else:
            self.viewer.close()

    def compensate_dynamics(self) :

        self.a1_corrected = self.a1 + np.cross(self.g1_d,self.d1) + np.cross(self.g1,np.cross(self.g1,self.d1))
        self.a2_corrected = self.a2 - np.cross(self.g2_d,self.d2) + np.cross(self.g2,np.cross(self.g2,self.d2))

    def observe(self):
        self.g1_d[:,0] = (self.data.sensor('gyro_1').data.copy() - self.g1[:,0])/(self.dt)
        self.g2_d[:,0] = (self.data.sensor('gyro_2').data.copy() - self.g2[:,0])/(self.dt)
        self.a1[:,0] = self.data.sensor('acc_1').data.copy()
        self.a2[:,0] = self.data.sensor('acc_2').data.copy()
        self.g1[:,0] = self.data.sensor('gyro_1').data.copy() + self.gyrometer_bias_1
        self.g2[:,0] = self.data.sensor('gyro_2').data.copy() + self.gyrometer_bias_2
        self.m1[:,0] = self.data.sensor('mag_1').data.copy()
        self.m2[:,0] = self.data.sensor('mag_2').data.copy()

    def register_log(self):
        self.a1_log = np.concatenate((self.a1_log, self.a1),1)
        self.a2_log = np.concatenate((self.a2_log, self.a2),1)
        self.g1_log = np.concatenate((self.g1_log, self.g1),1)
        self.g2_log = np.concatenate((self.g2_log, self.g2),1)
        self.m1_log = np.concatenate((self.m1_log, self.m1),1)
        self.m2_log = np.concatenate((self.m2_log, self.m2),1)
        self.g1_d_log = np.concatenate((self.g1_d_log, self.g1_d),1)
        self.g2_d_log = np.concatenate((self.g2_d_log, self.g2_d),1)
        self.a1_corrected_log = np.concatenate((self.a1_corrected_log, self.a1_corrected),1)
        self.a2_corrected_log = np.concatenate((self.a2_corrected_log, self.a2_corrected),1)

    def noise(self):
        self.a1[:,0] += np.random.multivariate_normal(self.mean_accelerometer,self.covariance_accelerometer)
        self.a2[:,0] += np.random.multivariate_normal(self.mean_accelerometer,self.covariance_accelerometer)
        self.g1[:,0] += np.random.multivariate_normal(self.mean_gyrometer,self.covariance_gyrometer) 
        self.g2[:,0] += np.random.multivariate_normal(self.mean_gyrometer,self.covariance_gyrometer) 
        self.m1[:,0] += np.random.multivariate_normal(self.mean_magnetometer,self.covariance_magnetometer) 
        self.m2[:,0] += np.random.multivariate_normal(self.mean_magnetometer,self.covariance_magnetometer)

    def display(self):
        sensor_figure = plt.figure()
        acc1 = sensor_figure.add_subplot(321)
        acc1.plot(self.a1_log[0][1:], color = 'teal', label='x')
        acc1.plot(self.a1_log[1][1:], color = 'lightsalmon', label='y')
        acc1.plot(self.a1_log[2][1:], color = 'seagreen', label = 'z')
        acc1.set_ylabel(r"acceleration $[m/s^2]$")
        acc2 = sensor_figure.add_subplot(322)
        acc2.plot(self.a2_log[0][1:], color = 'teal', label='x')
        acc2.plot(self.a2_log[1][1:], color = 'lightsalmon', label='y')
        acc2.plot(self.a2_log[2][1:], color = 'seagreen', label = 'z')
        gyr1 = sensor_figure.add_subplot(323)
        gyr1.plot(self.g1_log[0][1:], color = 'teal', label='x')
        gyr1.plot(self.g1_log[1][1:], color = 'lightsalmon', label='y')
        gyr1.plot(self.g1_log[2][1:], color = 'seagreen', label = 'z')
        gyr1.set_ylabel(r"angular velocity $[rad/s]$")
        gyr2 = sensor_figure.add_subplot(324)
        gyr2.plot(self.g2_log[0][1:], color = 'teal', label='x')
        gyr2.plot(self.g2_log[1][1:], color = 'lightsalmon', label='y')
        gyr2.plot(self.g2_log[2][1:], color = 'seagreen', label = 'z')
        mag1 = sensor_figure.add_subplot(325)
        mag1.plot(self.m1_log[0][1:], color = 'teal', label='x')
        mag1.plot(self.m1_log[1][1:], color = 'lightsalmon', label='y')
        mag1.plot(self.m1_log[2][1:], color = 'seagreen', label = 'z')
        mag1.set_ylabel(r"magnetic field $[T]$")
        mag2 = sensor_figure.add_subplot(326)
        mag2.plot(self.m2_log[0][1:], color = 'teal', label='x')
        mag2.plot(self.m2_log[1][1:], color = 'lightsalmon', label='y')
        mag2.plot(self.m2_log[2][1:], color = 'seagreen', label = 'z')
        plt.legend()
        plt.show()

    

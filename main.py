import numpy as np
from matplotlib import pyplot as plt
from simulation import Simulation
from UKF_3DOFs_orientation import UKF

#_____________________________HYPERPARAMETERS________________________________________

xml_path = 'C:\\Users\\gabin\\Desktop\\programation\\pendulum\\geom.xml'
mean_accelerometer = np.zeros(3)
mean_gyrometer = np.zeros(3)
mean_magnetometer = np.zeros(3)
covariance_accelerometer = np.array([[0.3,-0.05,0.02],[-0.05,0.2,0.027],[0.02,0.027,0.28]])
covariance_gyrometer =  np.array([[0.3277,0.0051,-0.1482],[0.0051,0.6782,0.0798],[-0.1482,0.0798,0.6974]])
covariance_magnetometer = 1e-4*np.array([[1.1,-0.33,0.4],[-0.33,1.47,-0.079],[0.4,-0.079,1.76]])
bias_gyrometer_1 = np.array([0.01,0,-0.02])
bias_gyrometer_2 = np.array([-0.01,0.03,0.01])

#____________________________________________________________________________________



sim = Simulation(xml_path, 
                 mean_accelerometer,
                 mean_gyrometer,
                 mean_magnetometer,
                 covariance_accelerometer,
                 covariance_gyrometer, 
                 covariance_magnetometer,
                 bias_gyrometer_1,
                 bias_gyrometer_2)

P = 1e-14 * np.eye(7)
Q = np.zeros((6,6))
Q[:3,:3] = covariance_gyrometer
Q[3:,3:] = np.diag([0.01,0.1,0.1])
R = np.zeros((12,12))
R[:3,:3] = covariance_accelerometer
R[3:6,3:6] = covariance_magnetometer
R[6:9,6:9] = covariance_accelerometer
R[9:,9:] = covariance_magnetometer
b = bias_gyrometer_2 - bias_gyrometer_1
observer = UKF(7,b,P,Q,R)

state = []

for _ in range(2000):
    sim.step()
    observer.step(sim.a1[:,0],
                  sim.a2[:,0],
                  sim.g1[:,0],
                  sim.g2[:,0],
                  sim.m1[:,0],
                  sim.m2[:,0],
                  sim.dt)
    state.append(observer.x)
sim.display()

fig = plt.figure()
for i in range(7):
    ax = fig.add_subplot(7,1,i+1)
    ax.plot([s[i] for s in state])
    ax.set_ylim([-1.1,1.1])
plt.show()
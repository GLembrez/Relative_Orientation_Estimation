import numpy as np
from matplotlib import pyplot as plt
from simulation import Simulation
from UKF_3DOFs_orientation import UKF
from mag_free_MEKF import MEKF


def quat_to_ax_angle(q) : 
    axis = np.zeros((3,))
    angle = 2*np.arccos(q[0])
    s = np.sqrt(1-q[0]*q[0])
    if s < 0.001:
        axis = q[1:]
    else :
        axis = q[1:]/s
    return axis, angle

def estimate(sim,observers, Q_gyrometers, Q_accelerometers, Q_magnetometers, b_gyrometer):
    angle = np.zeros((sim.n-1,1))
    d1 = np.array([0,0,-0.05])  # AO1
    d2 = np.array([0,0,-0.05])  # O2A
    for i in range(1,sim.n):
        A1 = sim.accelerometer[i-1,:] + np.cross(sim.angular_acceleration[i-1,:],d1) + np.cross(sim.gyrometer[i-1,:],np.cross(sim.gyrometer[i-1,:],d1))
        A2 = sim.accelerometer[i,:] - np.cross(sim.angular_acceleration[i,:],d2) + np.cross(sim.gyrometer[i,:],np.cross(sim.gyrometer[i,:],d2))
        A1 = noise(A1,np.zeros((3,)),Q_accelerometers)
        A2 = noise(A2,np.zeros((3,)),Q_accelerometers)
        g1 = noise(sim.gyrometer[i-1,:],b_gyrometer[i-1],Q_gyrometers)
        g2 = noise(sim.gyrometer[i,:],b_gyrometer[i],Q_gyrometers)
        m1 = noise(sim.magnetometer[i-1,:],np.zeros((3,)),Q_magnetometers)
        m2 = noise(sim.magnetometer[i,:],np.zeros((3,)),Q_magnetometers)
        observers[i-1].step(A1,A2,g1,g2,m1,m2,sim.dt)
        _,angle[i-1,:] = quat_to_ax_angle(observers[i-1].x[:4])
    return angle

def noise(x,b,Q) :
    noisy_x = x + np.random.multivariate_normal(b,Q)
    return noisy_x




#_____________________________HYPERPARAMETERS________________________________________

xml_path = 'multibody_arm.xml'
covariance_accelerometer = np.array([[0.3,-0.05,0.02],
                                     [-0.05,0.2,0.027],
                                     [0.02,0.027,0.28]])
covariance_gyrometer =  np.array([[0.3277,0.0051,-0.1482],
                                  [0.0051,0.6782,0.0798],
                                  [-0.1482,0.0798,0.6974]])
covariance_magnetometer = 1e-4*np.array([[1.1,-0.33,0.4],
                                         [-0.33,1.47,-0.079],
                                         [0.4,-0.079,1.76]])
bias_gyrometer = np.array([[0.01, 0., -0.03],
                           [0., 0.03, -0.01],
                           [-0.01, 0.03, -0.03],
                           [0.01, 0.02, 0.02],
                           [-0.04, 0.03, -0.01],
                           [0.01, 0., -0.03]])

#____________________________________________________________________________________

P = np.block([[1e-3*np.eye(4),np.zeros((4,3))],
              [np.zeros((3,4)),1e-10*np.eye(3)]])
Q = np.zeros((6,6))
Q[:3,:3] = covariance_gyrometer
Q[3:,3:] = np.diag([0.01,0.1,0.1])
R = np.zeros((12,12))
R[:3,:3] = covariance_accelerometer
R[3:6,3:6] = covariance_magnetometer
R[6:9,6:9] = covariance_accelerometer
R[9:,9:] = covariance_magnetometer
UKF_observers = []
for i in range(5) :
    b = bias_gyrometer[i+1,:] - bias_gyrometer[i,:]
    UKF_observers.append(UKF(7,b,P,Q,R))

sim = Simulation(xml_path,6,0.001)
q_real = np.zeros((sim.n,1))
q_estimated = np.zeros((sim.n-1,1))
q_desired = np.zeros((sim.n,1))
running = True
while running:
    if sim.viewer.is_alive:
        sim.step()
        # angles = estimate(sim,
        #                   UKF_observers,
        #                   covariance_gyrometer,
        #                   covariance_gyrometer,
        #                   covariance_magnetometer,
        #                   bias_gyrometer)
        # q_estimated = np.concatenate([q_estimated,angles],axis=1)
        qprov = np.zeros((sim.n,1))
        qprov[:,0] = sim.data.qpos
        q_real = np.concatenate([q_real,qprov],axis=1)
        qprov[:,0] = sim.controller.q_d
        q_desired = np.concatenate([q_desired,qprov],axis=1)
    else:
        sim.viewer.close()
        running = False
fig = plt.figure()
for i in range(1,sim.n) : 
    ax = fig.add_subplot(sim.n-1,1,i)
    ax.plot(q_real[i,:],color = 'teal',label='real')
    ax.plot(q_estimated[i-1,:],color='lightsalmon',label='estimated')
    ax.plot(q_desired[i,:])
    ax.legend()
plt.show()


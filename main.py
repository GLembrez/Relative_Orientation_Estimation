import numpy as np
from matplotlib import pyplot as plt
from simulation import Simulation
from UKF_3DOFs_orientation import UKF
from mag_free_MEKF import MEKF


def quat_to_ax_angle(q) : 
    axis = np.zeros((3,))
    angle = (2*np.arctan2(q[0],np.linalg.norm(q[1:])))
    s = np.sqrt(1-q[0]*q[0])
    if s < 0.001:
        axis = q[1:]
    else :
        axis = q[1:]/s
    return axis, angle

def estimate_UKF(sim,observers, Q_gyrometers, Q_accelerometers, Q_magnetometers, b_gyrometer):
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

def estimate_MEKF(sim,observers, Q_gyrometers, Q_accelerometers,b_gyrometer):
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
        observers[i-1].step(A1,A2,g1,g2,sim.dt)
        q = observers[i-1].hamilton_product(conjugate(observers[i-1].q1), observers[i-1].q2)
        _,angle[i-1,:] = quat_to_ax_angle(q)
    return angle

def conjugate(q):
    return np.array([q[0],-q[1],-q[2],-q[3]])

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

P1 = np.block([[1e-4*np.eye(4),np.zeros((4,3))],
              [np.zeros((3,4)),1e-14*np.eye(3)]])
Q1 = np.zeros((6,6))
Q1[:3,:3] = covariance_gyrometer
Q1[3:,3:] = np.diag([0.001,0.001,0.001])
R1 = np.zeros((12,12))
R1[:3,:3] = covariance_accelerometer
R1[3:6,3:6] = covariance_magnetometer
R1[6:9,6:9] = covariance_accelerometer
R1[9:,9:] = covariance_magnetometer
P2 = 1e-6 * np.eye(6)
Q2 = 1e-5 * np.eye(6)
R2 = 1e-3 * np.eye(3)
UKF_observers = []
MEKF_observers = []
for i in range(5) :
    b = bias_gyrometer[i+1,:] - bias_gyrometer[i,:]
    UKF_observers.append(UKF(7,b,P1,Q1,R1))
    MEKF_observers.append(MEKF(np.array([1,0,0,0]),np.array([1,0,0,0]),np.zeros((6,)),P2,Q2,R2))

sim = Simulation(xml_path,6,0.01)
q_real = np.zeros((sim.n,1))
q_estimated_UKF = np.zeros((sim.n-1,1))
q_estimated_MEKF = np.zeros((sim.n-1,1))
running = True
while running:
    if sim.viewer.is_alive:
        sim.step()
        angles_UKF = estimate_UKF(sim, UKF_observers, covariance_gyrometer, covariance_gyrometer, covariance_magnetometer, bias_gyrometer)
        angles_MEKF = estimate_MEKF(sim, MEKF_observers, covariance_gyrometer, covariance_gyrometer, bias_gyrometer)
        q_estimated_MEKF = np.concatenate([q_estimated_MEKF,angles_MEKF],axis=1)
        q_estimated_UKF = np.concatenate([q_estimated_UKF,angles_UKF],axis=1)
        qprov = np.zeros((sim.n,1))
        qprov[:,0] = sim.data.qpos
        q_real = np.concatenate([q_real,qprov],axis=1)
    else:
        sim.viewer.close()
        running = False
fig = plt.figure()
for i in range(1,sim.n) : 
    ax = fig.add_subplot(sim.n-1,1,i)
    ax.plot(np.abs(q_real[i,:]),color = 'teal',label='real')
    ax.plot(q_estimated_UKF[i-1,:],color='lightsalmon',label='estimated (UKF)')
    ax.plot(q_estimated_MEKF[i-1,:],color='red',label='estimated (MEKF)')
    ax.legend()
plt.show()


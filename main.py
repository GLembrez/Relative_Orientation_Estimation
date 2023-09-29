import numpy as np
from matplotlib import pyplot as plt
from simulation import Simulation
from UKF_3DOFs_orientation import UKF
from mag_free_MEKF import MEKF
from scipy.spatial.transform import Rotation
from scipy.signal import butter, lfilter, freqz

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def hamilton_product(q1,q2):
        q = np.zeros(4)
        q[0] = q1[0]*q2[0] - q1[1:].dot(q2[1:])
        q[1:] = q1[0]*q2[1:] + q2[0]*q1[1:] + np.cross(q1[1:],q2[1:])
        return q
 
def euler_from_quaternion(q_input):
    q = np.array([q_input[1],q_input[2],q_input[3],q_input[0]])
    rot = Rotation.from_quat(q)
    angles = rot.as_euler('xyz', degrees=False)
    return angles[0],angles[1],angles[2]

def get_euler_angles(sim):
    X,Y,Z = np.zeros((sim.n-1,1)),np.zeros((sim.n-1,1)),np.zeros((sim.n-1,1))
    for i in range(1,sim.n):
        q_prev = sim.data.body("link_"+str(i)).xquat
        q = sim.data.body("link_"+str(i+1)).xquat
        x,y,z = euler_from_quaternion(hamilton_product(q,conjugate(q_prev)))
        X[i-1],Y[i-1],Z[i-1] = x,y,z
    return X,Y,Z


def quat_to_ax_angle(q) : 
    axis = np.zeros((3,))
    angle = (2*np.arctan2(q[0],np.linalg.norm(q[1:])))
    s = np.sqrt(1-q[0]*q[0])
    if s < 0.001:
        axis = q[1:]
    else :
        axis = q[1:]/s
    return axis, np.pi - angle

def estimate_UKF(sim,observers):
    angle = np.zeros((sim.n-1,1))
    d1 = np.array([0,0,-0.05])  # AO1
    d2 = np.array([0,0,-0.05])  # O2A
    X,Y,Z = np.zeros((sim.n-1,1)),np.zeros((sim.n-1,1)),np.zeros((sim.n-1,1))
    for i in range(1,sim.n):
        a1 = sim.accelerometer[i-1,:]
        a2 = sim.accelerometer[i,:]
        g1 = sim.gyrometer[i-1,:]
        g2 = sim.gyrometer[i,:]
        A1 = a1 + np.cross(sim.angular_acceleration[i-1,:],d1) + np.cross(g1,np.cross(g1,d1))
        A2 = a2 - np.cross(sim.angular_acceleration[i,:],d2) + np.cross(g2,np.cross(g2,d2))
        A1,A2 = A1/np.linalg.norm(A1),A2/np.linalg.norm(A2)
        m1 = sim.magnetometer[i-1,:]
        m2 = sim.magnetometer[i,:]
        m1,m2 = m1/np.linalg.norm(m1),m2/np.linalg.norm(m2)
        observers[i-1].step(A1,A2,g1,g2,m1,m2,sim.dt)
        _,angle[i-1,:] = quat_to_ax_angle(observers[i-1].x[:4])
        x,y,z = euler_from_quaternion(observers[i-1].x[:4])
        X[i-1],Y[i-1],Z[i-1] = x,y,z
    return angle,X,Y,Z

def estimate_MEKF(sim,observers):
    angle = np.zeros((sim.n-1,1))
    d1 = np.array([0,0,-0.05])  # AO1
    d2 = np.array([0,0,-0.05])  # O2A
    X,Y,Z = np.zeros((sim.n-1,1)),np.zeros((sim.n-1,1)),np.zeros((sim.n-1,1))
    for i in range(1,sim.n):
        a1 = sim.accelerometer[i-1,:]
        a2 = sim.accelerometer[i,:]
        g1 = sim.gyrometer[i-1,:]
        g2 = sim.gyrometer[i,:]
        A1 = a1 + np.cross(sim.angular_acceleration[i-1,:],d1) + np.cross(g1,np.cross(g1,d1))
        A2 = a2 - np.cross(sim.angular_acceleration[i,:],d2) + np.cross(g2,np.cross(g2,d2))
        observers[i-1].step(A1,A2,g1,g2,sim.dt)
        q = observers[i-1].hamilton_product(observers[i-1].q2, conjugate(observers[i-1].q1) )
        _,angle[i-1,:] = quat_to_ax_angle(q)
        x,y,z = euler_from_quaternion(q)
        X[i-1],Y[i-1],Z[i-1] = x,y,z
    return angle,X,Y,Z

def conjugate(q):
    return np.array([q[0],-q[1],-q[2],-q[3]])

def generate_covariance_matrix(main,cross):
    Q = np.zeros((3,3))
    for r in range(3):
        Q[r,r] = np.random.uniform(cross,main,(1,))
        for c in range(r+1,3):
            Q[c,r] = np.random.uniform(-cross,cross,(1,))
            Q[r,c] = Q[c,r]
    return Q

def update_bias(b):
    n = np.shape(b)[0]
    sigma = 0.05
    dt = 0.01
    b = b + dt * np.random.multivariate_normal(np.zeros((n,)),sigma*np.eye(n),size=(3)).T
    return b

def RMSE(x,y):
    return np.linalg.norm(x-y)/np.sqrt(np.size(x))



#_____________________________HYPERPARAMETERS________________________________________

xml_path = 'multibody_arm.xml'
n_links = 6
covariance_accelerometer = np.zeros((n_links,3,3))
covariance_gyrometer = np.zeros((n_links,3,3))
covariance_magnetometer = np.zeros((n_links,3,3))
bias_gyrometer = np.zeros((n_links,3))
accelerometer_Q = generate_covariance_matrix(0.1,0.01)
gyrometer_Q = generate_covariance_matrix(0.1,0.01)
magnetometer_Q =  1e-4 * generate_covariance_matrix(1.,0.1)
for i in range(n_links):
    covariance_accelerometer[i,:,:] = accelerometer_Q
    covariance_gyrometer[i,:,:] = accelerometer_Q
    covariance_magnetometer[i,:,:] = magnetometer_Q
    bias_gyrometer[i,:] =  np.random.uniform(-0.005,0.005,(3,))
sim = Simulation(xml_path,6,0.01,covariance_accelerometer,covariance_gyrometer,covariance_magnetometer,bias_gyrometer)
d1 = np.array([0,0,-0.05])  # AO1
d2 = np.array([0,0,-0.05])  # O2A
#____________________________________________________________________________________



#________________________________CALIBRATION_________________________________________
Q_a = np.zeros((n_links,3,3))
# Q_a = covariance_accelerometer[0]
Q_g = covariance_gyrometer[1:] + covariance_gyrometer[:-1]
Q_m = np.zeros((n_links,3,3))
# Q_m = covariance_magnetometer[0] 
b_g = bias_gyrometer[1:] - bias_gyrometer[:-1]

order = 6
fs = 100     
cutoff = 20

duration_calibration = 1000
a = np.zeros((n_links,3,duration_calibration))
g = np.zeros((n_links,3,duration_calibration))
m = np.zeros((n_links,3,duration_calibration))
A = np.zeros((n_links,3,duration_calibration))
noise_A = np.zeros((n_links,3,duration_calibration))
noise_m = np.zeros((n_links,3,duration_calibration))
for t in range(duration_calibration):
    sim.step()
    for i in range(n_links):
        a[i,:,t] = sim.accelerometer[i,:]
        g[i,:,t] = sim.gyrometer[i,:]
        A[i,:,t] = a[i,:,t] - np.cross(sim.angular_acceleration[i,:],d2) + np.cross(g[i,:,t],np.cross(g[i,:,t],d2))
        A[i,:,t] = A[i,:,t]/np.linalg.norm(A[i,:,t])
        m[i,:,t] = sim.magnetometer[i,:]/np.linalg.norm(sim.magnetometer[i,:])

for i in range(n_links):
    for ax in range(3):
        noise_A[i,ax,:] = A[i,ax,:]-butter_lowpass_filter(A[i,ax,:],cutoff, fs, order)
        noise_m[i,ax,:] = m[i,ax,:]-butter_lowpass_filter(m[i,ax,:],cutoff, fs, order)

for link in range(n_links):
    for i in range(3):
        for j in range(3):
            Q_a[link,i,j] = 1/duration_calibration * (noise_A[link,i,:].dot(noise_A[link,j,:]))
            Q_m[link,i,j] = 1/duration_calibration * (noise_m[link,i,:].dot(noise_m[link,j,:]))

sim.data.qpos = np.array([0,0,0,0,0,0])
sim.data.qvel = np.array([0,0,0,0,0,0])
sim.controller.q_f = np.array([0,0,0,0,0,0])
for i in range(6):
    sim.controller.randomize(i)

#_____________________________________________________________________________________



P1 = np.block([[1e-4*np.eye(4),np.zeros((4,3))],
              [np.zeros((3,4)),1e-4*np.eye(3)]])
P2 = 1e-6 * np.eye(6)
Q2 = 1e-5 * np.eye(6)
R2 = 1e-3 * np.eye(3)
UKF_observers = []
MEKF_observers = []
for i in range(5) :
    Q1 = np.zeros((6,6))
    Q1[:3,:3] = Q_g[i]
    Q1[3:,3:] = 5*1e-2 * np.eye(3)
    R1 = np.zeros((12,12))
    R1[:3,:3] = 1e-2 * np.mean(Q_a,0)
    R1[3:6,3:6] = 1e-2 * np.mean(Q_m,0)
    R1[6:9,6:9] = 1e-2 * np.mean(Q_a,0)
    R1[9:,9:] = 1e-2 * np.mean(Q_m,0)
    b = bias_gyrometer[i+1,:] - bias_gyrometer[i,:]
    UKF_observers.append(UKF(7,b,P1,Q1,R1))
    MEKF_observers.append(MEKF(np.array([1,0,0,0]),np.array([1,0,0,0]),np.zeros((6,)),P2,Q2,R2))

q_real = np.zeros((sim.n,1))
q_estimated_UKF = np.zeros((sim.n-1,1))
q_estimated_MEKF = np.zeros((sim.n-1,1))
X_UKF,Y_UKF,Z_UKF = np.zeros((sim.n-1,1)),np.zeros((sim.n-1,1)),np.zeros((sim.n-1,1))
X_MEKF,Y_MEKF,Z_MEKF = np.zeros((sim.n-1,1)),np.zeros((sim.n-1,1)),np.zeros((sim.n-1,1))
X_real,Y_real,Z_real = np.zeros((sim.n-1,1)),np.zeros((sim.n-1,1)),np.zeros((sim.n-1,1))
running = True
# while running:
for i in range(10000):
    if sim.viewer.is_alive:
        bias_gyrometer = update_bias(bias_gyrometer)
        sim.get_biases(bias_gyrometer)
        sim.step()
        angles_UKF,x_UKF,y_UKF,z_UKF = estimate_UKF(sim, UKF_observers)
        angles_MEKF,x_MEKF,y_MEKF,z_MEKF = estimate_MEKF(sim, MEKF_observers)
        x_real,y_real,z_real = get_euler_angles(sim)
        X_UKF = np.concatenate([X_UKF,x_UKF],1)
        Y_UKF = np.concatenate([Y_UKF,y_UKF],1)
        Z_UKF = np.concatenate([Z_UKF,z_UKF],1)
        X_MEKF = np.concatenate([X_MEKF,x_MEKF],1)
        Y_MEKF = np.concatenate([Y_MEKF,y_MEKF],1)
        Z_MEKF = np.concatenate([Z_MEKF,z_MEKF],1)
        X_real = np.concatenate([X_real,x_real],1)
        Y_real = np.concatenate([Y_real,y_real],1)
        Z_real = np.concatenate([Z_real,z_real],1)
        q_estimated_MEKF = np.concatenate([q_estimated_MEKF,angles_MEKF],axis=1)
        q_estimated_UKF = np.concatenate([q_estimated_UKF,angles_UKF],axis=1)
        qprov = np.zeros((sim.n,1))
        qprov[:,0] = sim.data.qpos
        q_real = np.concatenate([q_real,qprov],axis=1)
    else:
        sim.viewer.close()
        running = False

sim.viewer.close()


fig = plt.figure()
for i in range(1,sim.n) : 
    ax = fig.add_subplot(sim.n-1,1,i)
    ax.plot(np.abs(q_real[i,:]),color = 'teal',label='real')
    ax.plot(q_estimated_UKF[i-1,:],color='lightsalmon',label='estimated (UKF)')
    ax.plot(q_estimated_MEKF[i-1,:],color='red',label='estimated (MEKF)')
    ax.set_ylim([0,2*np.pi])
    ax.legend()

plt.show()

# for i in range(sim.n-1):
#     fig = plt.figure()
#     ax1 = fig.add_subplot(3,1,1)
#     ax1.plot(X_real[i,:],color='teal',label='real')
#     ax1.plot(X_UKF[i,:],color='lightsalmon',label='estimated (UKF)')
#     ax1.plot(X_MEKF[i,:],color='red',label='estimated (MEKF)')
#     ax1.set_ylim([-np.pi,np.pi])
#     ax1.legend()
#     ax1 = fig.add_subplot(3,1,2)
#     ax1.plot(Y_real[i,:],color='teal',label='real')
#     ax1.plot(Y_UKF[i,:],color='lightsalmon',label='estimated (UKF)')
#     ax1.plot(Y_MEKF[i,:],color='red',label='estimated (MEKF)')
#     ax1.set_ylim([-np.pi,np.pi])
#     ax1.legend()
#     ax1 = fig.add_subplot(3,1,3)
#     ax1.plot(Z_real[i,:],color='teal',label='real')
#     ax1.plot(Z_UKF[i,:],color='lightsalmon',label='estimated (UKF)')
#     ax1.plot(Z_MEKF[i,:],color='red',label='estimated (MEKF)')
#     ax1.set_ylim([-np.pi,np.pi])
#     ax1.legend()

# plt.show()

RMSE_UKF = np.zeros((n_links))
RMSE_MEKF = np.zeros((n_links))
for i in range(1,n_links):
    RMSE_UKF[i] = RMSE(np.abs(q_real[i,:]),q_estimated_UKF[i-1,:])
    RMSE_MEKF[i] = RMSE(np.abs(q_real[i,:]),q_estimated_MEKF[i-1,:])
RMSE_UKF_avg = np.mean(RMSE_UKF)
RMSE_MEKF_avg = np.mean(RMSE_MEKF) 
print("UKF : ", RMSE_UKF_avg)
print(RMSE_UKF)
print("MEKF : ", RMSE_MEKF_avg)
print(RMSE_MEKF)
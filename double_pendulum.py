import time
import matplotlib.pyplot as plt
import numpy as np
import UKF_draft
import mujoco
import mujoco_viewer



def quat_to_aa(q):
  theta = 2 * np.arccos(q[0])
  if abs(1-abs(q[0]))<1e-4:
    x = q[1]
    y = q[2]
    z = q[3]
  else:
    x = q[1]/np.sqrt(1-q[0]**2)
    y = q[2]/np.sqrt(1-q[0]**2)
    z = q[3]/np.sqrt(1-q[0]**2)
  return theta, np.array([x,y,z])


def add_noise(a1,a2,g1,g2,m1,m2,b1,b2) :
  
  a1 += np.random.normal(0,0.1,3)
  a2 += np.random.normal(0,0.1,3)
  g1 += np.random.normal(0,0.1,3) + b1
  g2 += np.random.normal(0,0.1,3) + b2
  m1 += np.random.normal(0,0.01,3)
  m2 += np.random.normal(0,0.01,3) 

  return a1,a2,g1,g2,m1,m2

def compensate_dynamics(d1,d2,a1,a2,g1,g2,g1_d,g2_d) :

    a1_corrected = a1 + np.cross(g1_d,d1) + np.cross(g1,np.cross(g1,d1))
    a2_corrected = a2 - np.cross(g2_d,d2) + np.cross(g2,np.cross(g2,d2))

    return a1_corrected,a2_corrected

def Hamilton_product(q1,q2) :

  q = np.zeros(4)
  q[0] = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
  q[1] = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2]
  q[2] = q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1]
  q[3] = q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]

  return q

def inverse_quaternion(q) :

  inv_q = np.array([q[0], -q[1], -q[2], -q[3]])

  return inv_q



m = mujoco.MjModel.from_xml_path('C:\\Users\\gabin\\Desktop\\programation\\pendulum\\geom.xml')
d = mujoco.MjData(m)
viewer = mujoco_viewer.MujocoViewer(m, d)

a1_list = []
a2_list = []
g1_list = []
g2_list = []
m1_list = []
m2_list = []
q_real = []
q_estimated = []
aa_real = []
aa_estimated = []
bias_estimated = []
g1 = 0
g2 = 0
g1_d = 0
g2_d = 0
d1 = np.array([0, 0, -0.2])
d2 = np.array([0, 0, 0.2])

t = 0
b1 = np.array([0.1,-0.3,-0.3])
b2 = np.array([0.02,-0.1,0.3])
x,P,Q,R = UKF_draft.initialisation()

dt = 0.01
while True:
  
  iteration = 0
  if viewer.is_alive:    
    mujoco.mj_step(m, d)
    mujoco.mj_kinematics(m, d)
    viewer.render()
    g1_d = (d.sensor('gyro_1').data.copy() - g1)/(0.01)
    g2_d = (d.sensor('gyro_2').data.copy() - g2)/(0.01)
    a1 = d.sensor('acc_1').data.copy()
    a2 = d.sensor('acc_2').data.copy()
    g1 = d.sensor('gyro_1').data.copy()
    g2 = d.sensor('gyro_2').data.copy()
    m1 = d.sensor('mag_1').data.copy()
    m2 = d.sensor('mag_2').data.copy()
    a1,a2,g1,g2,m1,m2 = add_noise(a1,a2,g1,g2,m1,m2,b1,b2)
    a1_list.append(a1)
    a2_list.append(a2)
    m1_list.append(m1)
    m2_list.append(m2)
    g1_list.append(g1)
    g2_list.append(g2)
    quat2 = d.body("link_2").xquat
    quat1 = d.body("link_1").xquat
    q_real.append(np.array(quat2))
    axis,angle = quat_to_aa(quat2)
    aa_real.append(axis*angle)


    a1,a2 = compensate_dynamics(d1,d2,a1,a2,g1,g2,g1_d,g2_d)
    x,P = UKF_draft.UKF(x,P,Q,R,a1,a2,g1,g2,m1,m2,dt)
    q_estimated.append(x[:4])
    axis,angle = quat_to_aa(x)
    aa_estimated.append(axis*angle)
    bias_estimated.append(x[4:].copy())

    iteration += 1

  else:
    break

viewer.close()







sensor_figure = plt.figure()

acc1 = sensor_figure.add_subplot(321)
acc1.plot([x[0] for x in a1_list], color = 'teal', label='x')
acc1.plot([x[1] for x in a1_list], color = 'lightsalmon', label='y')
acc1.plot([x[2] for x in a1_list], color = 'seagreen', label = 'z')
acc1.set_ylabel(r"acceleration $[m/s^2]$")
acc2 = sensor_figure.add_subplot(322)
acc2.plot([x[0] for x in a2_list], color = 'teal', label='x')
acc2.plot([x[1] for x in a2_list], color = 'lightsalmon', label='y')
acc2.plot([x[2] for x in a2_list], color = 'seagreen', label = 'z')

gyr1 = sensor_figure.add_subplot(323)
gyr1.plot([x[0] for x in g1_list], color = 'teal', label='x')
gyr1.plot([x[1] for x in g1_list], color = 'lightsalmon', label='y')
gyr1.plot([x[2] for x in g1_list], color = 'seagreen', label = 'z')
gyr1.set_ylabel(r"angular velocity $[rad/s]$")
gyr2 = sensor_figure.add_subplot(324)
gyr2.plot([x[0] for x in g2_list], color = 'teal', label='x')
gyr2.plot([x[1] for x in g2_list], color = 'lightsalmon', label='y')
gyr2.plot([x[2] for x in g2_list], color = 'seagreen', label = 'z')

mag1 = sensor_figure.add_subplot(325)
mag1.plot([x[0] for x in m1_list], color = 'teal', label='x')
mag1.plot([x[1] for x in m1_list], color = 'lightsalmon', label='y')
mag1.plot([x[2] for x in m1_list], color = 'seagreen', label = 'z')
mag1.set_ylabel(r"magnetic field $[T]$")
mag2 = sensor_figure.add_subplot(326)
mag2.plot([x[0] for x in m2_list], color = 'teal', label='x')
mag2.plot([x[1] for x in m2_list], color = 'lightsalmon', label='y')
mag2.plot([x[2] for x in m2_list], color = 'seagreen', label = 'z')
plt.legend()


estimation_figure  = plt.figure()
for i in range(3) :
  ax = estimation_figure.add_subplot(3,1,i+1)
  ax.plot([q[i] for q in aa_estimated], color = 'teal', label = 'estimation')
  ax.plot([q[i] for q in aa_real], color = 'lightsalmon', label = 'measurement')
plt.legend()

bias_figure = plt.figure()
for i in range(3) :
  ax = bias_figure.add_subplot(3,1,i+1)
  ax.plot([q[i] for q in bias_estimated], color = 'teal', label = 'estimation')


plt.show()




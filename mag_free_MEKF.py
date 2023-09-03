import numpy as np 
from scipy.linalg import sqrtm

class UKF():

    def __init__(self,n,x_0,P_0,Q_0,R_0):
        """
        Initializes unscented kalman filter

        n   - state dimension
        P   - state covariance
        Q   - state equation noise covariance
        R   - measurement equation noise covariance
        x   - state mean [q1^T ; q2^T]
        L   - state noise Jacobian
        M   - measurement noise Jacobian
        """

        self.n = n
        self.x = x_0
        self.P = P_0
        self.Q = Q_0
        self.R = R_0
        self.L = np.zeros((n,np.shape(Q_0)[0])) # one equation for each state dimension
        self.M = np.zeros((6,np.shape(R_0)[0])) # 6 measurement equations

    def estimation(self,q,u,dt):
        """
        computes the new position given a measurement of the velocity

        q     - initial quaternion
        u     - angular velocity of the gyrometers
        dt    - duration of the time step
        """
        # add estimation of bias to gyrometer measurements
        u_n = np.linalg.norm(u)
        S = self.skew_quaternion(u)
        if u_n < 1e-5:
            E = np.eye(4)
        else:
            E = np.cos(dt*u_n/2)*np.eye(4) + 1/u_n * np.sin(dt*u_n/2) * S
        # the estimation of the mean of bias vector remains unchanged
        return E.dot(q)
    
    def step(self,g1,g2,dt):

        # estimation
        self.x[:4] = self.estimation(self.x[:4],g1,dt)
        self.x[4:] = self.estimation(self.x[4:],g2,dt)

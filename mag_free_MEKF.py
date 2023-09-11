import numpy as np

class MEKF():

    def __init__(self,q10,q20,x0,P0,Q0,R0):

        self.q1 = q10
        self.q2 = q20
        self.x = x0
        self.P = P0
        self.Q = Q0
        self.R = R0

    def hamilton_product(self,q1,q2):
        q = np.zeros(4)
        q[0] = q1[0]*q2[0] + q1[1:].dot(q2[1:])
        q[1:] = q1[0]*q2[1:] + q2[0]*q1[1:] + np.cross(q1[1:],q2[1:])
        return q

    def exp_q(self,x):
        x_n = np.linalg.norm(x)
        q = np.zeros(4)
        q[0] = np.cos(x_n)
        q[1:] = x/x_n * np.sin(x_n)
        return q

    def quat_to_matrix(self,q) :
        """
        Conversion of the rotation represented by the quaternion q to its SO3 representation
        """     
        R = np.array([[q[0]**2+q[1]**2-q[2]**2-q[3]**2, 2*(q[1]*q[2]+q[0]*q[3]), 2*(q[1]*q[3]-q[0]*q[2])],
                    [2*(q[1]*q[2]-q[0]*q[3]), q[0]**2-q[1]**2+q[2]**2-q[3]**2, 2*(q[2]*q[3]+q[0]*q[1])],
                    [2*(q[1]*q[3]+q[0]*q[2]), 2*(q[2]*q[3]-q[0]*q[1]), q[0]**2-q[1]**2-q[2]**2+q[3]**2]])
        return R
    
    def skew(self,x):
        """
        computes skew symmetric matrix associated with vector x of R3
        """
        S = np.array([[0    ,-x[2],x[1] ],
                      [x[2] ,0    ,-x[0]],
                      [-x[1],x[0] ,0    ]])
        return S

    def step(self,a1,a2,g1,g2,dt):

        # estimation
        exp_1 = self.exp_q(dt/2 * g1)
        self.q1 = self.hamilton_product(self.q1,exp_1)
        exp_2 = self.exp_q(dt/2 * g2)
        self.q2 = self.hamilton_product(self.q2,exp_2)
        L =np.block([[dt * self.quat_to_matrix(self.q1),np.zeros([3,3])],
                     [np.zeros([3,3]),dt * self.quat_to_matrix(self.q2)]])
        self.P = self.P + L.dot(self.Q.dot(L.T))

        # measurement
        H = np.block([self.quat_to_matrix(self.q1).dot(self.skew(a1)),
                      -self.quat_to_matrix(self.q2).dot(self.skew(a2))])
        S = H.dot(self.P.dot(H.T)) + self.R
        K = self.P.dot(H.T).dot(np.linalg.inv(S)) # K of size 6,3
        self.x = K.dot(self.quat_to_matrix(self.q2).dot(a2) - self.quat_to_matrix(self.q2).dot(a1))

        # relinearization
        self.q1 = self.hamilton_product(self.q1,self.exp_q(0.5*self.x[:3]))
        self.q2 = self.hamilton_product(self.q2,self.exp_q(0.5*self.x[3:]))
        # J1 = 1/np.linalg.norm(self.q1)**3 * np.outer(self.q1,self.q1)
        # J2 = 1/np.linalg.norm(self.q2)**3 * np.outer(self.q2,self.q2)
        # J = np.block([[J1,np.zeros((3,3))],[np.zeros((3,3)),J2]])
        # self.P = J.dot(self.P - K.dot(S.dot(K))).dot(J.T)




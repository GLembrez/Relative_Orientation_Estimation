import numpy as np 
from scipy.linalg import sqrtm
from scipy.spatial.transform import Rotation

class UKF():

    def __init__(self,n,b_0,P_0,Q_0,R_0):
        """
        Initializes unscented kalman filter

        n   - state dimension
        b_0 - initial guess for bias
        P   - state covariance
        Q   - state equation noise covariance
        R   - measurement equation noise covariance
        x   - state mean
        L   - state noise Jacobian
        M   - measurement noise Jacobian
        """

        self.n = n
        self.x = np.concatenate((np.array([1,0,0,0]),b_0),0)
        self.P = P_0
        self.Q = Q_0
        self.R = R_0
        self.L = np.zeros((n,np.shape(Q_0)[0])) # one equation for each state dimension
        self.M = np.zeros((6,np.shape(R_0)[0])) # 6 measurement equations

    def unscented_transform(self,n,f,args):
        """
        Computes unscented transform of state x:
            1. Computes the 2n+1 sigma points
            2. propagates sigma points through non linear function f

        n    - dimension of the output of f()
        args - arguments of f in a tuple
        """
        sqrt_P = np.sqrt(self.n) * sqrtm(self.P)
        X = np.zeros((self.n,2*self.n+1))
        sigma = np.zeros((n,2*self.n+1))
        X[:,0] = self.x 
        for i in range(self.n) :
            X[:,i+1] = self.x + sqrt_P[:,i]
            X[:,self.n+1+i] = self.x - sqrt_P[:,i]
        for i in range(2*self.n+1):
            sigma[:,i] = f(X[:,i],args)
        return sigma

    def compute_covariance(self,X):
        """
        Computess the covariance matrix of the multivariate signal X
        """
        n = np.shape(X)[0]
        covar = np.zeros((n,n))
        DX = np.zeros((n,2*self.n+1))
        for i in range(2*self.n+1) :
            DX[:,i] = np.mean(X,1) - X[:,i]
            covar = covar + np.outer(DX[:,i],DX[:,i])
        return covar
    
    def estimation(self,input,args):
        """
        applies the non linear state equation

        input - homogeneous to a linear acceleration [m/s^2]
        u     - angular velocity difference between the two gyrometers
        dt    - duration of the time step
        """
        (u,dt) = args
        # add estimation of bias to gyrometer measurements
        u = u - input[4:]
        u_n = np.linalg.norm(u)
        S = self.skew_quaternion(u)
        if u_n < 1e-5:
            E = np.eye(4)
        else:
            E = np.cos(dt*u_n/2)*np.eye(4) + 1/u_n * np.sin(dt*u_n/2) * S
        # the estimation of the mean of bias vector remains unchanged
        output = np.concatenate((E.dot(input[:4]),input[4:]),0)
        return output
    
    def measurement(self,input,y1):
        """
        applies the non linear measurement equation

        input - homogeneous to a linear acceleration [m/s^2]
        yi    - concatenation of the readings of the accelerometer and magnetometer
        """
        output = np.concatenate((self.quat_to_matrix(input[:4]).dot(y1[:3]),
                                 self.quat_to_matrix(input[:4]).dot(y1[3:])))

        return output
    
    def compute_L(self,dt):
        """
        Computes first order linearisation of estimation noise Jacobian
        """
        self.L[:4,:3] = -dt/2 * np.array([[-self.x[1] , -self.x[2] , -self.x[3] ],
                                          [ self.x[0] , -self.x[3] ,  self.x[2] ],
                                          [ self.x[3] ,  self.x[0] , -self.x[1] ],
                                          [-self.x[2] ,  self.x[1] ,  self.x[0] ]])
        self.L[4:,3:] = dt * np.eye(3)

    def compute_M(self):
        """
        Computes first order linearisation of measurement noise Jacobian
        """
        self.M[:3,:3] = -self.quat_to_matrix(self.x[:4])
        self.M[3:,3:6] = -self.quat_to_matrix(self.x[:4])
        self.M[:,6:] = np.eye(6)

    def skew_quaternion(self,u):
        """
        returns the skew symmetric matrix associated with the quaternion 
            q = [0,u]^T
        Let p be a quaternion, Sp = q x p where x is the Hamilton product 
        """
        S = np.array([[0    ,-u[0],-u[1],-u[2]],
                      [u[0] ,0    ,u[2] ,-u[1]],
                      [u[1] ,-u[2],0    ,u[0]] ,
                      [u[2] ,u[1] ,-u[0],0   ]])
        return S

    def skew(self,x):
        """
        computes skew symmetric matrix associated with vector x of R3
        """
        S = np.array([[0    ,-x[2],x[1] ],
                      [x[2] ,0    ,-x[0]],
                      [-x[1],x[0] ,0    ]])
        return S
    
    def quat_to_matrix(self,q0) :
        """
        Conversion of the rotation represented by the quaternion q to its SO3 representation
        """      
        q = np.array([q0[1],q0[2],q0[3],q0[0]])
        R = Rotation.from_quat(q).as_matrix()   
        # R = np.array([[q[0]**2+q[1]**2-q[2]**2-q[3]**2, 2*(q[1]*q[2]+q[0]*q[3]), 2*(q[1]*q[3]-q[0]*q[2])],
        #             [2*(q[1]*q[2]-q[0]*q[3]), q[0]**2-q[1]**2+q[2]**2-q[3]**2, 2*(q[2]*q[3]+q[0]*q[1])],
        #             [2*(q[1]*q[3]+q[0]*q[2]), 2*(q[2]*q[3]-q[0]*q[1]), q[0]**2-q[1]**2-q[2]**2+q[3]**2]])
        return R
    
    def step(self,A1,A2,G1,G2,M1,M2,dt):    
        """
        main function of the UKF: computes new mean and covariance from sensor readings
        
        Ai - reading of the accelerometer i
        Gi - reading of the gyrometer i
        Mi - reading of the magnetometer i
        dt - duration of the timestep (can be variable)

            1. gather measurements
            2. do unscented transform for estimation equation
            3. do unscented transform for measurement equation
            4. compute Kalman gain
            5. update mean and covariance of state 
        """
        
        ### GATHER MEASUREMENTS
        u = G2-G1
        y1 = np.concatenate((A1,M1))
        y2 = np.concatenate((A2,M2))

        ### ESTIMATION PHASE
        X = self.unscented_transform(7,self.estimation,(u,dt))
        self.x = np.mean(X,1)
        P_tmp = self.compute_covariance(X)
        self.compute_L(dt)
        self.P = 1/(2*self.n+1) * P_tmp + self.L.dot(self.Q.dot(self.L.T))
        
        ### MEASUREMENT PHASE
        Y2 = self.unscented_transform(6,self.measurement, y1)
        Y2_mean = np.mean(Y2,1)
        Py_tmp = 1/(2*self.n+1) * self.compute_covariance(Y2)
        self.compute_M()
        Py =  Py_tmp + self.M.dot(self.R.dot(self.M.T))

        ### CORRECTION PHASE
        Pxy = np.zeros((7,6)) 
        for i in range(2*self.n+1) :
            # compute cross covariance
            Pxy = Pxy + np.outer(self.x-X[:,i],(Y2_mean-Y2[:,i]))
        K =  (1/(2*self.n+1)*Pxy).dot(np.linalg.inv(Py))
        self.P = self.P - K.dot(Py.dot(K.T))
        self.x = self.x + K.dot(y2-Y2_mean)
        self.x = np.real(self.x)
        self.x[:4] = 1/np.linalg.norm(self.x[:4]) * self.x[:4]
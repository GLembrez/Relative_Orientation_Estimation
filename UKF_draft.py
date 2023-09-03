import numpy as np
from scipy.linalg import sqrtm

def UKF(x,P,Q,R,A1,A2,G1,G2,M1,M2,dt) :

    u = G2-G1
    y1 = np.concatenate((A1,M1))
    y2 = np.concatenate((A2,M2))

    sqrt_P = np.sqrt(7) * sqrtm(P)

    X = np.zeros((7,15))
    X[:,0] = x 
    for i in range(7) :
        X[:,i+1] = x + sqrt_P[:,i]
        X[:,8+i] = x - sqrt_P[:,i]

    for i in range(15) :
        M = state_equation_matrix(u,X[:,i],dt)
        X[:,i] = M.dot(X[:,i])
    
    x = 1/15 * np.sum(X,1)

    P_tmp = np.zeros((7,7))
    DX = np.zeros((7,15))
    for i in range(15) :
        DX[:,i] = x - X[:,i]
        P_tmp = P_tmp + np.outer(DX[:,i],DX[:,i].T)


    L = np.zeros((7,6))
    L[:4,:3] = -dt/2 * np.array([[-x[1] , -x[2] , -x[3] ],
                                 [ x[0] , -x[3] ,  x[2] ],
                                 [ x[3] ,  x[0] , -x[1] ],
                                 [-x[2] ,  x[1] ,  x[0] ]])
    L[4:,3:] = dt * np.eye(3)
    Q = L.dot(Q.dot(L.T))
    P = 1/15 * P_tmp + Q

    

    sqrt_P = np.sqrt(7) * sqrtm(P)
    X = np.zeros((7,15))
    X[:,0] = x 
    for i in range(7) :
        X[:,i+1] = x + sqrt_P[:,i]
        X[:,8+i] = x - sqrt_P[:,i]

    Y2_avg = np.zeros(6)
    Y2 = np.zeros((6,15))
    for i in range(15) :
        B = np.zeros((6,6))
        B[:3,:3] = quat_to_matrix(X[:,i])
        B[3:,3:] = quat_to_matrix(X[:,i])
        Y2_avg = Y2_avg + 1/15 * B.dot(y1)
        Y2[:,i] = B.dot(y1)

    Py_tmp = np.zeros((6,6))
    for i in range(15) :
        Py_tmp = Py_tmp + np.outer((Y2_avg - Y2[:,i]),(Y2_avg - Y2[:,i]).T)
    B = np.zeros((6,6))
    B[:3,:3] = quat_to_matrix(x)
    B[3:,3:] = quat_to_matrix(x)
    M = np.zeros((6,12))
    M[:,:6] = -B
    M[:,6:] = np.eye(6)
    R = M.dot(R.dot(M.T))
    Py = 1/15 * Py_tmp + R

    Pxy_tmp = np.zeros((7,6))
    for i in range(15) :
        Pxy_tmp = Pxy_tmp + np.outer(DX[:,i],(Y2_avg-Y2[:,i]).T)
    Pxy = 1/15 * Pxy_tmp

    K = Pxy.dot(np.linalg.inv(Py))
    P = P - K.dot(Py.dot(K.T))
    x = x + K.dot(y2-Y2_avg)
    x = np.real(x)
    x[:4] = 1/np.linalg.norm(x[:4]) * x[:4]


    return x,P



def state_equation_matrix(u,x,dt) :
    
    u = u - x[4:]
    n_u = np.linalg.norm(u) 
    c = np.cos(dt*n_u/2)
    s = 1/n_u * np.sin(dt*n_u/2)
    M = np.array([ [c      , -u[0]*s , -u[1]*s , -u[2]*s , 0 , 0 , 0 ],
                   [u[0]*s , c       , u[2]*s  , -u[1]*s , 0 , 0 , 0 ],
                   [u[1]*s , -u[2]*s , c       , u[0]*s  , 0 , 0 , 0 ],
                   [u[2]*s , u[1]*s  , -u[0]*s , c       , 0 , 0 , 0 ],
                   [ 0     , 0       , 0       ,  0      , 1 , 0 , 0 ],
                   [ 0     , 0       , 0       ,  0      , 0 , 1 , 0 ],
                   [ 0     , 0       , 0       ,  0      , 0 , 0 , 1 ] ])

    return M 


def quat_to_matrix(q) :

    R = np.array([[q[0]**2+q[1]**2-q[2]**2-q[3]**2, 2*(q[1]*q[2]+q[0]*q[3]), 2*(q[1]*q[3]-q[0]*q[2])],
                  [2*(q[1]*q[2]-q[0]*q[3]), q[0]**2-q[1]**2+q[2]**2-q[3]**2, 2*(q[2]*q[3]+q[0]*q[1])],
                  [2*(q[1]*q[3]+q[0]*q[2]), 2*(q[2]*q[3]-q[0]*q[1]), q[0]**2-q[1]**2-q[2]**2+q[3]**2]])

    return R

def initialisation() :

  x = np.zeros(7)
  x[0] = 1
  P = 1e-4 * np.eye(7)
  Q = 1e-1 * np.eye(6)
  R = 1e-2 * np.eye(12)

  return x,P,Q,R



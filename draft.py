import numpy as np

M = np.ones((3,3))
a = np.array([1,2,3,4])
b = (1/np.linalg.norm(a) * a)
print(np.linalg.norm(b))

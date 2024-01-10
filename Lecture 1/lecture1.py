import numpy as np

v = np.array([
    [1],
    [2]
])
u = np.array([1, 2])
p = np.array([[2],[3]])

print(v + p)
print(v.T.dot(p))

A = np.array([[1,2],[3,4]])

print(A[0,:])
print(A[1,:])
print(A[:,0])
print(A[:,1])

print(np.max(v))
print(np.max(A))
print(np.min(A))
print(np.average(v))
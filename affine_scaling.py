import numpy as np
from scipy.linalg import solve_triangular
import scipy as sp

#Test
A = np.array([[3.0, 2.0, 1.0, 1.0, 0.0], [2.0, 5.0, 3.0, 0.0, 1.0]])
c = np.array([[-2.0], [-3.0], [-4.0], [0.0], [0.0]])
b = np.array([[10], [15]])
x0 = np.array([[1.0, 1.0, 1.0, 4.0, 5.0]]).transpose()


def affine_scaling(A, b, c, x0, r = 0.9, eps = 0.0000001, verbose = False):
    x = x0
    for i in range(100):
        s = x * c
        B = A * x.transpose()
        Bt = B.transpose()
        factor = sp.linalg.cho_factor(B @ Bt)
        w = sp.linalg.cho_solve(factor, B @ s)
        d = Bt @ w - s
        err = sp.linalg.norm(d, sp.inf)
        if err < eps: return x
        if verbose: print("{:>3}".format(i), x.transpose(), (A @ x).transpose(), err)
        alpha = r / err
        x = x - alpha * x * d


def find_feasible(A, b, c, x0 = None, r = 0.9, eps = 0.0000001, verbose = False):
    if x0:
        x = x0
    else:
        x = np.array([[1.0]] * len(c))
    A = np.c_[A, b - A @ x]
    c = np.array([[1.0]] * len(c) + [[0.0]])
    x = np.r_[x, [[1]]]
    for i in range(100):
        s = x * c
        B = A * x.transpose()
        Bt = B.transpose()
        factor = sp.linalg.cho_factor(B @ Bt)
        w = sp.linalg.cho_solve(factor, B @ s)
        d = Bt @ w - s
        err = sp.linalg.norm(d, sp.inf)
        if err < eps: break
        if verbose: print("{:>3}".format(i), x.transpose(), (A @ x).transpose(), err)
        alpha = r / err
        x = x - alpha * x * d
    return x[:-1,:]


def affine_scaling2(A, b, c, x0, r = 0.9, eps = 0.0000001, verbose = False):
    x = x0
    for i in range(100):
        s = x * c
        B = A * x.transpose()
        Bt = B.transpose()
        R = sp.linalg.qr(Bt, mode='economic')[1]
        w = sp.linalg.cho_solve((R, False), B @ s)
        d = Bt @ w - s
        err = sp.linalg.norm(d, sp.inf)
        if err < eps: return x
        if verbose: print("{:>3}".format(i), x.transpose(), (A @ x).transpose(), err)
        alpha = r / err
        x = x - alpha * x * d

#def affine_scaling3(A, b, c, x0, r, eps, verbose=True):
#    x = x0
#    R = sp.linalg.qr(A.transpose(), mode='economic')[1]
#    for i in range(20):
#        s = x * c
#        B = A * x.transpose()
#        Bs = B @ s
#        Rx = R0 * x[:2]
#        w = sp.linalg.cho_solve((Rx, False), Bs)
#        d = Bt @ w - s
#        err = sum(abs(d))
#        if err < eps:
#            return x
#        if verbose: print("{:>3}".format(i), x.transpose(), (A @ x).transpose(), err)
#        alpha = r / sp.linalg.norm(d, sp.inf)
#        x = x - alpha * x * d


print(affine_scaling(A, b, c, x0))
print(affine_scaling2(A, b, c, x0))
#print(affine_scaling3(A, b, c, x0, 0.9, 0.000001, False))
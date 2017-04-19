"""
created on: 19-04-2017.
@author: eduardo

Test routines for matrix.py
"""
import numpy as np
import scipy.sparse.linalg as sp
import matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

numberPoints = 6


def infiniteWell(x):
    if x == 0 or x == 2:
        return np.inf
    else:
        return 0


A = matrix.A1Dfull(numberPoints, np.vectorize(infiniteWell), 0, 2)
energies, psi = sp.eigsh(A)

# start = time.time()
# u = sp.spsolve(A,f,permc_spec='NATURAL')
# stop_all = time.time() - start_all
# stop = time.time() - start

# print "TIME:", stop/stop2, stop_all
# u = u.reshape(N+1,N+1)
# X, Y = np.meshgrid(np.linspace(0,1,N+1),np.linspace(0,1,N+1))
# u_exact = np.sin(Y*X)
# print "MAX",    np.max(np.abs(u-u_exact))





# fig = plt.figure()
# ax = fig.gca(projection="3d")
#
# # imgplot = plt.imshow(u-u_exact)
# # imgplot.set_cmap('spectral')
# # plt.colorbar()
#
# surf = ax.plot_surface(X,Y,(u - u_exact).transpose(),antialiased=False,cmap=cm.coolwarm)
# ax.set_xlabel('$x$',size=18)
# ax.set_ylabel('$y$',size=18)
# ax.set_zlabel('$err(x,y)$',size=18)#
#
# # surf = ax.plot_surface(X,Y,u_exact,antialiased=False)
#
#
#
# plt.show()

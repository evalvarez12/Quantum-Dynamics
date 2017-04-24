"""
created on: 19-04-2017.
@author: eduardo

Definition of the matrices that discretize the Hamiltonians
in 1 and 2 dimensions
"""

import numpy as np
import scipy.sparse as sp


def A1D(numberPoints, potentialFunc, domainStart, domainLength):
    """
    Hamiltonian discretization in 1d without boundaries.
    Uses Explicit method.
    Input:
        Number of points to evaluate on (float)
        Potential function (vectorised function)
        Location where domain starts (float)
        Length of domain (float)
    Output:
        Matrix A (scipy sparse matrix)
    """
    h = domainLength/numberPoints # Δx

    x = np.linspace(domainStart, domainStart + domainLength, numberPoints-1)
    v = potentialFunc(x)

    a = np.ones(numberPoints-1)*(2+(h**2 * v))
    b = np.ones(numberPoints-2)*-1
    A = sp.diags(a, 0) + sp.diags(b, 1) + sp.diags(b, -1)
    return (1/h**2) * A


def A1Dfull(numberPoints, potentialFunc, domainStart, domainLength):
    """
    Hamiltonian discretization in 1d with Dirichlet boundary conditions.
    Uses Explicit method to compute RHS of (A.68) in jos' book.
    Here we take hbar = 2m = 1.
    Input:
        Number of points to evaluate on (float)
        Potential function (vectorised function)
        Location where domain starts (float)
        Length of domain (float)
    Output:
        Matrix A, the discretised Hamiltonian (scipy sparse matrix)
    """
    h = domainLength/numberPoints # Δx

    x = np.linspace(domainStart, domainStart + domainLength, numberPoints+1)
    v = potentialFunc(x)

    a = np.ones(numberPoints+1)*(2+(h**2 * v))
    b = np.ones(numberPoints)*-1

    a[0] = h**2
    a[numberPoints] = h**2

    b[0] = 0
    b[numberPoints-1] = 0

    A = sp.diags(a, 0) + sp.diags(b, 1) + sp.diags(b, -1)
    return (1/h**2) * A


def A2D(numberPoints, potentialFunc, domainStart, domainLength):
    """
    Hamiltonian discretization in 2d without boundaries.
    Here we take hbar = 2m = 1.
    Input:
        Number of points to evaluate in each axis direction (float)
        Potential function (vectorised function)
        Location where domain starts (tuple)
        Length of domain (float)
    Output:
        Matrix A, the discretised Hamiltonian (scipy sparse matrix)
    """
    h = domainLength/numberPoints  # Δx

    o = np.ones(numberPoints)
    x = np.linspace(domainStart[0], domainStart[0] + domainLength,
                    numberPoints-1)
    y = np.linspace(domainStart[1], domainStart[1] + domainLength,
                    numberPoints-1)
    x = np.kron(x, o)
    y = np.kron(o, y)
    v = potentialFunc(x, y)

    a = np.ones(numberPoints-1)*(2 + h**2 * v)
    b = np.ones(numberPoints-2)*(-1)

    Id = np.identity(numberPoints-1)

    A1d = sp.diags(a, 0) + sp.diags(b, 1) + sp.diags(b, -1)
    A = sp.kron(Id, A1d) + sp.kron(A1d, Id)
    return (1/h**2) * A


def _Ih(numberPoints):
    """
    Returns Identity matrix of given length with zeros on the extremes.
    """
    Id = np.ones(numberPoints+1)
    Id[0] = 0
    Id[numberPoints] = 0
    return sp.diags(Id)


def _Th(numberPoints, potentialFunc, domainStart, domainLength):
    """
    Create a section of the matrix A
    """
    h = 1./numberPoints

    o = np.ones(numberPoints+1)
    x = np.linspace(domainStart[0], domainStart[0] + domainLength,
                    numberPoints+1)
    y = np.linspace(domainStart[1], domainStart[1] + domainLength,
                    numberPoints+1)
    x = np.kron(x, o)
    y = np.kron(o, y)
    v = potentialFunc(x, y)
    a = np.ones(numberPoints+1)*(4. + h**2 * v)
    b = np.ones(numberPoints)*(-1)

    a[0] = h**2
    a[numberPoints] = h**2

    b[0] = 0
    b[numberPoints-1] = 0

    T = sp.diags(a, 0) + sp.diags(b, -1) + sp.diags(b, 1)
    return T

def A2Dfull(numberPoints, potentialFunc, domainStart, domainLength):
    """
    Hamiltonian discretization in 2D with dirichlet boundary conditions
    """
    h = 1./numberPoints
    a = np.ones(numberPoints+1)
    b = np.ones(numberPoints)*(-1)
    b2 = np.zeros(numberPoints+1)
    a[0] = 0
    a[numberPoints] = 0
    b[0] = 0
    b[numberPoints-1] = 0
    b2[0] = 1
    b2[numberPoints] = 1

    Center1 = sp.diags(a, 0)
    Center2 = sp.diags(b, 1)
    Center3 = sp.diags(b, -1)
    Bounds = sp.diags(b2, 0)

    T = _Th(numberPoints, potentialFunc, domainStart, domainLength)
    Id = _Ih(numberPoints)
    I_N = (h**2)*sp.identity(numberPoints+1)

    A = sp.kron(Center1, T) + sp.kron(Center2, Id) + sp.kron(Center3, Id) + sp.kron(Bounds, I_N)
    return (1/h**2)*A


# def f(N,c) :
#     h = 1./N
#     x = np.linspace(0,1,N+1)
#     y = np.linspace(0,1,N+1)
#     X = np.kron(np.ones(N+1),x)
#     Y = np.kron(y,np.ones(N+1))
#
#     f_vec = np.sin(X * Y)*(X**2 + Y**2 + c)
#
#
#     for i in range(N+1) :
#         # X boundaries
#         f_vec[i] = np.sin(X[i]*Y[i])
#         f_vec[N**2 + N + i] = np.sin(X[N**2 + N +i]*Y[N**2 + N +i])
#
#         # Y boundaries
#         f_vec[(N + 1)*i] = np.sin(X[(N+1)*i]*Y[(N+1)*i])
#         f_vec[N + (N + 1)*i] = np.sin(X[N + (N+1)*i]*Y[N + (N+1)*i])
#
#     for i in range(1,N) :
#         # X boundaries
#         f_vec[N + 1 + i] += (1/h**2)*f_vec[i]
#         f_vec[N**2 -1 + i] += (1/h**2)*f_vec[N**2 + N + i]
#
#         # Y boundaries
#         f_vec[(N + 1)*i + 1] += (1/h**2)*f_vec[(N + 1)*i]
#         f_vec[N + (N+1)*i - 1] += (1/h**2)*f_vec[N + (N+1)*i]
#     return f_vec

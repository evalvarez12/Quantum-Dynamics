# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 15:10:24 2017

@author: Eoin
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import simulation as sm
import time
import scipy.sparse as sp

# Define the system parameters
dim = 1
numberPoints = 256
dt = .001
dirichletBC = False
startPoint = 0
domainLength = 15

sign = -1
if dirichletBC:
    sign = 1

x = np.linspace(startPoint, startPoint + domainLength, numberPoints + sign)

def potentialWell(x):
    if x > 6 and x < 10:
        return 500
    else:
        return 0


# Create the simulation for the system
sim = sm.Simulation(dim=dim, potentialFunc=potentialWell,
                    dirichletBC=dirichletBC, numberPoints=numberPoints,
                    startPoint=startPoint, domainLength=domainLength,
                    dt=dt)

# Create the initial wave function
sim.setPsiPulse(energy=500, center=2)

#%% Evolution methods

def evolve_cgs(sim):
    """Evolve the system using Conjugate Gradient squared iteration."""
    sim.psi = sp.linalg.cgs(sim.A, sim.B.dot(sim.psi),
                             x0=sim.psi)[0]
    return np.absolute(sim.psi)**2

def evolve_bicgstab(sim):
    """Evolve the system using BIConjugate Gradient STABilized iteration."""
    sim.psi = sp.linalg.bicgstab(sim.A, sim.B.dot(sim.psi),
                                  x0=sim.psi)[0]
    return np.absolute(sim.psi)**2

def evolve_gmres(sim):
    """Evolve the system using Generalized Minimal RESidual iteration."""
    sim.psi = sp.linalg.gmres(sim.A, sim.B.dot(sim.psi),
                               x0=sim.psi)[0]
    return np.absolute(sim.psi)**2

def evolve_lgmres(sim):
    """Evolve the system using the LGMRES algorithm."""
    sim.psi = sp.linalg.lgmres(sim.A, sim.B.dot(sim.psi),
                                x0=sim.psi)[0]
    return np.absolute(sim.psi)**2

def evolve_qmr(sim):
    """Evolve the system using Quasi-Minimal Residual iteration."""
    sim.psi = sp.linalg.qmr(sim.A, sim.B.dot(sim.psi),
                             x0=sim.psi)[0]
    return np.absolute(sim.psi)**2
    
#%% Speed testing


def solver_test():
    '''Tests multiple solvers to find the fastest. Outputs times to console'''
    x = 10000
    
    start = time.clock()
    for i in range(x):
        sim.evolve()
    linalg_spsolve = time.clock() - start
    print('linalg_spsolve = ' + str(linalg_spsolve) + 's')
    
    sim.setPsiPulse(energy=500, center=2)
    start = time.clock()
    for i in range(x):
        evolve_cgs(sim)
    cgs = time.clock() - start
    print('cgs = ' + str(cgs) + 's')
    
    sim.setPsiPulse(energy=500, center=2)
    start = time.clock()
    for i in range(x):
        evolve_bicgstab(sim)
    bicgstab = time.clock() - start
    print('bicgstab = ' + str(bicgstab) + 's')

    sim.setPsiPulse(energy=500, center=2)
    start = time.clock()
    for i in range(x):
        evolve_gmres(sim)
    gmres = time.clock() - start
    print('gmres = ' + str(gmres) + 's')

    sim.setPsiPulse(energy=500, center=2)
    start = time.clock()
    for i in range(x):
        evolve_lgmres(sim)
    lgmres = time.clock() - start
    print('lgmres = ' + str(lgmres) + 's')
    
    sim.setPsiPulse(energy=500, center=2)
    start = time.clock()
    for i in range(x):
        evolve_qmr(sim)
    qmr = time.clock() - start
    print('qmr = ' + str(qmr) + 's')  
      

solver_test()
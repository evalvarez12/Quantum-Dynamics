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

# Define the system parameters
dim = 1
numberPoints = 256
dt = .001
dirichletBC = False
startPoint = 0
domainLength = 10

sign = -1
if dirichletBC:
    sign = 1

x = np.linspace(startPoint, startPoint + domainLength, numberPoints + sign)

def potentialWell(x):
    if x > 4 and x < 8:
        return 100
    else:
        return 0


# Create the simulation for the system
sim = sm.Simulation(dim=dim, potentialFunc=potentialWell,
                    dirichletBC=dirichletBC, numberPoints=numberPoints,
                    startPoint=startPoint, domainLength=domainLength,
                    dt=dt)

# Create the initial wave function
sim.setPsiPulse()


def solver_test():
    '''Tests multiple solvers to find the fastest'''
    x = 10000
    
    start = time.clock()
    for i in range(x):
        sim.evolve()
    linalg_spsolve = time.clock() - start
    print('linalg_spsolve = ' + str(linalg_spsolve) + 's')
    
    
    start = time.clock()
    for i in range(x):
        sim.evolve_cgs()
    cgs = time.clock() - start
    print('cgs = ' + str(cgs) + 's')
    
    sim.setPsiPulse()
    start = time.clock()
    for i in range(x):
        sim.evolve_bicgstab()
    bicgstab = time.clock() - start
    print('bicgstab = ' + str(bicgstab) + 's')

    sim.setPsiPulse()
    start = time.clock()
    for i in range(x):
        sim.evolve_gmres()
    gmres = time.clock() - start
    print('gmres = ' + str(gmres) + 's')

    sim.setPsiPulse()
    start = time.clock()
    for i in range(x):
        sim.evolve_lgmres()
    lgmres = time.clock() - start
    print('lgmres = ' + str(lgmres) + 's')
    
    sim.setPsiPulse()
    start = time.clock()
    for i in range(x):
        sim.evolve_qmr()
    qmr = time.clock() - start
    print('qmr = ' + str(qmr) + 's')  
      

solver_test()
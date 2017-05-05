"""
A simple animation for the 1D case
with potential well.

Created on: 24-04-2017.
@author: eduardo
"""
import quantum_plots as qplots

import numpy as np
import simulation as sm
import matplotlib.pyplot as plt


# Define the system parameters and domain
dim = 1
numberPoints = 256
dt = .001
dirichletBC = False
startPoint = 0
domainLength = 15

sign = -1
if dirichletBC:
    sign = 1


def potentialWell(x):
    '''A top hat potential function.'''
    mag = 200
    domain = [6, 10]
    if domain[0] < x < domain[1]:
        return mag
    else:
        return 0


potentialFunc = np.vectorize(potentialWell)


# Create the simulation for the system
sim = sm.Simulation(dim=dim, potentialFunc=potentialFunc,
                    dirichletBC=dirichletBC, numberPoints=numberPoints,
                    startPoint=startPoint, domainLength=domainLength,
                    dt=dt)

# Create the initial wave function
sim.setPsiPulse(pulse="plane", energy=500, center=2)

# System evolution and Animation
ani = qplots.animation1D(sim, psi='real', V=potentialFunc)
plt.show()

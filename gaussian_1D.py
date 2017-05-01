"""
A simple animation for the 1D case
with potential well.

Created on: 24-04-2017.
@author: eduardo
"""
import quantum_plots as qplots

import numpy as np
import simulation as sm


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


def gaussian(x):
    mag = 200
    center = 4
    std = 1
    V = mag * np.exp(-(x-center)**2/(2*std**2))
    return V


# Create the simulation for the system
sim = sm.Simulation(dim=dim, potentialFunc=gaussian,
                    dirichletBC=dirichletBC, numberPoints=numberPoints,
                    startPoint=startPoint, domainLength=domainLength,
                    dt=dt)

# Create the initial wave function
sim.setPsiPulse(pulse="plane", energy=500, center=2)
# plt.plot(x, sim.realPsi())
# plt.show()


ani = qplots.animation1D(sim, x, psi='real', V=gaussian)

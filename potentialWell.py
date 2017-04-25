"""
A simple animation for the 1D case
with potential well.

Created on: 24-04-2017.
@author: eduardo
"""
import quantum_plots as qplots

import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
# plt.plot(x, sim.realPsi())
# plt.show()


ani = qplots.OneD_animation(sim, x, V=np.vectorize(potentialWell))

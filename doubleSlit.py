"""
Simulation of the double slit experiment.

Created on: 28-04-2017.
@author: eduardo

"""
import numpy as np
import simulation as sm
import quantum_plots as qplots
import matplotlib.pyplot as plt


# Define the system parameters and domain
dim = 2
numberPoints = 100
dt = .001
dirichletBC = False
startPoint = [0, 0]
domainLength = 2


def doubleSlit(x, y):
    '''A potential wall with two identical apertures'''
    sS = .5   # Slit separation
    sW = .1  #  Slits width
    spX = 1 # X coordinate start
    spY = 1   # Y coordinate of center of slits
    bW = 0.03 # Barrier width
    if x > spX and x < spX+bW and (y < spY-sS/2. or y > spY+sS/2.):
        return 50000
    if x > spX and x < spX+bW and (y > spY-sS/2.+sW and y < spY+sS/2.-sW):
        return 50000
    else:
        return 0


# Create the simulation for the system
sim = sm.Simulation(dim=dim, potentialFunc=doubleSlit,
                    dirichletBC=dirichletBC, numberPoints=numberPoints,
                    startPoint=startPoint, domainLength=domainLength,
                    dt=dt)

# Create the initial wave function
sim.setPsiPulse(pulse="circular", energy=500, center=[.1, 1],vel=[1, 0], width=.3)

# System evolution and Animation
ani = qplots.animation2D(sim, psi="norm",
                         potentialFunc=doubleSlit, save=False)
plt.show()

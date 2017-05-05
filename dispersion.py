"""
Simulation of Rutherford dispersion.

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

sign = -1
if dirichletBC:
    sign = 1
allPoints = numberPoints + sign


def dispersion(x, y):
    '''1/r^2 Dispersive force around a defined center '''
    center = [0.8, 1.]
    r = np.linalg.norm([x - center[0], y - center[1]])
    return 10./r**2


def dispersionVis(x, y):
    '''An exaggereted potential function to make it more visisble'''
    center = [0.8, 1.]
    r = np.linalg.norm([x - center[0], y - center[1]])
    return 10./((r + 1)**2)


# Create the simulation for the system
sim = sm.Simulation(dim=dim, potentialFunc=dispersion,
                    dirichletBC=dirichletBC, numberPoints=numberPoints,
                    startPoint=startPoint, domainLength=domainLength,
                    dt=dt)

# Create the initial wave function
sim.setPsiPulse(pulse="plane", energy=500, vel=1, center=.1, width=.1)

# System evolution and Animation
ani = qplots.animation2D(sim, psi="norm",
                         potentialFunc=dispersionVis, save=False)
plt.show()

"""
Simulation of Rutherford dispersion.

Created on: 28-04-2017.
@author: eduardo

"""
import numpy as np
import simulation as sm
import quantum_plots as qplots


# Define the system parameters
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

x = np.linspace(startPoint[0], startPoint[0] + domainLength, allPoints)
y = np.linspace(startPoint[1], startPoint[1] + domainLength,
                allPoints).reshape(-1, 1)


def dispersion(x, y):
    center = [0.8, 1.]
    r = np.linalg.norm([x - center[0], y - center[1]])
    return 10./r**2


def dispersionVis(x, y):
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


ani = qplots.animation2D(sim, [x, y], allPoints, psi="norm",
                     potentialFunc=dispersionVis, save=False)

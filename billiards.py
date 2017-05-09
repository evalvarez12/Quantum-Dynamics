"""
Simulation of multiple wave packets with random starting positions and
velocities.

Created on: 28-04-2017.
@author: Eoin

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
domainLength = 1


def lorentzGas(x, y):
    '''Potential corresponding to the Lorentz Gas billiard'''
    center = [0.5, 0.5]
    r = np.linalg.norm([x - center[0], y - center[1]])
    if r < .2:
        return 20000
    else:
        return 0


# Create the simulation for the system
sim = sm.Simulation(dim=dim, potentialFunc=lorentzGas,
                    dirichletBC=dirichletBC, numberPoints=numberPoints,
                    startPoint=startPoint, domainLength=domainLength,
                    dt=dt)


sim.setPsiPulse(pulse="circular", energy=1000, center=[.3, .2], vel=[.4, .1],
                width=.1)

for i in range(50):
    sim.evolve()

# System evolution and Animation
ani = qplots.animation2D(sim, psi="norm", potentialFunc=lorentzGas,
                         save=False)
plt.show()

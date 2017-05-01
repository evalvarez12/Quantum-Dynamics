"""
Simulation of multiple wave packets with random starting positions and
velocities.

Created on: 28-04-2017.
@author: Eoin

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
domainLength = 1
pulseFreq = 100

sign = -1
if dirichletBC:
    sign = 1
allPoints = numberPoints + sign

x = np.linspace(startPoint[0], startPoint[0] + domainLength, allPoints)
y = np.linspace(startPoint[1], startPoint[1] + domainLength,
                allPoints).reshape(-1, 1)


def table(x, y):
    return 0


# Create the simulation for the system
sim = sm.Simulation(dim=dim, potentialFunc=table,
                    dirichletBC=dirichletBC, numberPoints=numberPoints,
                    startPoint=startPoint, domainLength=domainLength,
                    dt=dt)

# Create the initial wave function
# Psi = np.zeros((numberPoints + sign)**2)
#location = [np.random.choice(x, size = numPulses),
#            np.random.choice(y.flatten(), size = numPulses)]
# location = np.array([np.random.choice(x, size = numPulses),
            # np.random.choice(y.flatten(), size = numPulses)])
# location = np.swapaxes(location, 0, 1)
vel = np.random.uniform(low=-1, high=1, size=(2, 1))

# for i in range(numPulses):
#     sim.setPsiPulse(energy=500, center=location[i], vel_x=vel[0, i],
#                     vel_y=vel[1, i], width=.1)
#     Psi = Psi + sim.psi

# sim.psi = Psi
sim.setPsiPulse(pulse="circular", energy=500, center=[.5, .5], vel=[0, 0], width=.1)


ani = qplots.animation2D(sim, [x, y], allPoints, psi="norm",
                         potentialFunc=table, save=False)

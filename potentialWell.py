"""
created on: 24-04-2017.
@author: eduardo

A simple animation for the 1D case
with potential well
"""
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

# Animation stuff
fig, ax = plt.subplots()
line, = ax.plot(x, sim.normPsi())


def animate(i):
    global sim
    waveFuncNorm = sim.evolve()
    line.set_ydata(waveFuncNorm)
    return line,


def init():
    line.set_ydata(sim.normPsi())
    return line,

ani = animation.FuncAnimation(fig, animate, frames=600, interval=10)
plt.show()

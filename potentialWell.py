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
domainLength = 15

sign = -1
if dirichletBC:
    sign = 1

x = np.linspace(startPoint, startPoint + domainLength, numberPoints + sign)


def potentialWell(x):
    if x > 6 and x < 10:
        return 100
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


# Animation stuff
fig, ax = plt.subplots()
axes = plt.gca()
axes.set_ylim([-1, 1])

line, = ax.plot(x, sim.realPsi())


p = np.vectorize(potentialWell)(x)
plt.plot(x, p, 'r')

def animate(i):
    global sim
    sim.evolve()
    line.set_ydata(sim.realPsi())
    return line,


ani = animation.FuncAnimation(fig, animate, frames=600, interval=10)
plt.show()

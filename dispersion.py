"""
Simulation of Rutherford dispersion.

Created on: 28-04-2017.
@author: eduardo

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import simulation as sm


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
y = np.linspace(startPoint[1], startPoint[1] + domainLength, allPoints).reshape(-1, 1)


def dispersion(x, y):
    center = [0.8, 1.]
    r = np.linalg.norm([x - center[0], y - center[1]])
    return 1./r**2

# Create the simulation for the system
sim = sm.Simulation(dim=dim, potentialFunc=dispersion,
                    dirichletBC=dirichletBC, numberPoints=numberPoints,
                    startPoint=startPoint, domainLength=domainLength,
                    dt=dt)

# Create the initial wave function
sim.setPsiPulse(energy=500, center=.1, width=.1)
# plt.show()


# Animation stuff
fig = plt.figure()
im = plt.imshow(np.transpose(sim.normPsi().reshape(allPoints, allPoints)),
                animated=True, cmap=plt.get_cmap('jet'))

# plt.imshow(np.vectorize(doubleSlit)(x, y), cmap=plt.get_cmap('rainbow'), alpha=1)

def animate(i):
    global sim  # Breaks the animation when used in a function
    sim.evolve()
    im.set_array(np.transpose(sim.normPsi().reshape(allPoints, allPoints)))
    return im,

ani = animation.FuncAnimation(fig, animate, frames=600, interval=10, blit=True)

plt.show()

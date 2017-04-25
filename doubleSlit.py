"""
Simulation of the double slit experiment.

Created on: 28-04-2017.
@author: eduardo

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import simulation as sm


# Define the system parameters
dim = 2
numberPoints = 128
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


def doubleSlit(x, y):
    if x > 0.4 and x < 0.5 and (y < 0.8 or y > 1.2):
        return 5000
    else:
        return 0


# Create the simulation for the system
sim = sm.Simulation(dim=dim, potentialFunc=doubleSlit,
                    dirichletBC=dirichletBC, numberPoints=numberPoints,
                    startPoint=startPoint, domainLength=domainLength,
                    dt=dt)

# Create the initial wave function
sim.setPsiPulse(energy=500, center=.2, width=.1)
# plt.show()


# Animation stuff
fig = plt.figure()
im = plt.imshow(np.transpose(sim.normPsi().reshape(allPoints, allPoints)),
                animated=True, cmap=plt.get_cmap('jet'))
# plt.show()
plt.colorbar()
plt.imshow(np.vectorize(doubleSlit)(x, y), cmap=plt.get_cmap('rainbow'), alpha=1)
plt.colorbar()

# line, = ax1.plot(x, sim.realPsi())
# ax1.set_xlabel(r'$x$')
# ax1.set_ylabel(r'$y$')
# ax1.tick_params('y', colors='b')

def animate(i):
    global sim  # Breaks the animation when used in a function
    sim.evolve()
    im.set_array(np.transpose(sim.normPsi().reshape(allPoints, allPoints)))
    return im,

ani = animation.FuncAnimation(fig, animate, frames=600, interval=10, blit=True)

# if not V == 'none':
#     ax2 = ax1.twinx()
#     ax2.imshow(V(x, y))
#     # ax2.set_ylabel('$V(x)$')
#     # ax2.tick_params('y', colors='r')

plt.show()

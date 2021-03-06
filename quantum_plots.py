"""
Hold all the methods for plotting and saving animations.

Created on Tue Apr 25 00:38:00 2017
@author: Eoin
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os.path


def animation1D(sim, V='none', psi='real', time=100, save=False):
    """
    Make an animation of a 1D system.

    Inputs:
        sim: (simulation object) An object of the simulation class.
        x: (N, numpy vector) x-coordinates of the domain.
        V: (vectorised function) The potential function, or "none", to
            indicate it should not be plotted.
        psi: (string) "real" or "norm" to determine whether to plot
            Re(Ψ) or |Ψ|^2
        time: (int) Number of frames to animate.
        save: (Boolean) Whether the animation should be saved.
    Outputs:
        Displays animation with both the evolving wavefunction norm and the
            potential function influencing it.
    """
    # Animation stuff
    x = sim.domain()
    fig, ax1 = plt.subplots()
    if psi == 'real':
        ax1.set_ylabel('$Re(\psi(x))$')
        line, = ax1.plot(x, sim.realPsi())
    else:
        ax1.set_ylabel('$|\psi(x)|^2$')
        line, = ax1.plot(x, sim.normPsi())
    ax1.tick_params('y', colors='b')
    ax1.set_xlabel('x')

    def animate(i):
        sim.evolve()
        if psi == 'real':
            line.set_ydata(sim.realPsi())
        else:
            line.set_ydata(sim.normPsi())
        return line,

    ani = animation.FuncAnimation(fig, animate, frames=time, interval=20,
                                  blit=True)

    if V != 'none':
        ax2 = ax1.twinx()
        ax2.plot(x, np.vectorize(V)(x), 'r')
        ax2.set_ylabel('$V(x)$')
        ax2.tick_params('y', colors='r')

    if save:
        _save(ani, sim)

    return ani


def animation2D(sim, potentialFunc, psi="norm", time=100, save=False):
    """
    Make a 2D animation of a 2D system.

    Inputs:
        sim: (simulation object) An object of the simulation class.
        potentialFunc: (vectorised function) The potential function, or
            "none", to indicate it should not be plotted.
        psi: (string) "real" or "norm" to determine whether to plot
            Re(Ψ) or |Ψ|^2
        time: (int) Number of frames to animate.
        save: (Boolean) Whether the animation should be saved.
    Outputs:
        Displays animation with both the evolving wavefunction norm and the
            potential function influencing it.
    """
    domain = sim.domain()
    allPoints = sim.allPoints

    fig = plt.figure()
    im = plt.imshow(np.transpose(sim.normPsi().reshape(allPoints, allPoints)),
                    animated=True, cmap=plt.get_cmap('jet'), alpha=.9,
                    origin='lower')

    if potentialFunc != 'none':
        # Only plot the potential if running locally, not in notebook.
        potentialPlot = np.vectorize(potentialFunc)(domain[0], domain[1])
        plt.imshow(potentialPlot, cmap=plt.get_cmap('Greys'), alpha=1,
                   origin='lower')

    plt.xticks([])
    plt.yticks([])
    plt.xlabel(r'$x$', fontsize=18)
    plt.ylabel(r'$y$', fontsize=18)


    def animate(i):
        sim.evolve()
        if psi == "norm":
            im.set_array(np.transpose(sim.normPsi().reshape(allPoints,
                         allPoints)))
        else:
            im.set_array(np.transpose(sim.realPsi().reshape(allPoints,
                         allPoints)))

        return im,

    ani = animation.FuncAnimation(fig, animate, frames=time, interval=60,
                                  blit=True)

    if save:
        _save(ani, sim)

    return ani


def frame2D(sim, potentialFunc, psi="norm"):
    """
    Make a 2D frame of a given psi of a 2D system.

    Inputs:
        sim: (simulation object) An object of the simulation class.
        potentialFunc: (vectorised function) The potential function, or
            "none", to indicate it should not be plotted.
        psi: (string) "real" or "norm" to determine whether to plot
            Re(Ψ) or |Ψ|^2
        time: (int) Number of frames to animate.
    Outputs:
        Displays a frame with both the evolving wavefunction norm and the
            potential function influencing it.
    """
    domain = sim.domain()
    allPoints = sim.allPoints

    if potentialFunc != 'none':
        # Only plot the potential if running locally, not in notebook.
        potentialPlot = np.vectorize(potentialFunc)(domain[0], domain[1])
        plt.imshow(potentialPlot, cmap=plt.get_cmap('Greys'), alpha=1,
                   origin='lower')

    plt.imshow(np.transpose(sim.normPsi().reshape(allPoints, allPoints)),
               cmap=plt.get_cmap('jet'), alpha=.9, origin='lower')

    plt.xticks([])
    plt.yticks([])
    plt.xlabel(r'$x$', fontsize=18)
    plt.ylabel(r'$y$', fontsize=18)

def probabilityGraph(P):
    '''
    Plots probability over time
    '''
    fig, ax = plt.subplots()
    ax.set_ylabel("$\Sigma (\psi(x) \psi(x)')$")
    ax.set_xlabel('t')
    ax.set_title('Probability Evolution')

    plt.plot(P)

    plt.show
    return fig


def _save(ani, sim):
    """
    Save animation to file. Requires installation of ffmpeg.

    Warning: Because this is a continous animation, instead of a fixed length
    one, this will cause the animation to not be displayed and saved instead.
    Need to close the figure window in order to allow it to finish.
    TODO : Fix it so that this is less hacky and doesn't cause the kernel to
    crash.
    Inputs:
        ani: (matplotlib animation object) The animation to be saved.
        sim: (simulation object) An object of the simulation class.
    Outputs:
        A movie, saved in .mp4 format to a subfolder.
    """
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=60,
                    metadata=dict(artist='Eduardo Villaseñor & Eoin Horgan'),
                    bitrate=3000)

    # Make folder if it doesn't already exist
    if not os.path.exists('Saved Animations'):
        os.makedirs('Saved Animations')

    if sim.sign == 1:
        dBC = 'yes'
    else:
        dBC = 'no'

    filepath = os.path.join("Saved Animations",
                            "D = " + str(sim.dim) +
                            ", n = " + str(sim.numberPoints) +
                            ", L = " + str(sim.domainLength) +
                            ", dBC = " + dBC +
                            ", dt = " + str(sim.dt) + '.mp4')
    ani.save(filepath, writer=writer)

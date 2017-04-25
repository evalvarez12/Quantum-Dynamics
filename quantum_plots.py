# -*- coding: utf-8 -*-
"""
Hold all the methods for plotting and saving animations.

Created on Tue Apr 25 00:38:00 2017
@author: Eoin
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os.path


def OneD_animation(sim, x, V='none', psi='real', save=False):
    """
    Make an animation of a 1D system.

    Inputs:
        sim: (simulation object) An object of the simulation class.
        x: (N, numpy vector) x-coordinates of the domain.
        V: (vectorised function) The potential function, or "none", to
            indicate it should not be plotted.
        psi: (string) "real" or "norm" to determine whether to plot
            Re(Ψ) or |Ψ|^2
        save: (Boolean) Whether the animation should be saved.
    Outputs:
        Displays animation with both the evolving wavefunction norm and the
            potential function influencing it.
    """
    # Animation stuff
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
        # global sim # Breaks the animation when used in a function
        sim.evolve()
        if psi == 'real':
            line.set_ydata(sim.realPsi())
        else:
            line.set_ydata(sim.normPsi())
        return line,

    ani = animation.FuncAnimation(fig, animate, frames=600, interval=10)

    if V != 'none':
        ax2 = ax1.twinx()
        ax2.plot(x, V(x), 'r')
        ax2.set_ylabel('$V(x)$')
        ax2.tick_params('y', colors='r')

    plt.show()

    if save:
        _ani_save(ani, sim)

    return ani


def _ani_save(ani, sim):
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
                            "D = " + str(sim.dim) + \
                            ", n = " + str(sim.numberPoints) + \
                            ", L = " + str(sim.domainLength) + \
                            ", dBC = " + dBC + \
                            ", dt = " + str(sim.dt) + '.mp4')
    ani.save(filepath, writer=writer)

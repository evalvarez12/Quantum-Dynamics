"""
simulation.py
Class that performs the evolution of the system.

created on: 24-04-2017.
@author: eduardo

"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import matrix


class Simulation:
    """
    This class handles the whole simulation of the quantum system.
    It takes a function for the potential and generates the discretised

    Hamiltonian. The evoluion of the system is made using Crank-Nicholson.
    """

    def __init__(self, dim, potentialFunc, dirichletBC, numberPoints,
                 startPoint, domainLength, dt):
        """Intilializes the object."""
        self.dim = dim
        self.numberPoints = numberPoints
        self.startPoint = startPoint
        self.domainLength = domainLength
        self.time = 0
        self.dt = dt

        self.sign = -1
        if dirichletBC:
            self.sign = 1

        self.allPoints = self.numberPoints + self.sign
        H = self._getHamiltonian(np.vectorize(potentialFunc))
        Id = sp.identity((self.numberPoints + self.sign)**self.dim)

        # Define the matrices used in CN evolution
        self.A = (Id + 1j*H*self.dt/2)
        self.B = (Id - 1j*H*self.dt/2)

        # Initialize wavefunction
        self.psi = np.zeros(self.allPoints**self.dim, dtype=np.complex128)
        self.pulse = np.zeros(self.allPoints**self.dim, dtype=np.complex128)

    def _getHamiltonian(self, potentialFunc):
        """
        Generate the Hamiltonian matrix using the functions
        defined in "matrix.py".
        """
        if self.dim == 1:
            if self.sign == -1:
                return matrix.A1D(self.numberPoints, potentialFunc,
                                  self.startPoint, self.domainLength)
            else:
                return matrix.A1Dfull(self.numberPoints, potentialFunc,
                                      self.startPoint, self.domainLength)

        if self.dim == 2:
            if self.sign == -1:
                return matrix.A2D(self.numberPoints, potentialFunc,
                                  self.startPoint, self.domainLength)
            else:
                return matrix.A2Dfull(self.numberPoints, potentialFunc,
                                      self.startPoint, self.domainLength)

    def setPsiPulse(self, pulse, energy, center, vel=1, width=.2):
        """
        Generate the initial wavefunction as a Gaussian wavepacket. By default
        it moves to the right.
        Inputs:
            pulse:  (string) plane wave or circular pulse
            energy: (int/float) The waves energy/size.
            center: (float or tuple) Either x or [x, y], for a guassian line
                    profile or 2D gaussian, respectively.
            vel:    (float or tuple) Velocity v_x or [v_x, v_y], the Velocity
                    of the pulse
            width:  Standard deviation of the Gaussian wave pulse.
        Output:
            Sets psi of the object to have the desired wave form.
        """
        if self.dim == 1:
            if pulse == "plane":
                x = self.domain()
                self.pulse = np.exp(1j * vel * np.sqrt(energy) * x) * \
                                    np.exp(-0.5 * (x-center)**2 / width**2)
                norm_Const = np.linalg.norm(self.pulse)
            else:
                self.pulse = np.zeros(self.allPoints)
                # Otherwise would divide by zero
                norm_Const = 1
        elif self.dim == 2:
            [x, y] = self.domain()

            if pulse == "plane":
                psix = np.exp(1j * vel * np.sqrt(energy) * x) * \
                              np.exp(-0.5 * (x-center)**2 / width**2)
                              
                y_const = np.ones(self.allPoints)
                self.pulse = np.kron(psix, y_const.flatten())
                norm_Const = np.linalg.norm(self.pulse)
            elif pulse == "circular":
                psix = np.exp(1j * vel[0] * np.sqrt(energy) * x) * \
                              np.exp(-0.5 * (x-center[0])**2 / width**2)
                psiy = np.exp(1j * vel[1] * np.sqrt(energy) * y) * \
                              np.exp(-0.5 * (y-center[1])**2 / width**2)
                              
                self.pulse = np.kron(psix, psiy.flatten())
                norm_Const = np.linalg.norm(self.pulse)
            else:
                self.pulse = np.zeros(self.allPoints**2)
                # Otherwise would divide by zero
                norm_Const = 1

        self.psi += self.pulse/norm_Const

    def normPsi(self):
        """Return the norm of the wave function."""
        return np.absolute(self.psi)**2

    def realPsi(self):
        """Return the real part of the wavefunction."""
        return np.real(self.psi)

    # Time evolutions
    def evolve(self):
        """Evolve the system using Crank-Nicholson."""
        self.time += self.dt
        self.psi = sp.linalg.spsolve(self.A, self.B.dot(self.psi),
                                     permc_spec='NATURAL')

    def evolvePulsed(self, freq):
        """Evolve the system using Crank-Nicholson."""
        self.time += 1
        if self.time % (freq) == 0:
            self.psi += self.pulse
            self.psi = self.psi/np.linalg.norm(self.psi)
        self.psi = sp.linalg.spsolve(self.A, self.B.dot(self.psi),
                                     permc_spec='NATURAL')

    def consistencyCheck(self):
        """Check if system is consistent by summing probabilities"""
        P = np.sum(self.normPsi())
        if abs(P-1) < .001:
            return True
        else:
            return False

    def domain(self):
        '''Generates evenly spaced vectors spanning the x and y domains'''
        if self.dim == 1:
            x = np.linspace(self.startPoint,
                            self.startPoint + self.domainLength,
                            self.allPoints)
            return x
        elif self.dim == 2:
            x = np.linspace(self.startPoint[0],
                            self.startPoint[0] + self.domainLength,
                            self.allPoints)
            y = np.linspace(self.startPoint[1],
                            self.startPoint[1] + self.domainLength,
                            self.allPoints).reshape(-1, 1)
            return [x, y]
        else:
            return 0

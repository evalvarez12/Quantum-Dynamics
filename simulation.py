"""
created on: 24-04-2017.
@author: eduardo

Class that performs the evolution of the system
"""
import matrix

import numpy as np
import scipy.signal as sg
import scipy.sparse as sp


class Simulation:
    '''
    This class handles the whole simulation of the quantum system.
    It takes a function for the potential and generates the discretised
    Hamiltonian. The evolution of the system is made using Crank-Nicholson.
    '''

    def __init__(self, dim, potentialFunc, dirichletBC, numberPoints, 
                 startPoint, domainLength, dt):
        self.dim = dim
        self.numberPoints = numberPoints
        self.startPoint = startPoint
        self.domainLength = domainLength
        self.dt = dt

        self.sign = -1
        if dirichletBC:
            self.sign = 1

        H = self._getHamiltonian(np.vectorize(potentialFunc))
        Id = sp.identity(self.numberPoints + self.sign)

        # Define the matrices used in CN evolution
        self.A = (Id - 1j*H*self.dt/2)
        self.B = (Id + 1j*H*self.dt/2)

    def _getHamiltonian(self, potentialFunc):
        '''
        Generates the Hamiltonian matrix using the functions
        defined in "matrix.py"
        '''
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

    def setPsiPulse(self):
        '''
        Generates the initial wavefunction as a Gaussian wavepacket
        TODO : Fix to make the initial wave paket to start going
        to the right
        '''
        if self.dim == 1:
            # x = np.linspace(self.startPoint -5, 
                    #self.domainLength -5, self.numberPoints + self.sign)
            # pulse = sg.gausspulse(x, fc=5)
            # self.psi = pulse
            # x = np.linspace(self.startPoint,
                    # self.startPoint + self.domainLength*2/self.domainLength,
                    # int(self.numberPoints*2/self.domainLength))
            # pulse = sg.gausspulse(x, fc=50)
            pulse = sg.gaussian(int(self.numberPoints*2/self.domainLength), 
                                std=4)
            self.psi = np.concatenate((pulse,
                    np.zeros(self.numberPoints - len(pulse) + self.sign)))

    def normPsi(self):
        '''Returns the norm of the wave function'''
        return np.absolute(self.psi)**2
    
#%% Time evolutions
    def evolve(self):
        '''Evolves the system using Crank-Nicholson'''
        self.psi = sp.linalg.spsolve(self.A, self.B.dot(self.psi),
                                     permc_spec='NATURAL')
        return np.absolute(self.psi)**2

    def evolve_cgs(self):
        '''Evolves the system using Conjugate Gradient squared iteration'''
        self.psi = sp.linalg.cgs(self.A, self.B.dot(self.psi),
                                     x0=self.psi)[0]
        return np.absolute(self.psi)**2
    
    def evolve_bicgstab(self):
        '''
        Evolves the system using BIConjugate Gradient STABilized iteration
        '''
        self.psi = sp.linalg.bicgstab(self.A, self.B.dot(self.psi),
                                     x0=self.psi)[0]
        return np.absolute(self.psi)**2
    
    def evolve_gmres(self):
        '''Evolves the system using Generalized Minimal RESidual iteration'''
        self.psi = sp.linalg.gmres(self.A, self.B.dot(self.psi),
                                     x0=self.psi)[0]
        return np.absolute(self.psi)**2
    
    def evolve_lgmres(self):
        '''Evolves the system using the LGMRES algorithm'''
        self.psi = sp.linalg.lgmres(self.A, self.B.dot(self.psi),
                                     x0=self.psi)[0]
        return np.absolute(self.psi)**2
    
    def evolve_qmr(self):
        '''Evolves the system using Quasi-Minimal Residual iteration'''
        self.psi = sp.linalg.qmr(self.A, self.B.dot(self.psi),
                                     x0=self.psi)[0]
        return np.absolute(self.psi)**2

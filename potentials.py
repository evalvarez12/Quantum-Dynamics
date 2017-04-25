# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 01:26:02 2017

@author: Eoin

Contains various functions which return useful potential functions.
"""
import numpy as np

def well_1D(mag, domain):
    '''
    A simple 1D potential well
    Inputs:
        mag: (float) Magnitude, the height or depth of the well.
        domain: (tuple) the location of the start and endpoints.
    Outputs:
        A vectorized potential punction, which accepts an Nx1 float array.
    '''
    
    def potentialWell(x):
        if domain[0] < x < domain[1]:
            return mag
        else:
            return 0
        
    return np.vectorize(potentialWell)

def well_2D(mag, xdomain, ydomain):
    '''
    A simple 2D potential well
    Inputs:
        mag: (float) Magnitude, the height or depth of the well.
        domain: (tuple) the location of the start and endpoints.
    Outputs:
        A vectorized potential punction, which accepts 2 Nx1 float arrays.
    '''
    
    def potentialWell(x, y):
        v = np.zeros((len(x),len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                if xdomain[0] < x[i] < xdomain[1] and \
                   ydomain[0] < y[j] < ydomain[1]:
                    v[i,j] = mag  
        return v
        
    return potentialWell

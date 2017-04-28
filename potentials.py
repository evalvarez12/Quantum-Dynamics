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
        A vectorized potential punction, which accepts an Nx1 float array and
            returns an Nx1 potential array.
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
        xdomain: (tuple) the location of the start and endpoints in x.
        ydomain: (tuple) the location of the start and endpoints in y.
    Outputs:
        A vectorized potential punction, which accepts 2 Nx1 float arrays and
            returns an NxN potential array.
    '''

    def potentialWell(x, y):
        v = np.zeros((len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                if xdomain[0] < x[i] < xdomain[1] and \
                  ydomain[0] < y[j] < ydomain[1]:
                    v[i, j] = mag
        return v

    return potentialWell


def double_slit(mag, xdomain, width, seperation):
    '''
    A 2D potential well to simulate the double slit experiment.
    Inputs:
        mag: (float) Magnitude, the height or depth of the well.
        xdomain: (tuple) the location of the start and endpoints of the wall.
        width: (float) How wide the slits should be.
        seperation: (float) The distance between the centers of the slits.
    Outputs:
        A vectorized potential punction, which accepts 2 Nx1 float arrays and
            returns an NxN potential array.
    '''

    def potentialWell(x, y):
        midway = y[-1]/2
        slit_centre = midway - seperation/2
        slits = np.array([slit_centre - width/2,
                          slit_centre + width/2,
                          y[-1] - slit_centre - width/2,
                          y[-1] - slit_centre + width/2])

        v = np.zeros((len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                if xdomain[0] < x[i] < xdomain[1]:
                    if 0 < y[j] < slits[0] or slits[1] < y[j] < slits[2] \
                      or slits[3] < y[j] < y[-1]:
                        v[i, j] = mag
        return v

    return potentialWell

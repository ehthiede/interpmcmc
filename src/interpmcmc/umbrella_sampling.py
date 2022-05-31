"""
Utility functions that are useful for umbrella sampling
"""

import numpy as np


def fk_from_stdev(stdev, kT=1.):
    """
    Calculates a force constant that corresponds to a gaussian distribution of
    standard deviation stdev. Convention is bias takes form 0.5 k (x - x_0)^2.

    Parameters
    ----------
    stdev : float
        Standad deviation of incoming gaussian
    kT : float, optional
        kT for the system

    Returns
    -------
    fk : float
        Force constant for the corresponding harmonic restraint.
    """
    return kT / (stdev**2)


def construct_centers_on_grid(centersx, centersy):
    """
    Constructs an array of window centers.

    Parameters
    ----------
    centersx : numpy array
        N_x x-coordinates in the grid
    centersy : numpy array
        N_y y-coordinates in the grid

    Returns
    -------
    centers : numpy array
        Array of the center for each window, of shape (N_x * N_y by 2).
    """
    X, Y = np.meshgrid(centersx, centersy)
    X_flat = X.ravel()
    Y_flat = Y.ravel()
    centers = np.vstack((X_flat, Y_flat)).T
    return centers

"""
Functions to perform Helmholtz decomposition for vector fields on a sphere
Resources
---------
Bessel functions in SciPy
    https://www.johndcook.com/blog/bessel_python/
Spherical Harmonic
    http://functions.wolfram.com/HypergeometricFunctions/SphericalHarmonicYGeneral/
Convention
----------
cartesian coordinates
    :(x, y, z):
spherical coordinates
    :(r, theta, phi): where
        :r:     radial coordinate; must be in (0, oo);
        :theta: polar coordinate; must be in [0, pi];
        :phi:   azimuthal coordinate; must be in [0, 2*pi];
"""

import numpy as np
from scipy.special import sph_harm


def Ylm(l, m, theta, phi): 
    """
    Redefine spherical harmonics from scipy.special
    to match physics convention.
    
    Parameters
    ----------
    l : int, array_like
        Degree of the harmonic (int); ``l >= 0``.
    m : int, array_like
        Order of the harmonic (int); ``|m| <= l``.
    theta : array_like
        Polar (colatitudinal) coordinate; must be in ``[0, pi]``.
    phi : array_like
        Azimuthal (longitudinal) coordinate; must be in ``[0, 2*pi]``.
    
    Returns
    -------
    Ylm : complex float
       The harmonic Ylm sampled at ``theta`` and ``phi``.
    """
    if np.abs(m) > l:
        Ylm = 0
    else:
        Ylm = sph_harm(m, l, phi, theta) # Perform redefinition, scipy version
    return Ylm
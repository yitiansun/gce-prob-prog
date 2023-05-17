import numpy as np
from reproject import reproject_from_healpix
from astropy.wcs import WCS
from astropy.coordinates import ICRS, Galactic


def make_wcs(center, size, pixelsize, frame="Galactic", projection="CAR"):
    """
    Create a WCS (World Coordinate System) object with the given parameters.
    
    Parameters
    ----------
    center : tuple of float
        The (x, y) coordinates of the image center in the given frame.
    size : tuple of int
        The (x, y) size of the image in pixels.
    pixelsize : float
        The size of a pixel in degrees.
    frame : str, optional, default: "Galactic"
        The coordinate frame to use, either "ICRS" or "Galactic".
    projection : str, optional, default: "CAR"
        The projection to use, either "TAN" (gnomonic) or "CAR" (plate carrée).

    Returns
    -------
    w : WCS object
        The created WCS object with the given parameters.

    Raises
    ------
    ValueError
        If an unsupported projection or an unknown frame is provided.
    """
    xcenter, ycenter = center 
    xsize, ysize = size
    if projection.upper() not in ["TAN", "CAR"]:
        raise ValueError("unsupported projection: " % projection)
    if frame.upper() == "ICRS":
        ctype = ["RA---" + projection.upper(), "DEC--" + projection.upper()]
    elif frame.upper() == "GALACTIC":
        ctype = ["GLON-" + projection.upper(), "GLAT-" + projection.upper()]
    else:
        raise ValueError("unknown frame: " % frame)

    w = WCS(naxis=2)
    w.wcs.ctype = ctype
    w.wcs.crval = np.array([xcenter, ycenter])
    w.wcs.crpix = np.array([xsize / 2.0 + 0.5, ysize / 2.0 + 0.5])
    w.wcs.cdelt = np.array([-pixelsize, pixelsize])
    w.wcs.cunit = ["deg", "deg"]
    w.wcs.equinox = 2000.0
    return w

def to_cart(temp_hp, n_pixels=80, pixelsize=0.5, frame="Galactic"):
    """
    Convert a HEALPix map to a Cartesian projection using the given parameters.
    
    Parameters
    ----------
    temp_hp : array-like
        The input HEALPix data to be reprojected.
    n_pixels : int, optional, default: 96
        The size of the output image in pixels. The image will be square with dimensions (n_pixels, n_pixels).
    pixelsize : float, optional, default: 0.5
        The size of a pixel in degrees.
    frame : str, optional, default: "Galactic"
        The coordinate frame to use, either "ICRS" or "Galactic".

    Returns
    -------
    2D array
        The reprojected image in Cartesian coordinates with dimensions (n_pixels, n_pixels).
    """
    wcs = make_wcs(center=(0.,0.), size=(n_pixels,n_pixels), pixelsize=pixelsize, frame=frame)
    return reproject_from_healpix((temp_hp, "Galactic"), wcs, shape_out=(n_pixels, n_pixels), nested=False)[0]
"""Rigid templates"""

import sys
sys.path.append("..")

import numpy as np
import healpy as hp
from astropy.io import fits
from reproject import reproject_to_healpix

import jax.numpy as jnp

from utils import create_mask as cm
from utils.cart import make_wcs


def interp1d(x, xp, fp):
    """Linear 1D interpolation along the first dimension. See jnp.interp.
    xp must be in increasing order."""
    ind = jnp.interp(x, xp, jnp.arange(len(xp)))
    ind_left = int(jnp.floor(ind))
    ind_right = int(jnp.ceil(ind))
    return fp[ind_left] * (ind_right - ind) + fp[ind_right] * (ind - ind_left)


#========== General templates ==========

class Template:
    """Energy independent template.
    
    Parameters
    ----------
    data : ndarray
    norm_mask : None or ndarray, same shape as data
        Mask used to normalize template. Not energy dependent. Not stored.
        1 or True means masking out, 0 or False means to include in fit.
    """
    
    def __init__(self, data, norm_mask=None):
        self.data = data
        if norm_mask is not None:
            self.data /= jnp.mean(self.data[~norm_mask])
        
    def at_bin(self, ie, mask=None):
        """Returns energy independent template."""
        return self.data if mask is None else self.data[~mask]


class EbinTemplate:
    """Templates with energy binning.
    
    Parameters
    ----------
    engs : ndarray, shape-(nebin,)
        Energy abscissa. Required for at_eng.
    data : ndarray, shape=(nebin, ...)
    norm_mask : None or ndarray, shape=(...)
        Mask used to normalize template. Not energy dependent. Not stored.
        1 or True means masking out, 0 or False means to include in fit.
    """
    
    def __init__(self, data, engs=None, norm_mask=None):
        self.data = data
        self.engs = engs
        if norm_mask is not None:
            self.data /= jnp.mean(self.data[:, ~norm_mask], axis=1)[:, None]
                
    def at_bin(self, ie, mask=None):
        """Returns template at ith E bin."""
        return self.data[ie] if mask is None else self.data[ie][~mask]
    
    def at_eng(self, eng, mask=None):
        """Returns interpolated template at energy."""
        interp_temp = interp1d(eng, self.engs, self.data)
        return interp_temp if mask is None else interp_temp[~mask]
    

#========== Bulge templates ==========

class BulgeTemplates:
    """Bulge templates from literature.
    The "mcdermott*" templates are from McDermott et al. 2022 (https://arxiv.org/abs/),
    downloaded from https://github.com/samueldmcdermott/gcepy/tree/main/gcepy/inputs/excesses.
    Other templates are downloaded from https://github.com/chrisgordon1/galactic_bulge_templates.
    
    Parameters
    ----------
    template_name : str
    nside_project : int
        HEALPix nside to project cartesian input maps to.
    nside_out : int
        HEALPix nside to downgrade output maps to.
    r_norm : float
        Normalization ROI radius in degrees.
    """
    
    def __init__(self, template_name="macias2019", nside_project=512, nside_out=128, r_norm=30.0):

        self.nside_out = nside_out
        self.nside_project = nside_project

        self.mask_norm = cm.make_mask_total(nside=nside_out, mask_ring=True, inner=0, outer=r_norm)

        bulge_data_dir = "../data/bulge_templates/"

        # From https://github.com/chrisgordon1/galactic_bulge_templates
        if template_name == "macias2019":
            self.template = fits.open(bulge_data_dir + "BoxyBulge_arxiv1901.03822_Normalized.fits")[0].data
            self.wcs = make_wcs([0, 0], [200, 200], 0.2)
        elif template_name == "coleman2019":
            self.template = fits.open(bulge_data_dir + "Bulge_modulated_Coleman_etal_2019_Normalized.fits")[0].data
            self.wcs = make_wcs([0, 0], [200, 200], 0.2)

        # From https://github.com/samueldmcdermott/gcepy/tree/main/gcepy/inputs/excesses
        elif template_name == "mcdermott2022":
            self.template = np.flip(np.load(bulge_data_dir + "bb_front_only_14_Ebin_20x20window_normal.npy")[0], -1)
            self.wcs = make_wcs([0, 0], [400, 400], 0.1)
        elif template_name == "mcdermott2022_bbp":
            self.template = np.flip(np.load(bulge_data_dir + "bbp_front_only_14_Ebin_20x20window_normal.npy")[0], -1)
            self.wcs = make_wcs([0, 0], [400, 400], 0.1)
        elif template_name == "mcdermott2022_x":
            self.template = np.flip(np.load(bulge_data_dir + "x_front_only_14_Ebin_20x20window_normal.npy")[0], -1)
            self.wcs = make_wcs([0, 0], [400, 400], 0.1)

    def __call__(self):
        template_hp, _ = np.nan_to_num(reproject_to_healpix((self.template, self.wcs), "galactic", nside=self.nside_project))
        template_hp = hp.ud_grade(template_hp, nside_out=self.nside_out)
        return template_hp / np.nan_to_num(template_hp[~self.mask_norm]).mean()
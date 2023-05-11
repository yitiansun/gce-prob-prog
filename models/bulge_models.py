import healpy as hp
import numpy as np
from reproject import reproject_to_healpix
from astropy.io import fits

from utils import create_mask as cm
from utils.cart import make_wcs


class BulgeTemplates:
    def __init__(self, template_name="macias2019", nside_project=512, nside_out=128, r_norm=30.0):
        """Load bulge templates from the literature, The "mcdermott*" templates are from McDermott et al. 2022 (https://arxiv.org/abs/2209.00006),
           downloaded from https://github.com/samueldmcdermott/gcepy/tree/main/gcepy/inputs/excesses. The other templates are downloaded from
           https://github.com/chrisgordon1/galactic_bulge_templates.

        Args:
            template_name (str, optional): Identifying string. Defaults to "macias2019".
            nside_project (int, optional): HEALPix nside to project cartesian input maps to. Defaults to 512.
            nside_out (int, optional): HEALPix nside to downgrade output maps to. Defaults to 128.
            r_norm (float, optional): Normalization ROI. Defaults to 30.0.
        """

        self.nside_out = nside_out
        self.nside_project = nside_project

        self.mask_norm = cm.make_mask_total(nside=nside_out, mask_ring=True, inner=0, outer=r_norm)

        # From https://github.com/chrisgordon1/galactic_bulge_templates
        if template_name == "macias2019":
            self.template = fits.open("../data/BoxyBulge_arxiv1901.03822_Normalized.fits")[0].data
            self.wcs = make_wcs([0, 0], [200, 200], 0.2)
        elif template_name == "coleman2019":
            self.template = fits.open("../data/Bulge_modulated_Coleman_etal_2019_Normalized.fits")[0].data
            self.wcs = make_wcs([0, 0], [200, 200], 0.2)

        # From https://github.com/samueldmcdermott/gcepy/tree/main/gcepy/inputs/excesses
        elif template_name == "mcdermott2022":
            self.template = np.flip(np.load("../data/external/bb_front_only_14_Ebin_20x20window_normal.npy")[0], -1)
            self.wcs = make_wcs([0, 0], [400, 400], 0.1)
        elif template_name == "mcdermott2022_bbp":
            self.template = np.flip(np.load("../data/external/bbp_front_only_14_Ebin_20x20window_normal.npy")[0], -1)
            self.wcs = make_wcs([0, 0], [400, 400], 0.1)
        elif template_name == "mcdermott2022_x":
            self.template = np.flip(np.load("../data/external/x_front_only_14_Ebin_20x20window_normal.npy")[0], -1)
            self.wcs = make_wcs([0, 0], [400, 400], 0.1)

    def __call__(self):
        template_hp, _ = np.nan_to_num(reproject_to_healpix((self.template, self.wcs), "galactic", nside=self.nside_project))
        template_hp = hp.ud_grade(template_hp, nside_out=self.nside_out)
        return template_hp / np.nan_to_num(template_hp[~self.mask_norm]).mean()

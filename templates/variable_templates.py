"""Variable templates"""

import sys
sys.path.append("..")

import numpy as np
import healpy as hp

import jax.numpy as jnp
from jax.scipy.special import gammaln
from jax import vmap, jit
from functools import partial

from utils import create_mask as cm


class NFWTemplate:
    def __init__(self, nside=128, outer=30.0, n_integ=500, rsun=8.224, r_s=20.0, r_norm=30.0):
        """Generate HEALPix map of NFW template

        Args:
            nside (int, optional): HEALPix nside of map. Defaults to 128.
            outer (float, optional): Maximum outer radius (smaller = generation). Defaults to 300
            n_integ (int, optional): Number of integration steps in line-of-sight integral. Defaults to 500.
            rsun (float, optional): Solar radius. Defaults to 8.224.
            r_s (float, optional): Scale radius of NFW profile. Defaults to 20.0.
            r_norm (float, optional): Normalization ROI. Defaults to 30.0.
        """

        mask = cm.make_mask_total(nside=nside, mask_ring=True, inner=0, outer=outer)
        mask_restrict = np.where(mask == 0)[0]
        theta_ary, phi_ary = hp.pix2ang(nside, mask_restrict)

        self.mask_norm = cm.make_mask_total(nside=nside, mask_ring=True, inner=0, outer=r_norm)

        self.npix = hp.nside2npix(nside)
        self.mask_idx = jnp.arange(self.npix)[~mask]
        self.b_ary = np.pi / 2.0 - theta_ary
        self.l_ary = NFWTemplate.mod(phi_ary + np.pi, 2.0 * np.pi) - np.pi
        self.s_ary = jnp.linspace(0, 40, n_integ)

        self.rsun = rsun
        self.r_s = r_s
        
    def set_mask(self, mask):
        """Set fit mask. Required before calling get_NFW2_masked_template."""
        if not self.npix == len(mask):
            raise ValueError('mask nside incorrect.')
        mask_restrict = np.where(mask == 0)[0]
        theta_ary, phi_ary = hp.pix2ang(hp.npix2nside(self.npix), mask_restrict)
        self.masked_b_ary = np.pi / 2.0 - theta_ary
        self.masked_l_ary = NFWTemplate.mod(phi_ary + np.pi, 2.0 * np.pi) - np.pi
        
    @partial(jit, static_argnums=(0,))
    def get_NFW2_masked_template(self, gamma=1.2):
        """Return LOS integral of density^2 only on masked indices"""
        int_rho2 = jnp.trapz(self.rho_NFW(self.rGC(self.s_ary, self.masked_b_ary, self.masked_l_ary, self.rsun), gamma=gamma, r_s=self.r_s) ** 2, self.s_ary, axis=1)
        return int_rho2 / jnp.nan_to_num(int_rho2).mean()

    @partial(jit, static_argnums=(0,))
    def get_NFW2_template(self, gamma=1.2):
        """Return LOS integral of density^2"""

        # LOS integral of density^2
        int_rho2_temp = jnp.trapz(self.rho_NFW(self.rGC(self.s_ary, self.b_ary, self.l_ary, self.rsun), gamma=gamma, r_s=self.r_s) ** 2, self.s_ary, axis=1)

        int_rho2 = jnp.zeros(self.npix)
        int_rho2 = int_rho2.at[self.mask_idx].set(int_rho2_temp)

        return int_rho2 / jnp.nan_to_num(int_rho2[~self.mask_norm]).mean()

    @staticmethod
    def mod(dividends, divisor):
        """Return dividends (array) mod divisor (double)
        Adapted from Nick's code
        """

        output = np.zeros(len(dividends))

        for i in range(len(dividends)):
            output[i] = dividends[i]
            done = False
            while not done:
                if output[i] >= divisor:
                    output[i] -= divisor
                elif output[i] < 0.0:
                    output[i] += divisor
                else:
                    done = True

        return output

    def rho_NFW(self, r, gamma=1.0, r_s=20.0):
        """Generalized NFW profile"""
        return (r / r_s) ** -gamma * (1 + (r / r_s)) ** (-3 + gamma)

    def rGC(self, s_ary, b_ary, l_ary, rsun=8.224):
        """Distance to GC as a function of LOS distance, latitude, longitude"""
        return jnp.sqrt(s_ary**2 - 2.0 * rsun * jnp.transpose(jnp.outer(s_ary, jnp.cos(b_ary) * jnp.cos(l_ary))) + rsun**2)


class LorimerDiskTemplate:
    def __init__(self, nside=128, outer=40, n_integ=2000, rsun=8.224, r_norm=30.0):
        """Lorimer disk spatial template in HEALPix projection

        Args:
            nside (int, optional): HEALPix nside. Defaults to 128.
            outer (float, optional): Maximum outer radius to use. Smaller = faster generation. Defaults to 40.
            n_integ (int, optional): Number of steps in line of sight integral. Defaults to 2000.
            rsun (float, optional): Solar radius. Defaults to 8.224.
            r_norm (float, optional): Radius to normalize template to. Defaults to 30.0.
        """

        mask = cm.make_mask_total(nside=nside, mask_ring=True, inner=0, outer=outer)
        mask_restrict = np.where(mask == 0)[0]
        theta_ary, phi_ary = hp.pix2ang(nside, mask_restrict)

        self.mask_norm = cm.make_mask_total(nside=nside, mask_ring=True, inner=0, outer=r_norm)

        self.npix = hp.nside2npix(nside)
        self.mask_idx = jnp.arange(self.npix)[~mask]
        self.b_ary = np.pi / 2.0 - theta_ary
        self.l_ary = NFWTemplate.mod(phi_ary + np.pi, 2.0 * np.pi) - np.pi
        self.s_ary = jnp.linspace(0, 100, n_integ)

        self.rsun = rsun

        self.L_integ_Lorimer_vmap = jit(vmap(self.L_integ_Lorimer, in_axes=(0, 0, None, None, None, None)))

    @partial(jit, static_argnums=(0,))
    def get_template(self, zs=0.63, B=0.0, C=5.94):
        int_rho_temp = self.L_integ_Lorimer_vmap(self.b_ary, self.l_ary, zs, B, C, self.rsun)
        int_rho = jnp.zeros(self.npix)
        int_rho = int_rho.at[self.mask_idx].set(int_rho_temp)
        return int_rho / jnp.nan_to_num(int_rho[~self.mask_norm]).mean()

    def R_z_GC(self, s, b, l, rsun=8.224):
        """Convert lon/lat to cylindrical coordinates

        :param s: distance from Earth [kpc]
        :param b: latitude in galactic coordinates [rad]
        :param l: longitude in galactic coordinates [rad]
        :returns: distance from GC [kpc]
        """
        R = jnp.sqrt(s**2 - 2 * rsun * s * jnp.cos(l) + rsun**2)
        z = s * jnp.tan(b)
        return R, z

    def rho_V_Lorimer(self, R, z, zs=0.63, B=2.75, C=5.94, rsun=8.224):
        """Spatial number density according to Lorimer disk profile (unnormalized)
        Eq. (6) of Bartels et al (1805.11097), after removing constant terms
        """
        pref = C ** (B + 2) / (4 * np.pi * rsun**2 * zs * jnp.exp(C) * jnp.exp(gammaln(B + 2)))
        return pref * (R / rsun) ** B * jnp.exp(-C * ((R - rsun) / rsun)) * jnp.exp(-jnp.abs(z) / zs)

    def rho_V_Lorimer_lonlat(self, s, b, l, zs, B, C, rsun):
        """Lorimer density, this time in lot/lat"""

        R, z = self.R_z_GC(s, b, l)
        return self.rho_V_Lorimer(R, z, zs, B, C, rsun)

    def L_integ_Lorimer(self, b, l, zs=0.63, B=0.0, C=5.94, rsun=8.224):
        """Line-of-sight integral (discrete sum) for Lorimer disk profile"""
        return jnp.trapz(self.rho_V_Lorimer_lonlat(self.s_ary, b, l, zs, B, C, rsun), self.s_ary)
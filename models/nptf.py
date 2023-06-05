"""Non-poissonian model"""

import numpy as np
import healpy as hp

import jax
import jax.numpy as jnp
from jax.example_libraries import stax

import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, autoguide
from numpyro import optim
from numpyro import handlers
from tensorflow_probability.substrates import jax as tfp

import optax
import arviz as az
from einops import repeat

from likelihoods.npll_jax import log_like_np, dnds
from templates.variable_templates import NFWTemplate, LorimerDiskTemplate
from templates.rigid_templates import BulgeTemplates
from utils import create_mask as cm
from utils.psf import KingPSF
from utils.psf_correction import PSFCorrection

import logging

data_dir = "../data"


class NPModel:
    """
    Parameters
    ----------
    ...
    non_poissonian : bool
        Whether to use non-poissonian template fitting.
    bulge_hybrid : bool
        If False, use the first template in bulge_template_names. If True, use a
        hybrid of the templates in bulge_template_names.
    ps_cat : {'3fgl', '4fgl'}
        Point source catalog to use for masks.
    band_mask_range : float
        |b| value [deg] below which the galactic plane is masked. Affects
        self.mask_roi .

    Attributes
    ----------
    ...
    normalization_mask: mask used to normalize templates.
    """

    def __init__(
        self,
        dif_names=["ModelO", "ModelA", "ModelF"],
        bulge_template_names=["mcdermott2022", "mcdermott2022_bbp", "mcdermott2022_x", "macias2019", "coleman2019"],
        ps_cat="3fgl",
        r_outer=25,
        band_mask_range=2.0,
        nside=128,
        n_exp=1,
    ):
        # ========== General ==========
        self.nside = nside
        self.ps_cat = ps_cat

        self.data_dir = f"{data_dir}/fermi_data_573w/fermi_data_{self.nside}"
        self.data = jnp.array(np.load("{}/fermidata_counts.npy".format(self.data_dir)).astype(np.int32))
        self.exposure_map = np.load("{}/fermidata_exposure.npy".format(self.data_dir))

        # ========== Mask ==========
        if ps_cat == "3fgl":
            # mask_ps = np.load("{}/fermidata_pscmask_{}.npy".format(self.data_dir, self.ps_cat)) == 1
            mask_ps = hp.ud_grade(np.load(f"{data_dir}/mask_3fgl_0p8deg.npy"), nside_out=self.nside) > 0
        elif ps_cat == "4fgl":
            logging.warning("Using 4fgl with non-poissonian fit.")
            mask_ps = hp.ud_grade(np.load(f"{data_dir}/fermi_data_573w/fermi_data_{nside}/fermidata_pscmask_4fgl.npy"), nside_out=self.nside) > 0
        else:
            raise NotImplementedError("Other catalogs not supported at the moment.")

        self.mask_roi = cm.make_mask_total(nside=self.nside, band_mask=True, band_mask_range=band_mask_range, mask_ring=True, inner=0, outer=r_outer, custom_mask=mask_ps)
        self.mask_plane = cm.make_mask_total(nside=self.nside, band_mask=True, band_mask_range=2.0, mask_ring=True, inner=0, outer=25)
        self.normalization_mask = self.mask_plane

        # ========== Templates ==========
        self.nfw_template = NFWTemplate(nside=self.nside)
        self.disk_template = LorimerDiskTemplate(nside=self.nside)

        self.dif_names = dif_names

        # Load all bulge templates
        self.bulge_templates = jnp.array([BulgeTemplates(template_name=template_name, nside_out=nside)() for template_name in bulge_template_names])
        self.n_bulge_templates = len(self.bulge_templates)

        # Individually normalize the bulge templates
        self.bulge_templates = self.bulge_templates / jnp.mean(self.bulge_templates[:, ~self.normalization_mask], axis=-1)[:, None]

        self.load_templates()

        # ========== NPTF ==========
        self.get_psf_correction()
        self.k_max = np.max(np.array(self.data)[~self.mask_roi])
        print("Max photon count is {}".format(self.k_max))

        self.get_exp_regions(n_exp)

    def get_psf_correction(self):
        kp = KingPSF()

        pc_inst = PSFCorrection(delay_compute=True, num_f_bins=15, nside=self.nside)
        pc_inst.psf_r_func = lambda r: kp.psf_fermi_r(r)
        pc_inst.sample_psf_max = 10.0 * kp.spe * (kp.score + kp.stail) / 2.0
        pc_inst.psf_samples = 10000
        pc_inst.psf_tag = "Fermi_PSF_2GeV2_nside{}".format(self.nside)
        pc_inst.make_or_load_psf_corr()

        self.f_ary = pc_inst.f_ary
        self.df_rho_div_f_ary = pc_inst.df_rho_div_f_ary

    def load_templates(self):
        self.temp_psc = np.load("{}/template_psc_{}.npy".format(self.data_dir, self.ps_cat))
        self.temp_iso = np.load("{}/template_iso.npy".format(self.data_dir))
        self.temp_bub = np.load("{}/template_bub.npy".format(self.data_dir))
        self.temp_dsk = np.load("{}/template_dsk_z0p3.npy".format(self.data_dir))

        # Load Model O templates
        self.temp_mO_pibrem = np.load("{}/template_Opi.npy".format(self.data_dir))
        self.temp_mO_ics = np.load("{}/template_Oic.npy".format(self.data_dir))

        # Load Model A templates
        self.temp_mA_pibrem = np.load("{}/template_Api.npy".format(self.data_dir))
        self.temp_mA_ics = np.load("{}/template_Aic.npy".format(self.data_dir))

        # Load Model F templates
        self.temp_mF_pibrem = np.load("{}/template_Fpi.npy".format(self.data_dir))
        self.temp_mF_ics = np.load("{}/template_Fic.npy".format(self.data_dir))

        self.pibrem = []
        self.ics = []

        if "ModelO" in self.dif_names:
            self.pibrem.append(self.temp_mO_pibrem)
            self.ics.append(self.temp_mO_ics)
        if "ModelA" in self.dif_names:
            self.pibrem.append(self.temp_mA_pibrem)
            self.ics.append(self.temp_mA_ics)
        if "ModelF" in self.dif_names:
            self.pibrem.append(self.temp_mF_pibrem)
            self.ics.append(self.temp_mF_ics)

        self.pibrem = jnp.array(self.pibrem)
        self.ics = jnp.array(self.ics)

        self.n_dif_templates = len(self.pibrem)

        self.svi = None
        self.svi_init_state = None

    def model(self, data):
        # Get mixed pibrem template
        theta_pibrem = numpyro.sample("theta_pibrem", dist.Dirichlet(jnp.ones((self.n_dif_templates,)) / self.n_dif_templates))
        temp_pibrem = jnp.sum(theta_pibrem[:, None] * self.pibrem, 0)

        # Get mixed ics template
        theta_ics = numpyro.sample("theta_ics", dist.Dirichlet(jnp.ones((self.n_dif_templates,)) / self.n_dif_templates))
        temp_ics = jnp.sum(theta_ics[:, None] * self.ics, 0)

        S_gce = numpyro.sample("S_gce", dist.Uniform(1e-5, 4.0))

        temps = [self.temp_iso, self.temp_bub, self.temp_psc, temp_pibrem, temp_ics]
        temp_labels = ["iso", "bub", "psc", "dif", "ics"]

        mu = jnp.zeros_like(data)

        for temp, temp_label in zip(temps, temp_labels):
            if temp_label in ["dif", "ics"]:
                prior_lo, prior_hi = 1e-3, 14.0
            else:
                prior_lo, prior_hi = 1e-3, 5.0

            prior_dist = dist.Uniform(prior_lo, prior_hi)
            S_temp = numpyro.sample("S_{}".format(temp_label), prior_dist)

            A_temp = S_temp / jnp.mean(temp[~self.normalization_mask])
            mu += A_temp * temp

        gamma_ps = numpyro.sample("gamma_ps", dist.Uniform(0.2, 2.0))
        gamma_poiss = numpyro.sample("gamma_poiss", dist.Uniform(0.2, 2.0))

        temp_gce_nfw_ps = self.nfw_template.get_NFW2_template(gamma=gamma_ps)
        temp_gce_nfw_poiss = self.nfw_template.get_NFW2_template(gamma=gamma_poiss)

        zs = numpyro.sample("zs", dist.Uniform(0.1, 2.5))
        C = numpyro.sample("C", dist.Uniform(0.05, 15.0))
        temp_dsk = self.disk_template.get_template(zs=zs, C=C)

        f_bulge_ps = numpyro.sample("f_bulge_ps", dist.Uniform(0.0, 1.0))
        f_bulge_poiss = numpyro.sample("f_bulge_poiss", dist.Uniform(0.0, 1.0))

        theta_bulge_poiss = numpyro.sample("theta_bulge_poiss", dist.Dirichlet(jnp.ones((self.n_bulge_templates,)) / self.n_bulge_templates))
        temp_bulge = jnp.sum(theta_bulge_poiss[:, None] * self.bulge_templates, 0)

        # Normalize to same mean
        A_gce_nfw = S_gce / jnp.mean(temp_gce_nfw_poiss[~self.normalization_mask])
        A_gce_bulge = S_gce / jnp.mean(temp_bulge[~self.normalization_mask])
        temp_gce_poiss = (1 - f_bulge_poiss) * A_gce_nfw * temp_gce_nfw_poiss + f_bulge_poiss * A_gce_bulge * temp_bulge

        A_gce = S_gce / jnp.mean(temp_gce_poiss[~self.normalization_mask])
        mu += A_gce * temp_gce_poiss

        # Get mixed bulge template
        theta_bulge_ps = numpyro.sample("theta_bulge_ps", dist.Dirichlet(jnp.ones((self.n_bulge_templates,)) / self.n_bulge_templates))
        temp_bulge = jnp.sum(theta_bulge_ps[:, None] * self.bulge_templates, 0)

        # Normalize to same mean
        A_gce_nfw = 1 / jnp.mean(temp_gce_nfw_ps[~self.normalization_mask])
        A_gce_bulge = 1 / jnp.mean(temp_bulge[~self.normalization_mask])

        # Get hybrid template
        temp_gce_ps = (1 - f_bulge_ps) * A_gce_nfw * temp_gce_nfw_ps + f_bulge_ps * A_gce_bulge * temp_bulge

        npt_compressed = jnp.array([temp_gce_ps, temp_dsk])

        theta = []

        for ips, ps in enumerate(["gce", "dsk"]):
            Sps = numpyro.sample("Sps_{}".format(ps), dist.Uniform(1e-5, 2.0))

            n1 = numpyro.sample("n1_{}".format(ps), dist.Uniform(4.0, 6.0))
            n2 = numpyro.sample("n2_{}".format(ps), dist.Uniform(0.5, 1.99))
            n3 = numpyro.sample("n3_{}".format(ps), dist.Uniform(-6.0, -5.0))
            sb1 = numpyro.sample("sb1_{}".format(ps), dist.Uniform(5.0, 40.0))
            lambda_s = numpyro.sample("lambdas_{}".format(ps), dist.Uniform(0.1, 0.95))

            theta_tmp = jnp.array([1.0, n1, n2, n3, sb1, lambda_s * sb1])

            s_ary = jnp.logspace(0.0, 2, 100)
            dnds_ary = dnds(s_ary, theta_tmp)

            A = Sps / jnp.mean(npt_compressed[ips][~self.normalization_mask] * jnp.trapz(s_ary * dnds_ary, s_ary))

            theta.append([A, n1, n2, n3, sb1, lambda_s * sb1])

        theta = jnp.array(theta)

        # Pad the last exposure region so that all are the same size
        exp_lens = [len(self.expreg_indices[i]) for i in range(len(self.expreg_indices))]
        n_pad = exp_lens[0] - exp_lens[-1]

        expreg_indices = jnp.zeros_like(self.expreg_indices)
        expreg_indices = expreg_indices.at[:-1].set(self.expreg_indices[:-1])
        expreg_indices = expreg_indices.at[-1].set(jnp.pad(self.expreg_indices[-1], (0, n_pad)))

        log_like_np_exp_vmapped = jax.vmap(log_like_np, in_axes=(0, 0, 1, 0, None, None, None, None))

        # Get relevant arrays for different exposure regions
        mu_batch = mu[~self.mask_roi][jnp.array(expreg_indices)]
        npt_compressed_batch = npt_compressed[:, ~self.mask_roi][:, jnp.array(expreg_indices)]
        data_batch = self.data[~self.mask_roi][jnp.array(expreg_indices)]

        exposure_multiplier = self.exposure_means_list / self.exposure_mean

        # Scale non-Poissonian parameters (norm divided by exposure ratio, breaks multiplied)
        theta = repeat(theta, "n_ps n_param -> n_exp n_ps n_param", n_exp=len(expreg_indices))
        theta = theta.at[:, :, 0].set(theta[:, :, 0] / exposure_multiplier[:, None])
        theta = theta.at[:, :, -1].set(theta[:, :, -1] * exposure_multiplier[:, None])
        theta = theta.at[:, :, -2].set(theta[:, :, -2] * exposure_multiplier[:, None])

        with numpyro.plate("data", size=len(mu[~self.mask_roi]), dim=-1):
            log_like_exp = log_like_np_exp_vmapped(theta, mu_batch, npt_compressed_batch, data_batch, self.f_ary, self.df_rho_div_f_ary, self.k_max, len(expreg_indices[0]))

            # Concatenate exposure regions
            loglike = jnp.concatenate(log_like_exp)[: len(mu[~self.mask_roi])]

            with handlers.mask(mask=~jnp.logical_or(jnp.isinf(loglike), jnp.isnan(loglike))):
                return numpyro.factor("log-likelihood", loglike)

    def get_exp_regions(self, nexp):
        """Divide up ROI into exposure regions"""

        # Determine the pixels of the exposure regions
        pix_array = np.where(self.mask_roi == False)[0]
        exp_array = np.array([[pix_array[i], self.exposure_map[pix_array[i]]] for i in range(len(pix_array))])
        array_sorted = exp_array[np.argsort(exp_array[:, 1])]

        # Convert from list of exreg pixels to masks (int as used to index)
        array_split = np.array_split(array_sorted, nexp)
        expreg_array = [np.array([array_split[i][j][0] for j in range(len(array_split[i]))], dtype="int32") for i in range(len(array_split))]

        npix = len(self.mask_roi)

        self.expreg_mask = []
        for i in range(nexp):
            temp_mask = np.logical_not(np.zeros(npix))
            for j in range(len(expreg_array[i])):
                temp_mask[expreg_array[i][j]] = False
            self.expreg_mask.append(temp_mask)

        # Store the total and region by region mean exposure
        expreg_values = [[array_split[i][j][1] for j in range(len(array_split[i]))] for i in range(len(array_split))]

        self.exposure_means_list = jnp.array([np.mean(expreg_values[i]) for i in range(nexp)])
        self.exposure_mean = jnp.mean(self.exposure_means_list)

        self.expreg_indices = []
        for i in range(nexp):
            expreg_indices_temp = np.array([np.where(pix_array == elem)[0][0] for elem in expreg_array[i]])
            self.expreg_indices.append(jnp.array(expreg_indices_temp))

        self.expreg_indices = jnp.array(self.expreg_indices)

    def fit_svi(self, rng_key=jax.random.PRNGKey(1), n_steps=5000, lr=5e-3, num_particles=2, guide="mvn", num_flows=4, hidden_dims=[64, 64]):
        if guide == "iaf":
            self.guide = autoguide.AutoIAFNormal(self.model, num_flows=num_flows, hidden_dims=hidden_dims, nonlinearity=stax.Tanh)
        elif guide == "mvn":
            self.guide = autoguide.AutoMultivariateNormal(self.model)
        else:
            raise NotImplementedError

        optimizer = optim.optax_to_numpyro(optax.chain(optax.clip(1.0), optax.adam(lr)))

        self.svi = SVI(self.model, self.guide, optimizer, Trace_ELBO(num_particles=num_particles))
        self.svi_results = self.svi.run(rng_key, n_steps, self.data)

        return self.svi_results

    def get_posterior_samples(self, rng_key=jax.random.PRNGKey(1), num_samples=50000, svi_results=None):
        """Sample from the variational posterior; returns a dictionary of posterior samples"""
        rng_key, key = jax.random.split(rng_key)
        if svi_results is None:
            svi_results = self.svi_results
        self.posterior_dict = self.guide.sample_posterior(rng_key=rng_key, params=svi_results.params, sample_shape=(num_samples,))
        return self.posterior_dict

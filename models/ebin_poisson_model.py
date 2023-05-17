"""Poisson models with energy binning."""

import numpy as np
import healpy as hp
import scipy.stats as scipy_stats
import scipy.optimize as scipy_optimize
import logging

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import jax.scipy.optimize as optimize
from jax.example_libraries import stax

import optax
import numpyro
import numpyro.distributions as dist
from numpyro import optim
from numpyro.infer import SVI, Trace_ELBO, autoguide
from numpyro.infer.reparam import NeuTraReparam
from numpyro.infer import MCMC, NUTS
from numpyro.contrib.tfp.mcmc import ReplicaExchangeMC
from tensorflow_probability.substrates import jax as tfp

from utils import create_mask as cm
from utils.sph_harm import Ylm

from models.templates import NFWTemplate, LorimerDiskTemplate
from models.bulge_models import BulgeTemplates
from likelihoods.pll_jax import log_like_poisson

# Because of masking, for now we use numpy up till the final construction of summed templates

class EbinTemplate:
    """Simple class for templates with energy binning.
    Currently only supports a predetermined energy binning.
    
    Parameters
    ----------
    data : ndarray, shape=(nebin, npix)
        Energy binned Healpix data arrays.
    fit_type : {'total norm', 'bin norm', 'power law'}
        Types of fit that should be carried out for this template.
    norm_mask : ndarray
        Mask used to normalize template. Not energy dependent. Not stored.
    norm_ie_range : tuple, (ie_from, ie_to), end exclusive
        Range of energy to normalize over.
    """
    
    def __init__(self, data, fit_type='bin norm', norm_mask=None, norm_ie_range=None):
        
        self.data = data
        self.fit_type = fit_type
        self.norm_ie_range = norm_ie_range
            
        if norm_mask is not None: # mask: 1 means to mask out, 0 means to include
            if self.fit_type == 'total norm':
                self.data /= jnp.mean(self.data[self.norm_ie_range[0]:self.norm_ie_range[1], ~norm_mask])
            elif self.fit_type == 'bin norm':
                self.data /= jnp.mean(self.data[:, ~norm_mask], axis=1)[:, None]
            else:
                raise NotImplementedError(self.fit_type)
                
    def at_bin(self, ie):
        """Get data at ith E bin."""
        return self.data[ie]
                
class Template:
    """Simple class for templates with NO energy dependence.
    
    Parameters
    ----------
    data : ndarray, shape=(npix,)
        Healpix data array.
    norm_mask : ndarray
        Mask used to normalize template. Not energy dependent. Not stored.
    """
    def __init__(self, data, norm_mask=None):
        
        self.data = data
        self.data /= jnp.mean(self.data[~norm_mask])
        
    def at_bin(self, ie):
        """Get data at ith E bin. (Returns non energy dependent template data.)"""
        return self.data
        
        
ebin_data_dir = '/n/holyscratch01/iaifi_lab/yitians/fermi/fermi-prob-prog/data/fermi_data_573w/ebin'
default_data_dir = '../data/fermi_data_573w/fermi_data_256'

DATA_NSIDE = 512

class EbinPoissonModel:
    """
    Energy binned model for poisson fits.

    Parameters
    ----------
    nside : 512 or lower powers of 2.
        HEALPix NSIDE parameter.
    ps_cat : str
        Point source catalog.
    data_class : str
        Data class.
    temp_class : str
        Template class.
    mask_class : str
        Mask class.
    mask_roi_r_outer : float
        Outer radius of the region of interest mask, in degrees.
    mask_roi_b : float
        Latitude boundary of the region of interest mask, in degrees.
    dif_names : list of str, can be empty
        List of diffuse model names.
    blg_names : list of str, can be empty
        List of bulge model names.
    nfw_gamma : {'vary', float}
        NFW gamma parameter, can be either 'vary' (default) or a float number.
    disk_option : {'vary', 'fixed', 'none'}
        Option for the disk model.
    l_max : int
        Maximum multipole moment for the harmonic expansion, default is -1 (turned off).
    """

    
    def __init__(
        self,
        nside = 256,
        ps_cat = '3fgl',
        data_class = 'fwhm000-0512-bestpsf-nopsc',
        temp_class = 'ultracleanveto-bestpsf',
        mask_class = 'fwhm000-0512-bestpsf-mask',
        mask_roi_r_outer = 20.,
        mask_roi_b = 2.,
        dif_names = ['ccwa', 'ccwf', 'mO'],
        blg_names = ['mcdermott2022', 'mcdermott2022_bbp', 'mcdermott2022_x', 'macias2019', 'coleman2019'],
        nfw_gamma = 'vary',
        disk_option = 'none',
        l_max=-1,
    ):
        
        self.nside = nside
        self.ps_cat = ps_cat
        self.data_class = data_class
        self.temp_class = temp_class
        self.mask_class = mask_class
        self.mask_roi_r_outer = mask_roi_r_outer
        self.mask_roi_b = mask_roi_b
        self.dif_names = dif_names
        self.blg_names = blg_names
        self.nfw_gamma = nfw_gamma
        self.disk_option = disk_option
        self.l_max = l_max
        
        #========== Data ==========
        to_nside = lambda x: hp.pixelfunc.ud_grade(x, self.nside)
        self.counts = np.array(
            to_nside(np.load(f'{ebin_data_dir}/counts-{self.data_class}.npy')) * (DATA_NSIDE/self.nside)**2,
            dtype = np.int32
        ) # sum rather than interpolate
        self.exposure = to_nside(np.load(f'{ebin_data_dir}/exposure-{self.data_class}.npy')) * (DATA_NSIDE/self.nside)**2  # sum rather than interpolate
        
        #========== Mask ==========
        self.mask_ps_arr = to_nside(np.load(f'{ebin_data_dir}/mask-{self.mask_class}.npy')) > 0
        self.mask_roi_arr = np.asarray([
            cm.make_mask_total(
                nside=self.nside,
                band_mask=True,
                band_mask_range=self.mask_roi_b,
                mask_ring=True,
                inner=0,
                outer=self.mask_roi_r_outer,
                custom_mask=mask_ps_at_eng
            )
            for mask_ps_at_eng in self.mask_ps_arr
        ])
        self.normalization_mask = np.asarray(
            cm.make_mask_total(
                nside=self.nside,
                band_mask=True,
                band_mask_range=2,
                mask_ring=True,
                inner=0,
                outer=25,
            )
        )
        
        #========== Rigid templates ==========
        self.temps = {
            'iso' : EbinTemplate(
                self.exposure.copy(),
                fit_type='bin norm',
                norm_mask=self.normalization_mask,
            ),
            'psc' : EbinTemplate(
                to_nside(np.load(f'{ebin_data_dir}/psc-bestpsf-3fgl.npy')),
                fit_type='bin norm',
                norm_mask=self.normalization_mask,
            ),
            'bub' : Template(
                to_nside(np.load(f'{default_data_dir}/template_bub.npy')),
                norm_mask=self.normalization_mask
            ),
            'dsk' : Template(
                to_nside(np.load(f'{default_data_dir}/template_dsk_z1p0.npy')),
                norm_mask=self.normalization_mask
            ),
            'nfw' : Template(
                to_nside(np.load(f'{default_data_dir}/template_nfw_g1p0.npy')),
                norm_mask=self.normalization_mask
            ),
        }
        
        if self.l_max >= 0:
            npix = hp.nside2npix(self.nside)
            theta_ary, phi_ary = hp.pix2ang(self.nside, np.arange(npix))
            Ylm_list = [
                [np.real(Ylm(l, m, theta_ary, phi_ary)) for m in range(-l + 1, l + 1)]
                for l in range(1, self.l_max + 1)
            ]
            self.Ylm_temps = np.array([item for sublist in Ylm_list for item in sublist])
        
        #========== Hybrid templates ==========
        self.n_dif_temps = len(self.dif_names)
        self.pib_temps = [
            EbinTemplate(
                to_nside(np.load(f'{ebin_data_dir}/{dif_name}pibrem-{self.temp_class}.npy')),
                fit_type='bin norm',
                norm_mask=self.normalization_mask,
            )
            for dif_name in dif_names
        ]
        self.ics_temps = [
            EbinTemplate(
                to_nside(np.load(f'{ebin_data_dir}/{dif_name}ics-{self.temp_class}.npy')),
                fit_type='bin norm',
                norm_mask=self.normalization_mask,
            )
            for dif_name in dif_names
        ]
        self.n_blg_temps = len(self.blg_names)
        self.blg_temps = [
            Template(
                BulgeTemplates(template_name=blg_name, nside_out=self.nside)(),
                norm_mask=self.normalization_mask,
            )
            for blg_name in blg_names
        ]
        
        #========== Variable template generators ==========
        self.nfw_temp = NFWTemplate(nside=self.nside)
        self.dsk_temp = LorimerDiskTemplate(nside=self.nside)
        
        #========== sample expand keys ==========
        self.samples_expand_keys = {
            'theta_pib' : [f'theta_pib_{n}' for n in self.dif_names],
            'theta_ics' : [f'theta_ics_{n}' for n in self.dif_names],
            'theta_blg' : [f'theta_blg_{n}' for n in self.blg_names],
        }
        
        
    #========== Model ==========
    def model_at_bin(self, data, ie=0):
        
        mu = jnp.zeros_like(data)
        
        #===== rigid templates =====
        # all templates should be already normalized
        for k in ['iso', 'psc', 'bub']:
            S_k = numpyro.sample(f'S_{k}', dist.Uniform(1e-3, 5))
            mu += S_k * jnp.asarray(self.temps[k].at_bin(ie))
            
        #===== hybrid templates =====
        # all templates should be already normalized
        if self.n_dif_temps > 0:
            S_pib = numpyro.sample('S_pib', dist.Uniform(1e-3, 10))
            S_ics = numpyro.sample('S_ics', dist.Uniform(1e-3, 10))
            if self.n_dif_temps == 1:
                mu += S_pib * self.pib_temps[0].at_bin(ie) + S_ics * self.ics_temps[0].at_bin(ie)
            else:
                theta_pib = numpyro.sample("theta_pib", dist.Dirichlet(jnp.ones((self.n_dif_temps,)) / self.n_dif_temps))
                theta_ics = numpyro.sample("theta_ics", dist.Dirichlet(jnp.ones((self.n_dif_temps,)) / self.n_dif_temps))
                pib_temps_at_bin = jnp.asarray([pib_temp.at_bin(ie) for pib_temp in self.pib_temps])
                ics_temps_at_bin = jnp.asarray([ics_temp.at_bin(ie) for ics_temp in self.ics_temps])
                mu += S_pib * jnp.dot(theta_pib, pib_temps_at_bin) + S_ics * jnp.dot(theta_ics, ics_temps_at_bin)
            
        if self.n_blg_temps > 0:
            S_blg = numpyro.sample('S_blg', dist.Uniform(1e-3, 10))
            if self.n_blg_temps == 1:
                mu += S_blg * self.blg_temps[0].at_bin(ie)
            else:
                theta_blg = numpyro.sample("theta_blg", dist.Dirichlet(jnp.ones((self.n_blg_temps,)) / self.n_blg_temps))
                blg_temps_at_bin = jnp.asarray([blg_temp.at_bin(ie) for blg_temp in self.blg_temps])
                mu += S_blg * jnp.dot(theta_blg, blg_temps_at_bin)
            
        #===== variable templates =====
        S_nfw = numpyro.sample('S_nfw', dist.Uniform(1e-3, 5))
        if self.nfw_gamma == 'vary':
            gamma = numpyro.sample("gamma", dist.Uniform(0.2, 2))
        else:
            gamma = self.nfw_gamma
        mu += S_nfw * self.nfw_temp.get_NFW2_template(gamma=gamma)
        
        if self.disk_option in ['vary', 'fixed']:
            S_dsk = numpyro.sample('S_dsk', dist.Uniform(1e-3, 5))
            if self.disk_option == 'vary':
                zs = numpyro.sample("zs", dist.Uniform(0.1, 2.5))
                C  = numpyro.sample("C",  dist.Uniform(0.05, 15.))
                temp_dsk = self.dsk_temp.get_template(zs=zs, C=C)
            else:
                temp_dsk = self.temps['dsk'].at_bin(ie)
            mu += S_dsk * temp_dsk
        
        #===== deterministic =====
        numpyro.deterministic('f_blg', S_blg / (S_blg + S_nfw))
            
        #===== data =====
        mu_roi = mu[~self.mask_roi_arr[ie]]
        data_roi = data[~self.mask_roi_arr[ie]]
        with numpyro.plate('data', size=len(mu_roi), dim=-1):
            return numpyro.factor('log_likelihood', log_like_poisson(mu_roi, data_roi))
        
    
    #========== SVI ==========
    def fit_SVI_at_bin(
        self,
        ie=10,
        rng_key=jax.random.PRNGKey(42),
        n_steps=7500,
        lr=5e-5,
        num_particles=8,
        num_base_mixture=8,
        guide='iaf',
        num_flows=5,
        hidden_dims=[256, 256]
    ):
        class AutoIAFMixture(autoguide.AutoIAFNormal):
            def get_base_dist(self):
                C = num_base_mixture
                mixture = dist.MixtureSameFamily(
                    dist.Categorical(probs=jnp.ones(C) / C),
                    dist.Normal(jnp.arange(float(C)), 1.)
                )
                return mixture.expand([self.latent_dim]).to_event()

        if guide == 'mvn':
            self.guide = autoguide.AutoMultivariateNormal(self.model_at_bin)
        elif guide == 'iaf':
            self.guide = autoguide.AutoIAFNormal(
                self.model_at_bin,
                num_flows=num_flows,
                hidden_dims=hidden_dims,
                nonlinearity=stax.Tanh
            )
        elif guide == 'iaf_mixture':
            self.guide = AutoIAFMixture(
                self.model_at_bin,
                num_flows=num_flows,
                hidden_dims=hidden_dims,
                nonlinearity=stax.Tanh
            )
        else:
            raise NotImplementedError

        # schedule = optax.exponential_decay(
        #     lr,
        #     transition_steps=2500,
        #     decay_rate=0.5,
        #     transition_begin=2500,
        #     staircase=True,
        #     end_value=1e-6
        # )
        optimizer = optim.optax_to_numpyro(
            optax.chain(
                optax.clip(1.),
                optax.adam(lr),
            )
        )
        svi = SVI(
            self.model_at_bin,
            self.guide,
            optimizer,
            Trace_ELBO(num_particles=num_particles),
            ie=ie
        )
        self.svi_results = svi.run(rng_key, n_steps, self.counts[ie])
        self.svi_ie = ie
        
        return self.svi_results
    
    
    def get_svi_samples(self, rng_key=jax.random.PRNGKey(42), num_samples=50000, expand_samples=True):
        
        rng_key, key = jax.random.split(rng_key)
        self.svi_samples = self.guide.sample_posterior(
            rng_key=rng_key,
            params=self.svi_results.params,
            sample_shape=(num_samples,)
        )
        
        if expand_samples:
            new_samples = {}
            for k in self.svi_samples.keys():
                if k in self.samples_expand_keys:
                    for i in range(self.svi_samples[k].shape[-1]):
                        new_samples[self.samples_expand_keys[k][i]] = self.svi_samples[k][...,i]
                else:
                    new_samples[k] = self.svi_samples[k]
            self.svi_samples = new_samples
            
        return self.svi_samples
        
    
    #========== NeuTra ==========
    def get_neutra_model(self):
        """Get model reparameterized via neural transport.
        """
        neutra = NeuTraReparam(self.guide, self.svi_results.params)
        model = lambda data: self.model_at_bin(data, ie=self.svi_ie)
        self.model_neutra = neutra.reparam(model)
        
    
    #========== NUTS ==========
    def run_nuts(self, num_chains=4, num_warmup=500, num_samples=5000, step_size=0.1, rng_key=jax.random.PRNGKey(0)):
        
        self.get_neutra_model()
        
        kernel = NUTS(self.model_neutra, max_tree_depth=4, dense_mass=False, step_size=step_size)
        self.nuts_mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, chain_method='vectorized')
        self.nuts_mcmc.run(rng_key, data=self.counts[self.svi_ie])
        
        return self.nuts_mcmc
    
    
    #========== PTHMC ==========
    def run_parallel_tempering_hmc(self, num_samples=5000, step_size_base=5e-2, num_leapfrog_steps=3, num_adaptation_steps=600, rng_key=jax.random.PRNGKey(0)):
        
        # Geometric temperatures decay
        inverse_temperatures = 0.5 ** jnp.arange(4.)

        # If everything was Normal, step_size should be ~ sqrt(temperature).
        step_size = step_size_base / jnp.sqrt(inverse_temperatures)[..., None]

        def make_kernel_fn(target_log_prob_fn):

            hmc = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            step_size=step_size, num_leapfrog_steps=num_leapfrog_steps)

            adapted_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
            inner_kernel=hmc,
            num_adaptation_steps=num_adaptation_steps)

            return adapted_kernel
        
        self.get_neutra_model()
        
        kernel = ReplicaExchangeMC(self.model_neutra, inverse_temperatures=inverse_temperatures, make_kernel_fn=make_kernel_fn)
        self.pt_mcmc = MCMC(kernel, num_warmup=num_adaptation_steps, num_samples=num_samples, num_chains=1, chain_method='vectorized')
        self.pt_mcmc.run(rng_key, self.counts[self.svi_ie])
        
        return self.pt_mcmc
    
    
    #========== MAP ==========
    def fit_MAP_at_bin(self, ie=10, rng_key=jax.random.PRNGKey(42), lr=0.1, n_steps=10000):
        
        #optimizer = numpyro.optim.Adam(lr=lr)
        model = lambda data: self.model_at_bin(data, ie=ie)
        guide = autoguide.AutoDelta(model)
        optimizer = optim.optax_to_numpyro(optax.chain(optax.clip(1.), optax.adamw(lr)))
        svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
        svi_results = svi.run(rng_key, n_steps, self.counts[ie])
        self.MAP_estimates = guide.median(svi_results.params)
        
        return svi_results
    
    
    #========== MLE ==========
    def fit_MLE_at_bin(self, ie, params0=None, bounds=None, method='scipy minimize'):
        """Already deprecated somehow."""
        mask = self.mask_roi_arr[ie]
        masked_temps = np.asarray([
            self.temps[k].data[ie, ~mask]
            for k in ['iso', 'psc', 'bub', 'dsk', 'pib', 'ics', 'blg', 'nfw']
        ])
        masked_data = self.counts[ie, ~mask]
        
        if params0 is None:
            params0 = np.ones((len(masked_temps),))
        if bounds is None:
            bounds = [(-10, 10) for _ in masked_temps]
            
        def log_likelihood(log_params):
            return - scipy_stats.poisson.logpmf(masked_data, np.dot(np.exp(log_params), masked_temps)).mean()
        
        if method == 'scipy minimize':
            result = scipy_optimize.minimize(log_likelihood, x0=np.log(params0), bounds=bounds)
            
        elif method == 'scipy shgo':
            result = scipy_optimize.shgo(log_likelihood, bounds=bounds)
            
        elif method == 'jax.scipy minimize':
            masked_temps = jnp.asarray(masked_temps, dtype=jnp.float32)
            masked_data  = jnp.asarray(masked_data,  dtype=jnp.int32)
            def log_likelihood(log_params):
                return - stats.poisson.logpmf(masked_data, jnp.dot(jnp.exp(log_params), masked_temps)).mean()
            result = optimize.minimize(log_likelihood, x0=jnp.log(params0), method='BFGS')
            
        else:
            raise NotImplementedError
            
        return result
    
    

"""Poissonian models for healpix with energy binning."""

import os
import sys
sys.path.append("..")

import numpy as np
import healpy as hp

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
from utils.map_utils import to_nside

from templates.rigid_templates import EbinTemplate, Template, BulgeTemplates
from templates.variable_templates import NFWTemplate, LorimerDiskTemplate
from likelihoods.pll_jax import log_like_poisson

# Because of masking, for now we use numpy up till the final construction of
# summed templates


class EbinPoissonModel:
    """
    Energy binned model for poisson fits.

    Parameters
    ----------
    nside : 512 or lower powers of 2.
        HEALPix NSIDE parameter.
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
        nside = 128,
        #ps_cat = '3fgl',
        data_class = 'fwhm000-0512-bestpsf-nopsc',
        temp_class = 'ultracleanveto-bestpsf',
        mask_class = 'fwhm000-0512-bestpsf-mask',
        mask_roi_r_outer = 20.,
        mask_roi_b = 2.,
        dif_names = ['ccwa', 'ccwf', 'modelo'],
        blg_names = ['mcdermott2022', 'mcdermott2022_bbp', 'mcdermott2022_x', 'macias2019', 'coleman2019'],
        nfw_gamma = 'vary',
        disk_option = 'none',
        l_max = -1,
    ):
        
        self.nside = nside
        #self.ps_cat = ps_cat
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
        
        if self.nside > 128:
            ebin_data_dir = '../data/fermi_data_573w/ebin'
            if not os.path.isdir(ebin_data_dir):
                print('NSIDE > 128 requires ebin_512 dataset.')
        else:
            ebin_data_dir = '../data/fermi_data_573w/ebin_128'
        default_data_dir = '../data/fermi_data_573w/fermi_data_256'
        
        #========== Data ==========
        self.counts = np.array(
            to_nside(
                np.load(f'{ebin_data_dir}/counts-{self.data_class}.npy'),
                self.nside,
                mode='sum',
            ),
            dtype = np.int32
        )
        self.exposure = to_nside(
            np.load(f'{ebin_data_dir}/exposure-{self.data_class}.npy'),
            self.nside,
            mode='sum',
        )
        
        #========== Mask ==========
        self.mask_ps_arr = to_nside(np.load(f'{ebin_data_dir}/mask-{self.mask_class}.npy'), self.nside) > 0
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
                norm_mask=self.normalization_mask,
            ),
            'psc' : EbinTemplate(
                to_nside(np.load(f'{ebin_data_dir}/psc-bestpsf-3fgl.npy'), self.nside),
                norm_mask=self.normalization_mask,
            ),
            'bub' : Template(
                to_nside(np.load(f'{default_data_dir}/template_bub.npy'), self.nside),
                norm_mask=self.normalization_mask
            ),
            'dsk' : Template(
                to_nside(np.load(f'{default_data_dir}/template_dsk_z1p0.npy'), self.nside),
                norm_mask=self.normalization_mask
            ),
            'nfw' : Template(
                to_nside(np.load(f'{default_data_dir}/template_nfw_g1p0.npy'), self.nside),
                norm_mask=self.normalization_mask
            ),
        }
        
        # if self.l_max >= 0:
        #     npix = hp.nside2npix(self.nside)
        #     theta_ary, phi_ary = hp.pix2ang(self.nside, np.arange(npix))
        #     Ylm_list = [
        #         [np.real(Ylm(l, m, theta_ary, phi_ary)) for m in range(-l + 1, l + 1)]
        #         for l in range(1, self.l_max + 1)
        #     ]
        #     self.Ylm_temps = np.array([item for sublist in Ylm_list for item in sublist])
        
        #========== Hybrid (rigid) templates ==========
        self.n_dif_temps = len(self.dif_names)
        self.pib_temps = [
            EbinTemplate(
                to_nside(np.load(f'{ebin_data_dir}/{dif_name}pibrem-{self.temp_class}.npy'), self.nside),
                norm_mask=self.normalization_mask,
            )
            for dif_name in dif_names
        ]
        self.ics_temps = [
            EbinTemplate(
                to_nside(np.load(f'{ebin_data_dir}/{dif_name}ics-{self.temp_class}.npy'), self.nside),
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
        
        #========== Variable templates ==========
        self.nfw_temp = NFWTemplate(nside=self.nside)
        self.dsk_temp = LorimerDiskTemplate(nside=self.nside)
        
        #========== sample expand keys ==========
        self.samples_expand_keys = {
            'theta_pib' : [f'theta_pib_{n}' for n in self.dif_names],
            'theta_ics' : [f'theta_ics_{n}' for n in self.dif_names],
            'theta_blg' : [f'theta_blg_{n}' for n in self.blg_names],
        }
        
        
    #========== Model ==========
    def config_model(self, ebin=10):
        
        if ebin == 'all':
            raise NotImplementedError
        else:
            ie = int(ebin)
            
        self.nfw_temp = NFWTemplate(nside=self.nside)
        self.nfw_temp.set_mask(self.mask_roi_arr[ie])
        
    
    def model(self, ebin=10):
        
        if ebin == 'all':
            raise NotImplementedError
        else:
            ie = int(ebin)
            
        mask = self.mask_roi_arr[ie]
        data = self.counts[ie][~mask]
        mu = jnp.zeros_like(data)
        
        #===== rigid templates =====
        # all templates should be already normalized
        for k in ['iso', 'psc', 'bub']:
            S_k = numpyro.sample(f'S_{k}', dist.Uniform(1e-3, 5))
            mu += S_k * jnp.asarray(self.temps[k].at_bin(ie, mask=mask))
            
        #===== hybrid templates =====
        # all templates should be already normalized
        if self.n_dif_temps > 0:
            S_pib = numpyro.sample('S_pib', dist.Uniform(1e-3, 10))
            S_ics = numpyro.sample('S_ics', dist.Uniform(1e-3, 10))
            if self.n_dif_temps == 1:
                mu += S_pib * self.pib_temps[0].at_bin(ie, mask=mask) + S_ics * self.ics_temps[0].at_bin(ie, mask=mask)
            else:
                theta_pib = numpyro.sample("theta_pib", dist.Dirichlet(jnp.ones((self.n_dif_temps,)) / self.n_dif_temps))
                theta_ics = numpyro.sample("theta_ics", dist.Dirichlet(jnp.ones((self.n_dif_temps,)) / self.n_dif_temps))
                pib_temps_at_bin = jnp.asarray([pib_temp.at_bin(ie, mask=mask) for pib_temp in self.pib_temps])
                ics_temps_at_bin = jnp.asarray([ics_temp.at_bin(ie, mask=mask) for ics_temp in self.ics_temps])
                mu += S_pib * jnp.dot(theta_pib, pib_temps_at_bin) + S_ics * jnp.dot(theta_ics, ics_temps_at_bin)
            
        if self.n_blg_temps > 0:
            S_blg = numpyro.sample('S_blg', dist.Uniform(1e-3, 10))
            if self.n_blg_temps == 1:
                mu += S_blg * self.blg_temps[0].at_bin(ie, mask=mask)
            else:
                theta_blg = numpyro.sample("theta_blg", dist.Dirichlet(jnp.ones((self.n_blg_temps,)) / self.n_blg_temps))
                blg_temps_at_bin = jnp.asarray([blg_temp.at_bin(ie, mask=mask) for blg_temp in self.blg_temps])
                mu += S_blg * jnp.dot(theta_blg, blg_temps_at_bin)
            
        #===== variable templates =====
        S_nfw = numpyro.sample('S_nfw', dist.Uniform(1e-3, 5))
        if self.nfw_gamma == 'vary':
            gamma = numpyro.sample("gamma", dist.Uniform(0.2, 2))
        else:
            gamma = self.nfw_gamma
        mu += S_nfw * self.nfw_temp.get_NFW2_masked_template(gamma=gamma)
        
        if self.disk_option in ['vary', 'fixed']:
            S_dsk = numpyro.sample('S_dsk', dist.Uniform(1e-3, 5))
            if self.disk_option == 'vary':
                zs = numpyro.sample("zs", dist.Uniform(0.1, 2.5))
                C  = numpyro.sample("C",  dist.Uniform(0.05, 15.))
                temp_dsk = self.dsk_temp.get_template(zs=zs, C=C)[~mask]
            else:
                temp_dsk = self.temps['dsk'].at_bin(ie, mask=mask)
            mu += S_dsk * temp_dsk
        
        #===== deterministic =====
        numpyro.deterministic('f_blg', S_blg / (S_blg + S_nfw))
            
        #===== data =====
        #mu_roi = mu[~self.mask_roi_arr[ie]]
        #data_roi = self.counts[ie][~self.mask_roi_arr[ie]]
        with numpyro.plate('data', size=len(mu), dim=-1):
            return numpyro.factor('log_likelihood', log_like_poisson(mu, data))
        
    
    #========== SVI ==========
    def fit_SVI(
        self, rng_key=jax.random.PRNGKey(42),
        guide='iaf', num_flows=5, hidden_dims=[256, 256],
        n_steps=7500, lr=5e-5, num_particles=8,
        **model_static_kwargs,
    ):
        if guide == 'mvn':
            self.guide = autoguide.AutoMultivariateNormal(self.model)
        elif guide == 'iaf':
            self.guide = autoguide.AutoIAFNormal(
                self.model,
                num_flows=num_flows,
                hidden_dims=hidden_dims,
                nonlinearity=stax.Tanh
            )
        elif guide == 'iaf_mixture':
            num_base_mixture = 8
            class AutoIAFMixture(autoguide.AutoIAFNormal):
                def get_base_dist(self):
                    C = num_base_mixture
                    mixture = dist.MixtureSameFamily(
                        dist.Categorical(probs=jnp.ones(C) / C),
                        dist.Normal(jnp.arange(float(C)), 1.)
                    )
                    return mixture.expand([self.latent_dim]).to_event()
            self.guide = AutoIAFMixture(
                self.model,
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
            self.model, self.guide, optimizer,
            Trace_ELBO(num_particles=num_particles),
            **model_static_kwargs,
        )
        self.svi_results = svi.run(rng_key, n_steps)
        self.svi_model_static_kwargs = model_static_kwargs
        
        return self.svi_results
    
    
    def get_svi_samples(self, rng_key=jax.random.PRNGKey(42), num_samples=50000, expand_samples=True):
        
        rng_key, key = jax.random.split(rng_key)
        self.svi_samples = self.guide.sample_posterior(
            rng_key=rng_key,
            params=self.svi_results.params,
            sample_shape=(num_samples,)
        )
        
        if expand_samples:
            self.svi_samples = self.expand_samples(self.svi_samples)
            
        return self.svi_samples
    
    def expand_samples(self, samples):
        new_samples = {}
        for k in samples.keys():
            if k in self.samples_expand_keys:
                for i in range(samples[k].shape[-1]):
                    new_samples[self.samples_expand_keys[k][i]] = samples[k][...,i]
            elif k in ['auto_shared_latent']:
                pass
            else:
                new_samples[k] = samples[k]
        return new_samples
        
    
    #========== NeuTra ==========
    def get_neutra_model(self):
        """Get model reparameterized via neural transport."""
        neutra = NeuTraReparam(self.guide, self.svi_results.params)
        model = lambda x: self.model(**self.svi_model_static_kwargs)
        self.model_neutra = neutra.reparam(model)
        
    
    #========== NUTS ==========
    def run_nuts(self, num_chains=4, num_warmup=500, num_samples=5000, step_size=0.1,
                 rng_key=jax.random.PRNGKey(0), use_neutra=True, **model_static_kwargs):
        
        if use_neutra:
            self.get_neutra_model()
            model = self.model_neutra
        else:
            model = self.model
        
        kernel = NUTS(model, max_tree_depth=4, dense_mass=False, step_size=step_size)
        self.nuts_mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, chain_method='vectorized')
        if use_neutra:
            self.nuts_mcmc.run(rng_key, None)
        else:
            self.nuts_mcmc.run(rng_key, **model_static_kwargs)
        
        return self.nuts_mcmc
    
    
    #========== PTHMC ==========
    def run_parallel_tempering_hmc(self, num_samples=5000, step_size_base=5e-2, num_leapfrog_steps=3, num_adaptation_steps=600, rng_key=jax.random.PRNGKey(0), use_neutra=True):
        
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
        
        if use_neutra:
            self.get_neutra_model()
            model = self.model_neutra
        else:
            model = lambda x: self.model(**self.svi_model_static_kwargs)
        
        kernel = ReplicaExchangeMC(model, inverse_temperatures=inverse_temperatures, make_kernel_fn=make_kernel_fn)
        self.pt_mcmc = MCMC(kernel, num_warmup=num_adaptation_steps, num_samples=num_samples, num_chains=1, chain_method='vectorized')
        self.pt_mcmc.run(rng_key, None)
        
        return self.pt_mcmc
    
    
    #========== MAP ==========
    def fit_MAP(
        self, rng_key=jax.random.PRNGKey(42),
        lr=0.1, n_steps=10000, num_particles=8,
        **model_static_kwargs,
    ):
        guide = autoguide.AutoDelta(self.model)
        optimizer = optim.optax_to_numpyro(optax.chain(optax.clip(1.), optax.adamw(lr)))
        svi = SVI(
            self.model, guide, optimizer,
            loss=Trace_ELBO(num_particles=num_particles),
            **model_static_kwargs,
        )
        svi_results = svi.run(rng_key, n_steps)
        self.MAP_estimates = guide.median(svi_results.params)
        
        return svi_results
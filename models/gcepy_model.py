"""Models from gcepy
https://github.com/samueldmcdermott/gcepy/tree/main/gcepy
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax.example_libraries import stax
import numpyro
import numpyro.distributions as dist
from numpyro import optim
from numpyro.infer import SVI, Trace_ELBO, autoguide
import optax

from likelihoods.pll_jax import log_like_poisson
from models.ebin_poisson_model import EbinPoissonModel

class EbinPoissonModelGCEPy (EbinPoissonModel):
    
    def __init__(self):
        
        ddir = "../data/external/gcepy/inputs"
        postfix = 'front_only_14_Ebin_20x20window_normal'
        self.n_ebin = 14
        
        #========== templates ==========
        self.temps = {
            'pib_7p' : jnp.load(f"{ddir}/templates_lowdim/bremss_model_7p_{postfix}.npy") \
                     + jnp.load(f"{ddir}/templates_lowdim/pion0_model_7p_{postfix}.npy"),
            'pib_8t' : jnp.load(f"{ddir}/templates_lowdim/bremss_model_8t_{postfix}.npy") \
                     + jnp.load(f"{ddir}/templates_lowdim/pion0_model_8t_{postfix}.npy"),
            'ics_7p' : jnp.load(f"{ddir}/templates_lowdim/ics_model_7p_{postfix}.npy"),
            'ics_8t' : jnp.load(f"{ddir}/templates_lowdim/ics_model_8t_{postfix}.npy"),
            'iso'    : jnp.load(f"{ddir}/utils/isotropic_{postfix}.npy"),
            'bub'    : jnp.load(f"{ddir}/utils/bubble_{postfix}.npy"),
            'gce_bb' : jnp.load(f"{ddir}/excesses/bb_{postfix}.npy"),
            'gce_bbp': jnp.load(f"{ddir}/excesses/bbp_{postfix}.npy"),
            'gce_dm' : jnp.load(f"{ddir}/excesses/dm_{postfix}.npy"),
            'gce_x'  : jnp.load(f"{ddir}/excesses/x_{postfix}.npy"),
        }
        self.mask = jnp.array(jnp.load(f"{ddir}/utils/mask_4FGL-DR2_14_Ebin_20x20window_normal.npy"), dtype=bool)
        self.temps_masked_ebin = [
            {
                key : temp[ie][self.mask[ie]]
                for key, temp in self.temps.items()
            }
            for ie in range(self.n_ebin)
        ]
        self.temps_masked_full = {
            key : jnp.concatenate([
                temp[ie][self.mask[ie]]
                for ie in range(self.n_ebin)
            ])
            for key, temp in self.temps.items()
        }
        
        #========== errors ==========
        temp_errors = jnp.load(f"{ddir}/utils/external_errors.npy")
        self.temp_errors = {
            'iso' : temp_errors[0],
            'bub' : temp_errors[1],
        }
        
        #========== counts ==========
        self.counts = jnp.array(jnp.load(f"{ddir}/utils/fermi_w009_to_w670_{postfix}.npy"), dtype=jnp.int32)
        self.counts_masked_ebin = [
            self.counts[ie][self.mask[ie]]
            for ie in range(self.n_ebin)
        ]
        self.counts_masked_full = jnp.concatenate(self.counts_masked_ebin)
        
        #========== sample expand keys ==========
        self.samples_expand_keys = {
            'theta_pib' : [f'theta_pib_{n}' for n in ['7p', '8t']],
            'theta_ics' : [f'theta_ics_{n}' for n in ['7p', '8t']],
            'theta_gce' : [f'theta_gce_{n}' for n in ['bb', 'bbp', 'dm', 'x']],
        }
        
        
    def model(self, ebin='all', error_mode='none'):
        """
        Model for a single energy bin or all bins.
        
        Parameters
        ----------
        ebin : 'all' or int
            'all' or index of energy bin fitted.
        error_mode : {'none', 'll', 'prior'}
            High latitude chi-squared error mode: none, included in
            log-likelihood, or included in the prior.
        """
        
        #===== parameters =====
        if error_mode == 'prior':
            
            raise NotImplementedError
            
            if ebin == 'all':
                sigma_iso = jnp.sqrt(jnp.sum(self.temp_errors['iso']**2))
                sigma_bub = jnp.sqrt(jnp.sum(self.temp_errors['bub']**2))
            else:
                sigma_iso = self.temp_errors['iso'][ebin]
                sigma_bub = self.temp_errors['bub'][ebin]
            S_iso = numpyro.sample('S_iso', dist.Normal(
                loc=1, scale=sigma_iso, constraint=dist.constraints.positive
            ))
            S_bub = numpyro.sample('S_bub', dist.Normal(
                loc=1, scale=sigma_iso, constraint=dist.constraints.positive
            ))
        else:
            S_iso = numpyro.sample('S_iso', dist.LogUniform(1e-2, 1e2))
            S_bub = numpyro.sample('S_bub', dist.LogUniform(1e-2, 1e2))
            
        S_pib = numpyro.sample('S_pib', dist.LogUniform(1e-2, 1e3))
        S_ics = numpyro.sample('S_ics', dist.LogUniform(1e-2, 1e3))
        S_gce = numpyro.sample('S_gce', dist.LogUniform(1e-2, 1e2))
        
        theta_pib = numpyro.sample('theta_pib', dist.Dirichlet(jnp.ones((2,)) / 2))
        theta_ics = numpyro.sample('theta_ics', dist.Dirichlet(jnp.ones((2,)) / 2))
        theta_gce = numpyro.sample('theta_gce', dist.Dirichlet(jnp.ones((4,)) / 4)) # gce includes 4 bulges and dm
        
        #===== calculate mu =====
        if ebin == 'all':
            temps = self.temps_masked_full
            data  = self.counts_masked_full
        else:
            temps = self.temps_masked_ebin[ebin]
            data  = self.counts_masked_ebin[ebin]
        mu =  (temps['pib_7p'] * theta_pib[0] + temps['pib_8t'] * theta_pib[1]) * S_pib
        mu += (temps['ics_7p'] * theta_ics[0] + temps['ics_8t'] * theta_ics[1]) * S_ics
        mu +=  temps['bub'] * S_bub
        mu +=  temps['iso'] * S_iso
        mu += (temps['gce_bb'] * theta_gce[0] + temps['gce_bbp'] * theta_gce[1] \
              +temps['gce_dm'] * theta_gce[2] + temps['gce_x'] * theta_gce[3]) * S_gce
        
        #===== likelihood =====
        with numpyro.plate('data', size=mu.shape[0], dim=-1):
            
            ll = log_like_poisson(mu, data)
            if error_mode == 'll':
                npix = mu.shape[0]
                ll += jnp.sum( ((S_iso - 1.)/self.temp_errors['iso']) ** 2 ) / npix
                ll += jnp.sum( ((S_bub - 1.)/self.temp_errors['bub']) ** 2 ) / npix
            return numpyro.factor('log_likelihood', ll)
        
        
    #==============================
    def fit_SVI(
        self, rng_key=jax.random.PRNGKey(42),
        num_flows=5, hidden_dims=[256, 256],
        n_steps=7500, lr=5e-5, num_particles=8,
        **model_static_kwargs,
    ):
        self.guide = autoguide.AutoIAFNormal(
            self.model,
            num_flows=num_flows,
            hidden_dims=hidden_dims,
            nonlinearity=stax.Tanh
        )
        optimizer = optim.optax_to_numpyro(
            optax.chain(
                optax.clip(1.),
                optax.adamw(lr),
            )
        )
        svi = SVI(
            self.model, self.guide, optimizer,
            Trace_ELBO(num_particles=num_particles),
            **model_static_kwargs,
        )
        self.svi_results = svi.run(rng_key, n_steps)
        
        return self.svi_results
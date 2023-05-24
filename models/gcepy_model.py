"""gcepy model"""

import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from likelihoods.pll_jax import log_like_poisson
from models.ebin_poisson_model import EbinPoissonModel

class EbinPoissonModelGCEPy (EbinPoissonModel):
    
    def __init__(self):
        
        ddir = "../data/external/gcepy/inputs"
        postfix = 'front_only_14_Ebin_20x20window_normal'
        
        self.temps = {
            'pib_7p' : np.load(f"{ddir}/templates_lowdim/bremss_model_7p_{postfix}.npy") \
                     + np.load(f"{ddir}/templates_lowdim/pion0_model_7p_{postfix}.npy"),
            'pib_8t' : np.load(f"{ddir}/templates_lowdim/bremss_model_8t_{postfix}.npy") \
                     + np.load(f"{ddir}/templates_lowdim/pion0_model_8t_{postfix}.npy"),
            'ics_7p' : np.load(f"{ddir}/templates_lowdim/ics_model_7p_{postfix}.npy"),
            'ics_8t' : np.load(f"{ddir}/templates_lowdim/ics_model_8t_{postfix}.npy"),
            'iso'    : np.load(f"{ddir}/utils/isotropic_{postfix}.npy"),
            'bub'    : np.load(f"{ddir}/utils/bubble_{postfix}.npy"),
            'gce_bb' : np.load(f"{ddir}/excesses/bb_{postfix}.npy"),
            'gce_bbp': np.load(f"{ddir}/excesses/bbp_{postfix}.npy"),
            'gce_dm' : np.load(f"{ddir}/excesses/dm_{postfix}.npy"),
            'gce_x'  : np.load(f"{ddir}/excesses/x_{postfix}.npy"),
        }
        self.mask = np.array(np.load(f"{ddir}/utils/mask_4FGL-DR2_14_Ebin_20x20window_normal.npy"), dtype=bool)
        self.temps_masked = {
            k : [
                jnp.array(t[ie][self.mask[ie]])
                for ie in range(t.shape[0])
            ]
            for k, t in self.temps.items()
        }
        
        self.counts = np.array(np.load(f"{ddir}/utils/fermi_w009_to_w670_{postfix}.npy"), dtype=np.int32)
        self.counts_masked = [
            jnp.array(self.counts[ie][self.mask[ie]], dtype=jnp.int32)
            for ie in range(self.counts.shape[0])
        ]
        
        #========== sample expand keys ==========
        self.samples_expand_keys = {
            'theta_pib' : [f'theta_pib_{n}' for n in ['7p', '8t']],
            'theta_ics' : [f'theta_ics_{n}' for n in ['7p', '8t']],
            'theta_gce' : [f'theta_gce_{n}' for n in ['bb', 'bbp', 'dm', 'x']],
        }
        
        
    def model_at_bin(self, data, ie=0):
        
        #===== random variables =====
        log10S_pib = numpyro.sample('log10S_pib', dist.Uniform(-2, 3))
        log10S_ics = numpyro.sample('log10S_ics', dist.Uniform(-2, 3))
        log10S_bub = numpyro.sample('log10S_bub', dist.Uniform(-2, 2))
        log10S_iso = numpyro.sample('log10S_iso', dist.Uniform(-2, 2))
        log10S_gce = numpyro.sample('log10S_gce', dist.Uniform(-2, 2))
        
        theta_pib = numpyro.sample('theta_pib', dist.Dirichlet(jnp.ones((2,)) / 2))
        theta_ics = numpyro.sample('theta_ics', dist.Dirichlet(jnp.ones((2,)) / 2))
        theta_gce = numpyro.sample('theta_gce', dist.Dirichlet(jnp.ones((4,)) / 4))
        
        #===== calculate mu =====
        t = self.temps_masked
        mu =  (t['pib_7p'][ie] * theta_pib[0] + t['pib_8t'][ie] * theta_pib[1]) * 10**log10S_pib
        mu += (t['ics_7p'][ie] * theta_ics[0] + t['ics_8t'][ie] * theta_ics[1]) * 10**log10S_ics
        mu +=  t['bub'][ie] * 10**log10S_bub
        mu +=  t['iso'][ie] * 10**log10S_iso
        mu += (t['gce_bb'][ie] * theta_gce[0] + t['gce_bbp'][ie] * theta_gce[1] \
              +t['gce_dm'][ie] * theta_gce[2] + t['gce_x'][ie] * theta_gce[3]) * 10**log10S_gce

        data_masked = self.counts_masked[ie]
        
        with numpyro.plate('data', size=len(mu), dim=-1):
            return numpyro.factor('log_likelihood', log_like_poisson(mu, data_masked))
import sys

sys.path.append("./")

import jax
from jax import jit, vmap
import jax.numpy as jnp
import numpy as np
from jax.config import config

config.update("jax_enable_x64", True)

from functools import partial


def dnds(s, theta):
    a, n1, n2, n3, sb1, sb2 = theta
    dnds = a * (sb2 / sb1) ** -n2 * jnp.where(s < sb2, (s / sb2) ** (-n3), jnp.where((s >= sb2) * (s < sb1), (s / sb2) ** (-n2), (sb1 / sb2) ** (-n2) * (s / sb1) ** (-n1)))
    return dnds


@partial(jit, static_argnums=(6, 7))
def log_like_np(theta, pt_sum_compressed, npt_compressed, data, f_ary, df_rho_div_f_ary, k_max, npixROI):
    """Organize and combine non-Poissonian likelihoods across multiple templates"""

    x_m_ary = jnp.zeros((npixROI, k_max + 1))
    x_m_sum = jnp.zeros(npixROI)

    s_ary = jnp.logspace(0.0, 2.5, 200)

    return_x_m_vmapped = vmap(return_x_m, in_axes=(None, None, 0, None, None, 0, None))

    dnds_ary = vmap(dnds, in_axes=(None, 0))(s_ary, theta)
    x_m, x_m_sum = return_x_m_vmapped(f_ary, df_rho_div_f_ary, npt_compressed, data, s_ary, dnds_ary, k_max)

    x_m = jnp.sum(x_m, axis=0)
    x_m_sum = jnp.sum(x_m_sum, axis=0)

    return log_like_internal(pt_sum_compressed, data, x_m, x_m_sum, k_max, npixROI)


@partial(jit, static_argnums=(4, 5))
def log_like_internal(pt_sum_compressed, data, x_m_ary, x_m_sum, k_max, npixROI):
    """Non-Poissonian likelihood for single template, given x_m and x_m_sum"""

    f0_ary = -(pt_sum_compressed + x_m_sum)
    f1_ary = pt_sum_compressed + x_m_ary[:, 1]

    pk_ary = jnp.zeros((npixROI, k_max + 1))

    pk_ary = pk_ary.at[:, 0].set(jnp.exp(f0_ary))
    pk_ary = pk_ary.at[:, 1].set(pk_ary[:, 0] * f1_ary)

    for k in np.arange(2, k_max + 1):
        n = jnp.arange(k - 1)
        pk_ary = pk_ary.at[:, k].set(jnp.sum((k - n) / k * x_m_ary[:, k - n] * pk_ary[:, n], axis=1) + f1_ary * pk_ary[:, k - 1] / k)

    pk_dat_ary = pk_ary[jnp.arange(npixROI), data]

    return jnp.log(pk_dat_ary)


@partial(jit, static_argnums=(6,))
def return_x_m(f_ary, df_rho_div_f_ary, npt_compressed, data, s_ary, dnds_ary, k_max):
    """Dedicated calculation of x_m and x_m_sum"""

    m_ary = jnp.arange(k_max + 1, dtype=jnp.float64)
    gamma_ary = jnp.exp(jax.lax.lgamma(m_ary + 1))

    x_m_ary = df_rho_div_f_ary[:, None] * f_ary[:, None] * jnp.trapz(((dnds_ary * jnp.exp(-jnp.outer(f_ary, s_ary)))[:, :, None] * jax.lax.pow(jnp.outer(f_ary, s_ary)[:, :, None], m_ary[None, None, :]) / gamma_ary), s_ary, axis=1)
    x_m_ary = jnp.sum(x_m_ary, axis=0)

    x_m_ary = jnp.outer(npt_compressed, x_m_ary)

    x_m_sum_ary = jnp.sum((df_rho_div_f_ary * f_ary)[:, None] * jnp.trapz(dnds_ary, s_ary), axis=0)
    x_m_sum_ary = jnp.sum(x_m_sum_ary, axis=0)

    x_m_sum_ary = npt_compressed * x_m_sum_ary - x_m_ary[:, 0]

    return x_m_ary, x_m_sum_ary

from jax import jit
import numpyro.distributions as dist

@jit
def log_like_poisson(mu, data):
    return dist.discrete.Poisson(mu).log_prob(data)
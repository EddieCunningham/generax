import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
import einops
import equinox as eqx
from abc import ABC, abstractmethod
from jaxtyping import Array, PRNGKeyArray
import generax.util.misc as misc
from generax.flows.base import BijectiveTransform
import numpy as np
import generax.util as util

__all__ = ['LogisticCDFMixtureLogit']

################################################################################################################

@jax.custom_jvp
def logistic_cdf_mixture_logit(weight_logits, means, scales, x):
  # weight_logits doesn't have to be normalized with log_softmax.
  # This normalization happens automatically when we compute z.

  shifted_x = x[...,None] - means
  x_hat = shifted_x*scales

  t1 = -jax.nn.softplus(-x_hat)
  t2 = t1 - x_hat

  t = weight_logits + jnp.concatenate([t1[None], t2[None]], axis=0)
  lse_t = jax.nn.logsumexp(t, axis=-1)
  log_z, log_1mz = lse_t
  z = log_z - log_1mz
  return z

@logistic_cdf_mixture_logit.defjvp
def jvp(primals, tangents):
  # We get the gradients (almost) for free when we evaluate the function
  weight_logits, means, scales, x = primals

  shifted_x = x[...,None] - means
  x_hat = shifted_x*scales

  t1 = -jax.nn.softplus(-x_hat)
  t2 = t1 - x_hat

  t12 = jnp.concatenate([t1[None], t2[None]], axis=0)
  t = weight_logits + t12
  lse_t = jax.nn.logsumexp(t, axis=-1)
  log_z, log_1mz = lse_t
  z = log_z - log_1mz

  # dz/dz_score
  softmax_t = jnp.exp(t - lse_t[...,None])
  softmax_t1, softmax_t2 = softmax_t
  sigma, sigma_bar = jnp.exp(t12)
  dx_hat = softmax_t1*sigma_bar + softmax_t2*sigma

  # Final gradients
  dmeans         = -dx_hat*scales
  dx             = -dmeans.sum(axis=-1)
  dscales        = dx_hat*shifted_x
  dweight_logits = softmax_t1 - softmax_t2

  tangent_out = jnp.sum(dweight_logits*tangents[0], axis=-1)
  tangent_out += jnp.sum(dmeans*tangents[1], axis=-1)
  tangent_out += jnp.sum(dscales*tangents[2], axis=-1)
  tangent_out += dx*tangents[3]

  return z, tangent_out

################################################################################################################

class LogisticCDFMixtureLogit(BijectiveTransform):
  """Used in Flow++ https://arxiv.org/pdf/1902.00275.pdf
  This is a logistic CDF mixture model followed by a logit.

  **Attributes**:
  - `theta`: The parameters of the transformation.
  """

  theta: Array

  K: int = eqx.field(static=True)

  def __init__(self,
               input_shape: Tuple[int],
               K: int = 8,
               *,
               key: PRNGKeyArray,
               **kwargs):
    """**Arguments**:

    - `input_shape`: The input shape.  Output size is the same as shape.
    - `key`: A `jax.random.PRNGKey` for initialization
    - `K`: The number of knots to use.
    """
    super().__init__(input_shape=input_shape,
                     **kwargs)
    self.K = K

    x_dim = util.list_prod(input_shape)
    self.theta = random.normal(key, shape=(x_dim*(3*self.K),))*0.1

  def __call__(self,
               x: Array,
               y: Optional[Array] = None,
               inverse: bool=False,
               **kwargs) -> Array:
    """**Arguments**:

    - `x`: The input to the transformation
    - `y`: The conditioning information
    - `inverse`: Whether to inverse the transformation

    **Returns**:
    `(z, log_det)`
    """
    assert x.shape == self.input_shape, 'Only works on unbatched data'
    # Flatten x
    x = x.ravel()

    theta = self.theta.reshape(x.shape + (3*self.K,))

    # Split the parameters
    weight_logits, means, scales = theta[...,:self.K], theta[...,self.K:2*self.K], theta[...,2*self.K:]
    scales = misc.square_plus(scales, gamma=1.0) + 1e-4

    # Create the jvp function that we'll need
    def f_and_df(x, *args):
      primals = weight_logits, means, scales, x
      tangents = jax.tree_util.tree_map(jnp.zeros_like, primals[:-1]) + (jnp.ones_like(x),)
      return jax.jvp(logistic_cdf_mixture_logit, primals, tangents)

    if inverse == False:
      # Only need a single pass
      z, dzdx = f_and_df(x)
    else:
      # Invert with bisection method.
      f = lambda x, *args: f_and_df(x, *args)[0]
      lower, upper = -1000.0, 1000.0
      lower, upper = jnp.broadcast_to(lower, x.shape), jnp.broadcast_to(upper, x.shape)
      z = util.bisection(f, lower, upper, x)
      reconstr, dzdx = f_and_df(z)
    ew_log_det = jnp.log(dzdx)

    log_det = ew_log_det.sum()

    if inverse:
      log_det *= -1

    # Unflatten the output
    z = z.reshape(self.input_shape)
    return z, log_det

################################################################################################################

if __name__ == '__main__':
  from debug import *
  import matplotlib.pyplot as plt
  from generax.flows.base import Sequential

  key = random.PRNGKey(0)
  x = random.normal(key, shape=(10, 2))

  layer = LogisticCDFMixtureLogit(input_shape=x.shape[1:],
                                  key=key)

  z, log_det = layer(x[0])
  x_reconstr, log_det2 = layer(z, inverse=True)
  assert jnp.allclose(log_det, -log_det2)

  G = jax.jacobian(lambda x: layer(x)[0])(x[0])
  log_det_true = jnp.linalg.slogdet(G)[1]

  assert jnp.allclose(log_det, log_det_true)
  assert jnp.allclose(x[0], x_reconstr)

  z, log_det = eqx.filter_vmap(layer)(x)

  import pdb; pdb.set_trace()

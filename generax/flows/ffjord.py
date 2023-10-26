import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
import einops
import equinox as eqx
from abc import ABC, abstractmethod
from jaxtyping import Array, PRNGKeyArray
import generax.nn.util as util
from generax.flows.base import BijectiveTransform
from generax.nn.neural_ode import NeuralODE

class FFJORD(BijectiveTransform):
  """Flow parametrized by a neural ODE https://arxiv.org/pdf/1810.01367.pdf

  **Attributes**:
  - `s_unbounded`: The unbounded scaling parameter.
  - `b`: The shift parameter.
  """

  neural_ode: NeuralODE
  trace_estimate_likelihood: bool

  def __init__(self,
               *_,
               x: Array,
               y: Optional[Array] = None,
               key: PRNGKeyArray,
               vf_type: type,
               controller_rtol: Optional[float] = 1e-3,
               controller_atol: Optional[float] = 1e-5,
               trace_estimate_likelihood: Optional[bool] = False,
               **kwargs):
    """**Arguments**:

    - `x`: A JAX array with shape `shape`. This is *required*
           to be batched!
    - `y`: A JAX array with shape `shape` representing conditioning
          information.  Should also be batched.
    - `key`: A `jax.random.PRNGKey` for initialization
    - `vf_type`: The type of vector field to use.  Should initializei an eqx.Module
                 that accepts `vf(t, x, y=y)` and returns an array of the same shape as `x`.
    """
    vf = vf_type(x=x, y=y, key=key, **kwargs)

    self.neural_ode = NeuralODE(vf=vf,
                                adjoint='seminorm',
                                controller_rtol=controller_rtol,
                                controller_atol=controller_atol)

    super().__init__(x=x,
                     y=y,
                     key=key,
                     **kwargs)

  def __call__(self,
               x: Array,
               y: Optional[Array] = None,
               inverse: bool=False) -> Array:
    """**Arguments**:

    - `x`: The input to the transformation
    - `y`: The conditioning information
    - `inverse`: Whether to inverse the transformation

    **Returns**:
    `(z, log_det)`
    """
    assert x.shape == self.input_shape, 'Only works on unbatched data'

    z, log_det = self.neural_ode(x,
                                 y=y,
                                 inverse=inverse,
                                 log_likelihood=True,
                                 trace_estimate_likelihood=self.trace_estimate_likelihood,
                                 save_at=None)
    return z, log_det

################################################################################################################

if __name__ == '__main__':
  from debug import *
  import matplotlib.pyplot as plt
  from generax.flows.base import Sequential

  key = random.PRNGKey(0)
  x = random.normal(key, shape=(10, 2, 2, 2))

  # composition = Sequential(ShiftScale,
  #                          ShiftScale,
  #                          ShiftScale,
  #                          x=x,
  #                          key=key)
  # z, log_det = eqx.filter_vmap(composition)(x)

  # import pdb; pdb.set_trace()

  layer = PLUAffine(x=x, key=key)
  z, log_det = eqx.filter_vmap(layer)(x)

  z, log_det = layer(x[0])
  x_reconstr, log_det2 = layer(z, inverse=True)

  G = jax.jacobian(lambda x: layer(x)[0])(x[0])
  G = einops.rearrange(G, 'h1 w1 c1 h2 w2 c2 -> (h1 w1 c1) (h2 w2 c2)')
  log_det_true = jnp.linalg.slogdet(G)[1]

  assert jnp.allclose(log_det, log_det_true)
  assert jnp.allclose(x[0], x_reconstr)


  import pdb; pdb.set_trace()



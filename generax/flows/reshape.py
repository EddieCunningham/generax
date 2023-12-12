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
from generax.flows.base import *
import numpy as np

__all__ = ['Reverse',
           'Slice']

class Reverse(BijectiveTransform):
  """Reverse an input
  """

  def __init__(self,
               input_shape: Tuple[int],
               key: PRNGKeyArray,
               **kwargs):
    """**Arguments**:

    - `input_shape`: The input shape.  Output size is the same as shape.
    - `key`: A `jax.random.PRNGKey` for initialization
    """
    super().__init__(input_shape=input_shape,
                     **kwargs)

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
    The transformed input and 0
    """
    assert x.shape == self.input_shape
    z = x[..., ::-1]
    log_det = jnp.array(0.0)
    return z, log_det

class Slice(InjectiveTransform):
  """Slice an input to reduce the dimension
  """

  def __init__(self,
               input_shape: Tuple[int],
               output_shape: Tuple[int],
               key: PRNGKeyArray,
               **kwargs):
    """**Arguments**:

    - `input_shape`: The input shape.  Output size is the same as shape.
    - `key`: A `jax.random.PRNGKey` for initialization
    """
    assert input_shape[:-1] == output_shape[:-1], 'Need to keep the same spatial dimensions'
    super().__init__(input_shape=input_shape,
                     output_shape=output_shape,
                     **kwargs)

  def __call__(self,
               x: Array,
               y: Optional[Array] = None,
               inverse: bool = False,
               **kwargs) -> Array:
    """**Arguments**:

    - `x`: The input to the transformation
    - `y`: The conditioning information
    - `inverse`: Whether to inverse the transformation

    **Returns**:
    (z, log_det)
    """
    if inverse == False:
      assert x.shape == self.input_shape, 'Only works on unbatched data'
    else:
      assert x.shape == self.output_shape, 'Only works on unbatched data'

    if inverse == False:
      z = x[..., :self.output_shape[-1]]
    else:
      pad_shape = self.input_shape[:-1] + (self.input_shape[-1] - self.output_shape[-1],)
      z = jnp.concatenate([x, jnp.zeros(pad_shape)], axis=-1)

    log_det = jnp.array(0.0)
    return z, log_det

  def log_determinant(self,
                      z: Array,
                      **kwargs) -> Array:
    """Compute -0.5*log(det(J^TJ))

    **Arguments**:

    - `z`: An element of the base space

    **Returns**:
    The log determinant of (J^TJ)^0.5
    """
    return jnp.array(0.0)

################################################################################################################

if __name__ == '__main__':
  from debug import *
  import matplotlib.pyplot as plt
  from generax.flows.base import Sequential
  # switch to x64
  jax.config.update("jax_enable_x64", True)

  key = random.PRNGKey(0)
  x, y = random.normal(key, shape=(2, 10, 2, 2, 5))

  layer = Slice(input_shape=x.shape[1:],
                          output_shape=(2, 2, 3),
                          key=key)
  x = eqx.filter_vmap(layer.project)(x)


  z, log_det = layer(x[0])
  x_reconstr, log_det2 = layer(z, inverse=True)

  G = jax.jacobian(lambda x: layer(x)[0])(x[0])
  G = einops.rearrange(G, 'h1 w1 c1 h2 w2 c2 -> (h1 w1 c1) (h2 w2 c2)')
  log_det_true = 0.5*jnp.linalg.slogdet(G@G.T)[1]

  assert jnp.allclose(log_det, log_det_true)
  assert jnp.allclose(x[0], x_reconstr)


  import pdb; pdb.set_trace()



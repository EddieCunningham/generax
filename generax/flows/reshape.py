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

__all__ = ['Flatten',
           'Reverse',
           'Checkerboard',
           'Squeeze',
           'Slice']

class Flatten(BijectiveTransform):
  """Flatten
  """

  def __init__(self,
               input_shape: Tuple[int],
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
               inverse: bool = False,
               **kwargs) -> Array:
    """**Arguments**:

    - `x`: The input to the transformation
    - `y`: The conditioning information
    - `inverse`: Whether to inverse the transformation

    **Returns**:
    The transformed input and 0
    """
    log_det = jnp.array(0.0)
    if inverse == False:
      assert x.shape == self.input_shape
      return x.ravel(), log_det
    else:
      return x.reshape(self.input_shape), log_det

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

class Checkerboard(BijectiveTransform):
  """Checkerboard pattern from https://arxiv.org/pdf/1605.08803.pdf"""

  output_shape: Tuple[int] = eqx.field(static=True)

  def __init__(self,
               input_shape: Tuple[int],
               **kwargs):
    """**Arguments**:

    - `input_shape`: The input shape.  Output size is the same as shape.
    """
    H, W, C = input_shape
    assert W%2 == 0, 'Need even width'
    super().__init__(input_shape=input_shape,
                     **kwargs)
    self.output_shape = (H, W//2, C*2)

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
    The transformed input and 0
    """
    if inverse == False:
      assert x.shape == self.input_shape
      z = einops.rearrange(x, 'h (w k) c -> h w (c k)', k=2)
    else:
      assert x.shape == self.output_shape
      z = einops.rearrange(x, 'h w (c k) -> h (w k) c', k=2)

    log_det = jnp.array(0.0)
    return z, log_det

class Squeeze(BijectiveTransform):
  """Space to depth.  (H, W, C) -> (H//2, W//2, C*4)"""

  output_shape: Tuple[int] = eqx.field(static=True)

  def __init__(self,
               input_shape: Tuple[int],
               **kwargs):
    """**Arguments**:

    - `input_shape`: The input shape.  Output size is the same as shape.
    - `key`: A `jax.random.PRNGKey` for initialization
    """
    H, W, C = input_shape
    assert H % 2 == 0, 'Need even height'
    assert W % 2 == 0, 'Need even width'
    super().__init__(input_shape=input_shape,
                     **kwargs)
    self.output_shape = (H//2, W//2, C*4)

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
    The transformed input and 0
    """
    if inverse == False:
      assert x.shape == self.input_shape
      z = einops.rearrange(x, '(h m) (w n) c -> h w (c m n)', m=2, n=2)
    else:
      assert x.shape == self.output_shape
      z = einops.rearrange(x, 'h w (c m n) -> (h m) (w n) c', m=2, n=2)

    log_det = jnp.array(0.0)
    return z, log_det

class Slice(InjectiveTransform):
  """Slice an input to reduce the dimension
  """

  def __init__(self,
               input_shape: Tuple[int],
               output_shape: Tuple[int],
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
  x, y = random.normal(key, shape=(2, 10, 8, 8, 3))

  layer = Squeeze(input_shape=x.shape[1:],
                       key=key)
  layer_inv = layer.get_inverse()

  z, log_det = layer(x[0])
  x_reconstr, log_det2 = layer(z, inverse=True)
  check, log_det3 = layer_inv(z)

  G = jax.jacobian(lambda x: layer(x)[0])(x[0])
  if x.ndim > 2:
    G = einops.rearrange(G, 'h1 w1 c1 h2 w2 c2 -> (h1 w1 c1) (h2 w2 c2)')
  log_det_true = 0.5*jnp.linalg.slogdet(G@G.T)[1]

  assert jnp.allclose(log_det, log_det_true)
  assert jnp.allclose(log_det, log_det2)
  assert jnp.allclose(log_det, log_det3)
  assert jnp.allclose(x[0], x_reconstr)
  assert jnp.allclose(check, x_reconstr)


  import pdb; pdb.set_trace()



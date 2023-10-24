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
from generax.flow.base import BijectiveTransform
import numpy as np

__all__ = ['ShiftScale',
           'DenseLinear',
           'DenseAffine',
           'PLUAffine',]

class ShiftScale(BijectiveTransform):
  """This represents a shift and scale transformation.
  This is RealNVP https://arxiv.org/pdf/1605.08803.pdf when used
  in a coupling layer.


  **Attributes**:
  - `s_unbounded`: The unbounded scaling parameter.
  - `b`: The shift parameter.
  """

  s_unbounded: Array
  b: Array

  def __init__(self,
               *_,
               x: Array,
               y: Optional[Array] = None,
               key: PRNGKeyArray,
               **kwargs):
    """**Arguments**:

    - `x`: A JAX array with shape `shape`. This is *required*
           to be batched!
    - `y`: A JAX array with shape `shape` representing conditioning
          information.  Should also be batched.
    - `key`: A `jax.random.PRNGKey` for initialization
    """
    super().__init__(x=x,
                     y=y,
                     key=key,
                     **kwargs)

    assert x.shape[1:] == self.input_shape
    mean, std = util.mean_and_std(x, axis=0)
    std += 1e-4

    # Initialize the parameters so that z will have
    # zero mean and unit variance
    self.b = mean
    self.s_unbounded = std - 1/std

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

    # s must be strictly positive
    s = util.square_plus(self.s_unbounded, gamma=1.0) + 1e-4
    log_s = jnp.log(s)

    if inverse == False:
      z = (x - self.b)/s
    else:
      z = x*s + self.b

    log_det = -log_s.sum()
    return z, log_det

################################################################################################################

class DenseLinear(BijectiveTransform):
  """Multiply the last axis by a dense matrix.  When applied to images,
  this is GLOW https://arxiv.org/pdf/1807.03039.pdf

  **Attributes**:
  - `W`: The weight matrix
  """

  W: Array

  def __init__(self,
               *_,
               x: Array,
               y: Optional[Array] = None,
               key: PRNGKeyArray,
               **kwargs):
    """**Arguments**:

    - `x`: A JAX array with shape `shape`. This is *required*
           to be batched!
    - `y`: A JAX array with shape `shape` representing conditioning
    - `key`: A `jax.random.PRNGKey` for initialization
    """
    super().__init__(x=x,
                     y=y,
                     key=key,
                     **kwargs)
    dim = self.input_shape[-1]
    self.W = random.normal(key, shape=(dim, dim))
    self.W = util.whiten(self.W)

  def __call__(self,
               x: Array,
               y: Optional[Array] = None,
               inverse: bool = False) -> Array:
    """**Arguments**:

    - `x`: The input to the transformation
    - `y`: The conditioning information
    - `inverse`: Whether to inverse the transformation

    **Returns**:
    (z, log_det)
    """
    assert x.shape == self.input_shape, 'Only works on unbatched data'

    if inverse == False:
      z = jnp.einsum('ij,...j->...i', self.W, x)
    else:
      W_inv = jnp.linalg.inv(self.W)
      z = jnp.einsum('ij,...j->...i', W_inv, x)

    # Need to multiply the log determinant by the number of times
    # that we're applying the transformation.
    if len(self.input_shape) > 1:
      dim_mult = np.prod(self.input_shape[:-1])
    else:
      dim_mult = 1
    log_det = jnp.linalg.slogdet(self.W)[1]*dim_mult
    return z, log_det

################################################################################################################

class DenseAffine(BijectiveTransform):
  """Multiply the last axis by a dense matrix.  When applied to images,
  this is GLOW https://arxiv.org/pdf/1807.03039.pdf

  **Attributes**:
  - `W`: The weight matrix
  - `b`: The bias vector
  """

  W: DenseLinear
  b: Array

  def __init__(self,
               *_,
               x: Array,
               y: Optional[Array] = None,
               key: PRNGKeyArray,
               **kwargs):
    """**Arguments**:

    - `x`: A JAX array with shape `shape`. This is *required*
           to be batched!
    - `y`: A JAX array with shape `shape` representing conditioning
    - `key`: A `jax.random.PRNGKey` for initialization
    """
    super().__init__(x=x,
                     y=y,
                     key=key,
                     **kwargs)
    # Return mean to 0
    self.b = -jnp.mean(x, axis=0)

    self.W = DenseLinear(x=x - self.b,
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
    (z, log_det)
    """
    assert x.shape == self.input_shape, 'Only works on unbatched data'

    if inverse == False:
      z, log_det = self.W(x, y=y, inverse=False)
      z = z + self.b
    else:
      x = x - self.b
      z, log_det = self.W(x, y=y, inverse=True)

    return z, log_det

################################################################################################################

tri_solve = jax.scipy.linalg.solve_triangular
L_solve = partial(tri_solve, lower=True, unit_diagonal=True)
U_solve = partial(tri_solve, lower=False, unit_diagonal=True)
U_solve_with_diag = partial(tri_solve, lower=False, unit_diagonal=False)

class PLUAffine(BijectiveTransform):
  """Multiply the last axis by a matrix that is parametrized using the LU decomposition.  This is more efficient
  than the dense parametrization

  **Attributes**:
  - `A`: The weight matrix components.  The top half is the upper triangular matrix, and the bottom half is the
          lower triangular matrix and the diagonal is ignored.
  - `b`: The bias vector
  """

  A: Array
  b: Array

  def __init__(self,
               *_,
               x: Array,
               y: Optional[Array] = None,
               key: PRNGKeyArray,
               **kwargs):
    """**Arguments**:

    - `x`: A JAX array with shape `shape`. This is *required*
           to be batched!
    - `y`: A JAX array with shape `shape` representing conditioning
    - `key`: A `jax.random.PRNGKey` for initialization
    """
    super().__init__(x=x,
                     y=y,
                     key=key,
                     **kwargs)

    # Return mean to 0
    self.b = -jnp.mean(x, axis=0)

    # Initialize so that this will be approximately the identity matrix
    dim = x.shape[-1]
    self.A = random.normal(key, shape=(dim, dim))*0.01
    self.A = self.A.at[jnp.arange(dim),jnp.arange(dim)].set(1.0)

  def __call__(self,
               x: Array,
               y: Optional[Array] = None,
               inverse: bool=False) -> Array:
    """**Arguments**:

    - `x`: The input to the transformation
    - `y`: The conditioning information
    - `inverse`: Whether to inverse the transformation

    **Returns**:
    (z, log_det)
    """
    assert x.shape == self.input_shape, 'Only works on unbatched data'

    dim = x.shape[-1]
    mask = jnp.ones((dim, dim), dtype=bool)
    upper_mask = jnp.triu(mask)
    lower_mask = jnp.tril(mask, k=-1)

    if inverse == False:
      z = jnp.einsum("ij,...j->...i", self.A*upper_mask, x + self.b)
      z = jnp.einsum("ij,...j->...i", self.A*lower_mask, z) + z
    else:
      # vmap in order to handle images
      L_solve_vmap = L_solve
      U_solve_vmap = U_solve_with_diag
      for _ in x.shape[:-1]:
        L_solve_vmap = jax.vmap(L_solve_vmap, in_axes=(None, 0))
        U_solve_vmap = jax.vmap(U_solve_vmap, in_axes=(None, 0))
      z = L_solve_vmap(self.A*lower_mask, x)
      z = U_solve_vmap(self.A*upper_mask, z) - self.b

    log_det = jnp.log(jnp.abs(jnp.diag(self.A))).sum()*util.list_prod(x.shape[:-1])
    return z, log_det

################################################################################################################

if __name__ == '__main__':
  from debug import *
  import matplotlib.pyplot as plt
  from generax.flow.base import Sequential

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



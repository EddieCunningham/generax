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

__all__ = ['ShiftScale',
           'DenseLinear',
           'DenseAffine',
           'CaleyOrthogonalMVP',
           'PLUAffine',
           'ConditionalOptionalTransport']

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
               input_shape: Tuple[int],
               key: PRNGKeyArray,
               **kwargs):
    """**Arguments**:

    - `input_shape`: The input shape.  Output size is the same as shape.
    - `key`: A `jax.random.PRNGKey` for initialization
    """
    super().__init__(input_shape=input_shape,
                     **kwargs)

    # Initialize the parameters randomly
    self.s_unbounded, self.b = random.normal(key, shape=(2,) + input_shape)

  def data_dependent_init(self,
                          x: Array,
                          y: Optional[Array] = None,
                          key: PRNGKeyArray = None) -> BijectiveTransform:
    """Initialize the parameters of the layer based on the data.

    **Arguments**:

    - `x`: The data to initialize the parameters with.
    - `y`: The conditioning information
    - `key`: A `jax.random.PRNGKey` for initialization

    **Returns**:
    A new layer with the parameters initialized.
    """
    assert x.shape[1:] == self.input_shape, 'x must be batched'
    mean, std = misc.mean_and_std(x, axis=0)
    std += 1e-4

    # Initialize the parameters so that z will have
    # zero mean and unit variance
    b = mean
    s_unbounded = std - 1/std

    # Turn the new parameters into a new module
    get_b = lambda tree: tree.b
    get_s_unbounded = lambda tree: tree.s_unbounded
    updated_layer = eqx.tree_at(get_b, self, b)
    updated_layer = eqx.tree_at(get_s_unbounded, updated_layer, s_unbounded)

    return updated_layer

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

    # s must be strictly positive
    s = misc.square_plus(self.s_unbounded, gamma=1.0) + 1e-4
    log_s = jnp.log(s)

    if inverse == False:
      z = (x - self.b)/s
    else:
      z = x*s + self.b

    if inverse == False:
      log_det = -log_s.sum()
    else:
      log_det = log_s.sum()

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
               input_shape: Tuple[int],
               key: PRNGKeyArray,
               **kwargs):
    """**Arguments**:

    - `input_shape`: The input shape.  Output size is the same as shape.
    - `key`: A `jax.random.PRNGKey` for initialization
    """
    super().__init__(input_shape=input_shape,
                     **kwargs)

    dim = self.input_shape[-1]
    self.W = random.normal(key, shape=(dim, dim))
    self.W = misc.whiten(self.W)

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

    if inverse:
      log_det *= -1

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
               input_shape: Tuple[int],
               key: PRNGKeyArray,
               **kwargs):
    """**Arguments**:

    - `input_shape`: The input shape.  Output size is the same as shape.
    - `key`: A `jax.random.PRNGKey` for initialization
    """
    super().__init__(input_shape=input_shape,
                     **kwargs)

    self.W = DenseLinear(input_shape=input_shape,
                         key=key,
                         **kwargs)
    self.b = jnp.zeros(input_shape)

  def data_dependent_init(self,
                          x: Array,
                          y: Optional[Array] = None,
                          key: PRNGKeyArray = None) -> BijectiveTransform:
    """Initialize the parameters of the layer based on the data.

    **Arguments**:

    - `x`: The data to initialize the parameters with.
    - `y`: The conditioning information
    - `key`: A `jax.random.PRNGKey` for initialization

    **Returns**:
    A new layer with the parameters initialized.
    """
    assert x.shape[1:] == self.input_shape, 'x must be batched'
    b = -jnp.mean(x, axis=0)
    return eqx.tree_at(lambda tree: tree.b, self, b)

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
    (z, log_det)
    """
    assert x.shape == self.input_shape, 'Only works on unbatched data'

    if inverse == False:
      x = x + self.b
      z, log_det = self.W(x, y=y, inverse=False)
    else:
      z, log_det = self.W(x, y=y, inverse=True)
      z = z - self.b
    return z, log_det

################################################################################################################

class CaleyOrthogonalMVP(BijectiveTransform):
  """Caley transform parametrization of an orthogonal matrix. This performs
  a matrix vector product with an orthogonal matrix.

  **Attributes**:
  - `W`: The weight matrix
  - `b`: The bias vector
  """

  W: Array
  b: Array

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

    dim = self.input_shape[-1]
    self.W = random.normal(key, shape=(dim, dim))
    self.b = jnp.zeros(input_shape)

  def data_dependent_init(self,
                          x: Array,
                          y: Optional[Array] = None,
                          key: PRNGKeyArray = None) -> BijectiveTransform:
    assert x.shape[1:] == self.input_shape, 'x must be batched'
    b = -jnp.mean(x, axis=0)
    return eqx.tree_at(lambda tree: tree.b, self, b)

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
    (z, log_det)
    """
    assert x.shape == self.input_shape, 'Only works on unbatched data'

    A = self.W - self.W.T
    dim = self.input_shape[-1]

    # So that we can multiply with channel dim of images
    @partial(jnp.vectorize, signature='(i,j),(j)->(i)')
    def matmul(A, x):
      return A@x

    if inverse == False:
      x += self.b
      IpA_inv = jnp.linalg.inv(jnp.eye(dim) + A)
      y = matmul(IpA_inv, x)
      z = y - matmul(A, y)
    else:
      ImA_inv = jnp.linalg.inv(jnp.eye(dim) - A)
      y = matmul(ImA_inv, x)
      z = y + matmul(A, y)
      z -= self.b

    log_det = jnp.zeros(1)
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
               input_shape: Tuple[int],
               key: PRNGKeyArray,
               **kwargs):
    """**Arguments**:

    - `input_shape`: The input shape.  Output size is the same as shape.
    - `key`: A `jax.random.PRNGKey` for initialization
    """
    super().__init__(input_shape=input_shape,
                     **kwargs)

    # Initialize so that this will be approximately the identity matrix
    dim = input_shape[-1]
    self.A = random.normal(key, shape=(dim, dim))*0.01
    self.A = self.A.at[jnp.arange(dim),jnp.arange(dim)].set(1.0)

    self.b = jnp.zeros(input_shape)

  def data_dependent_init(self,
                          x: Array,
                          y: Optional[Array] = None,
                          key: PRNGKeyArray = None) -> BijectiveTransform:
    """Initialize the parameters of the layer based on the data.

    **Arguments**:

    - `x`: The data to initialize the parameters with.
    - `y`: The conditioning information
    - `key`: A `jax.random.PRNGKey` for initialization

    **Returns**:
    A new layer with the parameters initialized.
    """
    assert x.shape[1:] == self.input_shape, 'x must be batched'
    b = -jnp.mean(x, axis=0)
    return eqx.tree_at(lambda tree: tree.b, self, b)

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
    (z, log_det)
    """
    assert x.shape == self.input_shape, 'Only works on unbatched data'

    dim = x.shape[-1]
    mask = jnp.ones((dim, dim), dtype=bool)
    upper_mask = jnp.triu(mask)
    lower_mask = jnp.tril(mask, k=-1)

    if inverse == False:
      x += self.b
      z = jnp.einsum("ij,...j->...i", self.A*upper_mask, x)
      z = jnp.einsum("ij,...j->...i", self.A*lower_mask, z) + z
    else:
      # vmap in order to handle images
      L_solve_vmap = L_solve
      U_solve_vmap = U_solve_with_diag
      for _ in x.shape[:-1]:
        L_solve_vmap = jax.vmap(L_solve_vmap, in_axes=(None, 0))
        U_solve_vmap = jax.vmap(U_solve_vmap, in_axes=(None, 0))
      z = L_solve_vmap(self.A*lower_mask, x)
      z = U_solve_vmap(self.A*upper_mask, z)
      z -= self.b

    log_det = jnp.log(jnp.abs(jnp.diag(self.A))).sum()*misc.list_prod(x.shape[:-1])
    if inverse:
      log_det *= -1
    return z, log_det

################################################################################################################

class ConditionalOptionalTransport(TimeDependentBijectiveTransform):
  """Given x1, compute f(t, x0) = t*x1 + (1-t)*x0.  This is the optimal transport
  map between the two points.  Used in flow matching https://arxiv.org/pdf/2210.02747.pdf

  Non-inverse mode goes t -> 0 while inverse mode goes t -> 1.

  **Attributes**:
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
               t: Array,
               x: Array,
               y: Optional[Array] = None,
               inverse: bool=False,
               **kwargs) -> Array:
    """**Arguments**:

    - `t`: The time point.
    - `x`: The input to the transformation
    - `y`: The conditioning information
    - `inverse`: Whether to invert the transformation (0 -> t)

    **Returns**:
    `(z, log_det)`
    """
    assert x.shape == self.input_shape, 'Only works on unbatched data'
    if y is None:
      raise ValueError(f'Expected a conditional input')
    if y.shape != x.shape:
      raise ValueError(f'Expected y.shape ({y.shape}) to match x.shape ({x.shape})')

    x1 = y
    if inverse:
      x0 = x
      xt = (1 - t)*x0 + t*x1
      log_det = jnp.log(1 - t)
      return xt, log_det
    else:
      xt = x
      x0 = (xt - t*x1)/(1 - t)
      log_det = -jnp.log(1 - t)
      return x0, log_det

  def vector_field(self,
                   t: Array,
                   xt: Array,
                   y: Optional[Array] = None,
                   **kwargs) -> Array:
    """The vector field that samples evolve on as t changes

    **Arguments**:

    - `t`: Time.
    - `x0`: A point in the base space.
    - `y`: The conditioning information.

    **Returns**:
    The vector field that samples evolve on at (t, x).
    """
    assert x.shape == self.input_shape, 'Only works on unbatched data'
    if y is None:
      raise ValueError(f'Expected a conditional input')
    if y.shape != x.shape:
      raise ValueError(f'Expected y.shape ({y.shape}) to match x.shape ({x.shape})')
    return y - x

################################################################################################################

if __name__ == '__main__':
  from debug import *
  import matplotlib.pyplot as plt
  from generax.flows.base import Sequential
  # switch to x64
  jax.config.update("jax_enable_x64", True)

  key = random.PRNGKey(0)
  x, y = random.normal(key, shape=(2, 10, 2, 2, 2))

  # layer = ShiftScale(input_shape=x.shape[1:],
  #                    key=key)
  layer = CaleyOrthogonalMVP(input_shape=x.shape[1:],
                             key=key)
  # layer = ConditionalOptionalTransport(input_shape=x.shape[1:],
  #                                      key=key)

  # x = random.normal(key, shape=(2, 2, 2, 2))
  # layer = layer.data_dependent_init(x, key=key)

  # layer = PLUAffine(x=x, key=key)

  z, log_det = layer(x[0])
  x_reconstr, log_det2 = layer(z, inverse=True)

  G = jax.jacobian(lambda x: layer(x)[0])(x[0])
  G = einops.rearrange(G, 'h1 w1 c1 h2 w2 c2 -> (h1 w1 c1) (h2 w2 c2)')
  log_det_true = jnp.linalg.slogdet(G)[1]

  assert jnp.allclose(log_det, log_det_true)
  assert jnp.allclose(x[0], x_reconstr)


  import pdb; pdb.set_trace()



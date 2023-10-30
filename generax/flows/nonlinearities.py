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
import generax.util as util

__all__ = ['Softplus',
           'GaussianCDF',
           'LogisticCDF',
           'LeakyReLU',
           'SneakyReLU',
           'SquarePlus',
           'SquareSigmoid',
           'SquareLogit',
           'SLog',
          #  'CartesianToSpherical',
          #  'SphericalToCartesian',
           ]

class Softplus(BijectiveTransform):

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
    `(z, log_det)`
    """
    assert x.shape == self.input_shape, 'Only works on unbatched data'

    if inverse == True:
      x = jnp.where(x < 0.0, 1e-5, x)
      dx = jnp.log1p(-jnp.exp(-x))
      z = x + dx
      log_det = dx.sum()
    else:
      z = jax.nn.softplus(x)
      log_det = jnp.log1p(-jnp.exp(-z)).sum()

    if inverse:
      log_det = -log_det

    return z, log_det

class GaussianCDF(BijectiveTransform):

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
    `(z, log_det)`
    """
    assert x.shape == self.input_shape, 'Only works on unbatched data'

    if inverse == False:
      z = jax.scipy.stats.norm.cdf(x)
      log_det = jax.scipy.stats.norm.logpdf(x).sum()
    else:
      z = jax.scipy.stats.norm.ppf(x)
      log_det = jax.scipy.stats.norm.logpdf(z).sum()

    if inverse:
      log_det = -log_det

    return z, log_det

class LogisticCDF(BijectiveTransform):

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
    `(z, log_det)`
    """
    assert x.shape == self.input_shape, 'Only works on unbatched data'

    if inverse == False:
      z = jax.scipy.stats.logistic.cdf(x)
      log_det = jax.scipy.stats.logistic.logpdf(x).sum()
    else:
      z = jax.scipy.stats.logistic.ppf(x)
      log_det = jax.scipy.stats.logistic.logpdf(z).sum()

    if inverse:
      log_det = -log_det

    return z, log_det

class LeakyReLU(BijectiveTransform):

  alpha: float

  def __init__(self,
               input_shape: Tuple[int],
               *,
               key: PRNGKeyArray,
               alpha: Optional[float] = 0.01,
               **kwargs):
    """**Arguments**:

    - `input_shape`: The input shape.  Output size is the same as shape.
    - `key`: A `jax.random.PRNGKey` for initialization
    """
    super().__init__(input_shape=input_shape,
                     **kwargs)
    self.alpha = alpha

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

    if inverse == False:
      z = jnp.where(x > 0, x, self.alpha*x)
    else:
      z = jnp.where(x > 0, x, x/self.alpha)

    log_dx_dz = jnp.where(x > 0, 0, jnp.log(self.alpha))
    log_det = log_dx_dz.sum()

    if inverse:
      log_det = -log_det

    return z, log_det


class SneakyReLU(BijectiveTransform):
  """ Originally from https://invertibleworkshop.github.io/INNF_2019/accepted_papers/pdfs/INNF_2019_paper_26.pdf
  """

  alpha: float

  def __init__(self,
               input_shape: Tuple[int],
               *,
               key: PRNGKeyArray,
               alpha: Optional[float] = 0.01,
               **kwargs):
    """**Arguments**:

    - `input_shape`: The input shape.  Output size is the same as shape.
    - `key`: A `jax.random.PRNGKey` for initialization
    """
    super().__init__(input_shape=input_shape,
                     **kwargs)
    # Sneaky ReLU uses a different convention
    self.alpha = (1.0 - alpha)/(1.0 + alpha)

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

    if inverse == False:
      sqrt_1px2 = jnp.sqrt(1 + x**2)
      z = (x + self.alpha*(sqrt_1px2 - 1))/(1 + self.alpha)
      log_det = jnp.log(1 + self.alpha*x/sqrt_1px2) - jnp.log(1 + self.alpha)
    else:
      alpha_sq = self.alpha**2
      b = (1 + self.alpha)*x + self.alpha
      z = (jnp.sqrt(alpha_sq*(1 + b**2 - alpha_sq)) - b)/(alpha_sq - 1)
      sqrt_1px2 = jnp.sqrt(1 + z**2)
      log_det = jnp.log(1 + self.alpha*z/sqrt_1px2) - jnp.log(1 + self.alpha)

    log_det = log_det.sum()

    if inverse:
      log_det = -log_det
    return z, log_det


class SquarePlus(BijectiveTransform):

  gamma: float

  def __init__(self,
               input_shape: Tuple[int],
               *,
               key: PRNGKeyArray,
               gamma: Optional[float] = 0.5,
               **kwargs):
    """**Arguments**:

    - `input_shape`: The input shape.  Output size is the same as shape.
    - `key`: A `jax.random.PRNGKey` for initialization
    """
    super().__init__(input_shape=input_shape,
                     **kwargs)
    self.gamma = gamma

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

    if inverse == False:
      sqrt_arg = x**2 + 4*self.gamma
      z = 0.5*(x + jnp.sqrt(sqrt_arg))
      z = jnp.maximum(z, 0.0)
      dzdx = 0.5*(1 + x*jax.lax.rsqrt(sqrt_arg)) # Always positive
      dzdx = jnp.maximum(dzdx, 1e-5)
    else:
      z = x - self.gamma/x
      dzdx = 0.5*(1 + z*jax.lax.rsqrt(z**2 + 4*self.gamma))

    log_det = jnp.log(dzdx).sum()

    if inverse:
      log_det = -log_det

    return z, log_det


class SquareSigmoid(BijectiveTransform):

  gamma: float

  def __init__(self,
               input_shape: Tuple[int],
               *,
               key: PRNGKeyArray,
               gamma: Optional[float] = 0.5,
               **kwargs):
    """**Arguments**:

    - `input_shape`: The input shape.  Output size is the same as shape.
    - `key`: A `jax.random.PRNGKey` for initialization
    """
    super().__init__(input_shape=input_shape,
                     **kwargs)
    self.gamma = gamma

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

    if inverse == False:
      rsqrt = jax.lax.rsqrt(x**2 + 4*self.gamma)
      z = 0.5*(1 + x*rsqrt)
    else:
      arg = 2*x - 1
      z = 2*jnp.sqrt(self.gamma)*arg*jax.lax.rsqrt(1 - arg**2)
      rsqrt = jax.lax.rsqrt(z**2 + 4*self.gamma)

    dzdx = 2*self.gamma*rsqrt**3
    log_det = jnp.log(dzdx).sum()

    if inverse:
      log_det = -log_det

    return z, log_det

class SquareLogit(SquareSigmoid):

  def __call__(self,
               x: Array,
               y: Optional[Array] = None,
               inverse: bool=False,
               **kwargs) -> Array:
    return super().__call__(x, y=y, inverse=not inverse, **kwargs)


class SLog(BijectiveTransform):
  """ https://papers.nips.cc/paper/2019/file/b1f62fa99de9f27a048344d55c5ef7a6-Paper.pdf
  """

  alpha: Union[float,None]

  def __init__(self,
               input_shape: Tuple[int],
               *,
               key: PRNGKeyArray,
               alpha: Optional[float] = 0.0,
               **kwargs):
    """**Arguments**:

    - `input_shape`: The input shape.  Output size is the same as shape.
    - `key`: A `jax.random.PRNGKey` for initialization
    """
    super().__init__(input_shape=input_shape,
                     **kwargs)
    self.alpha = alpha

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

    # Bound alpha to be positive
    alpha = misc.square_plus(self.alpha) + 1e-4

    if inverse == False:
      log_det = jnp.log1p(alpha*jnp.abs(x))
      z = jnp.sign(x)/alpha*log_det
    else:
      z = jnp.sign(x)/alpha*(jnp.exp(alpha*jnp.abs(x)) - 1)
      log_det = jnp.log1p(alpha*jnp.abs(z))

    log_det = -log_det.sum()

    if inverse:
      log_det = -log_det
    return z, log_det

class CartesianToSpherical(BijectiveTransform):

  def __init__(self,
               input_shape: Tuple[int],
               *,
               key: PRNGKeyArray,
               **kwargs):
    """**Arguments**:

    - `input_shape`: The input shape.  Output size is the same as shape.
    - `key`: A `jax.random.PRNGKey` for initialization
    """
    # TODO: Fix this
    assert 0, 'Has bug'
    super().__init__(input_shape=input_shape,
                     **kwargs)

  def forward_fun(self, x, eps=1e-5):
    r = jnp.linalg.norm(x)
    denominators = jnp.sqrt(jnp.cumsum(x[::-1]**2)[::-1])[:-1]
    cos_phi = x[:-1]/denominators

    # cos_phi = jnp.maximum(-1.0 + eps, cos_phi)
    # cos_phi = jnp.minimum(1.0 - eps, cos_phi)
    phi = jnp.arccos(cos_phi)

    last_value = jnp.where(x[-1] >= 0, phi[-1], 2*jnp.pi - phi[-1])
    phi = phi.at[-1].set(last_value)

    return jnp.concatenate([r, phi])

  def inverse_fun(self, x):
    r = x[:1]
    phi = x[1:]
    sin_prod = jnp.cumprod(jnp.sin(phi))
    first_part = jnp.concatenate([jnp.ones(r.shape), sin_prod])
    second_part = jnp.concatenate([jnp.cos(phi), jnp.ones(r.shape)])
    return r*first_part*second_part

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

    @partial(jnp.vectorize, signature='(n)->(n),(),(k)')
    def forward(x):
      z = self.forward_fun(x)
      r, phi = z[0], z[1:]
      return z, r, phi

    @partial(jnp.vectorize, signature='(n)->(n),(),(k)')
    def inverse(x):
      z = self.inverse_fun(x)
      r, phi = x[0], x[1:]
      return z, r, phi

    if inverse == False:
      z, r, phi = forward(x)
      # z = self.forward_fun(x)
      # r, phi = z[0], z[1:]
    else:
      z, r, phi = inverse(x)
      # z = self.inverse_fun(x)
      # r, phi = x[0], x[1:]

    n = util.list_prod(self.input_shape)
    n_range = jnp.arange(n - 2, -1, -1)
    log_abs_sin_phi = jnp.log(jnp.abs(jnp.sin(phi)))
    log_det = -(n - 1)*jnp.log(r) - jnp.sum(n_range*log_abs_sin_phi, axis=-1)
    log_det = log_det.sum()

    if inverse:
      log_det = -log_det
    return z, log_det

class SphericalToCartesian(CartesianToSpherical):

  def __call__(self,
               x: Array,
               y: Optional[Array] = None,
               inverse: bool=False,
               **kwargs) -> Array:
    return super().__call__(x, y=y, inverse=not inverse, **kwargs)

################################################################################################################

if __name__ == '__main__':
  from debug import *
  import matplotlib.pyplot as plt
  from generax.flows.base import Sequential
  # switch to x64
  jax.config.update("jax_enable_x64", True)

  key = random.PRNGKey(0)
  x = random.normal(key, shape=(10, 2, 2, 2))
  x = jax.nn.sigmoid(x)
  x = jnp.clip(x, 1e-4, 1.0 - 1e-4)

  layer = LogisticCDF(input_shape=x.shape[1:],
                             key=key)

  x = random.normal(key, shape=(2, 2, 2, 2))
  layer = layer.data_dependent_init(x, key=key)

  # layer = PLUAffine(x=x, key=key)
  z, log_det = eqx.filter_vmap(layer)(x)

  z, log_det = layer(x[0])
  x_reconstr, log_det2 = layer(z, inverse=True)
  assert jnp.allclose(log_det, -log_det2)

  G = jax.jacobian(lambda x: layer(x)[0])(x[0])
  G = einops.rearrange(G, 'h1 w1 c1 h2 w2 c2 -> (h1 w1 c1) (h2 w2 c2)')
  log_det_true = jnp.linalg.slogdet(G)[1]

  assert jnp.allclose(log_det, log_det_true)
  assert jnp.allclose(x[0], x_reconstr)


  import pdb; pdb.set_trace()


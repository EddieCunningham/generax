from collections.abc import Callable
from typing import Literal, Optional, Union, Tuple
import jax
import jax.random as random
from jaxtyping import Array, PRNGKeyArray
import equinox as eqx
import jax.numpy as jnp

__all__ = ['GradWrapper',
           'TimeDependentGradWrapper']

class GradWrapper(eqx.Module):
  """An easy wrapper around a function that computes the gradient of a scalar function."""

  net: eqx.Module
  input_shape: Tuple[int, ...]

  def __init__(self,
               net: eqx.Module):
    self.net = net
    self.input_shape = net.input_shape

  def data_dependent_init(self,
                          x: Array,
                          y: Optional[Array] = None,
                          key: PRNGKeyArray = None) -> eqx.Module:
    """Initialize the parameters of the layer based on the data.

    **Arguments**:

    - `x`: The data to initialize the parameters with.
    - `y`: The conditioning information
    - `key`: A `jax.random.PRNGKey` for initialization

    **Returns**:
    A new layer with the parameters initialized.
    """
    assert x.shape[1:] == self.input_shape, 'Only works on batched data'
    out = self.net(x, y=y, key=key)
    assert out.shape == (1,)

  def __call__(self,
               x: Array,
               y: Optional[Array] = None,
               **kwargs) -> Array:
    """**Arguments:**

    - `t`: A JAX array with shape `()`.
    - `x`: A JAX array with shape `(input_shape,)`.
    - `y`: A JAX array with shape `(cond_shape,)`.

    **Returns:**

    A JAX array with shape `(input_shape,)`.
    """
    assert x.shape == self.input_shape

    def net(x):
      net_out = self.net(x, y=y, **kwargs)
      if net_out.shape != (1,):
        raise ValueError(f'Expected net to return a scalar, but got {net_out.shape}')
      return net_out.ravel()

    return eqx.filter_grad(net)(x)

  @property
  def energy(self):
    return self.net

class TimeDependentGradWrapper(GradWrapper):
  """An easy wrapper around a function that computes the gradient of a scalar function."""

  def data_dependent_init(self,
                          t: Array,
                          x: Array,
                          y: Optional[Array] = None,
                          key: PRNGKeyArray = None) -> eqx.Module:
    """Initialize the parameters of the layer based on the data.

    **Arguments**:

    - `t`: The time to initialize the parameters with.
    - `x`: The data to initialize the parameters with.
    - `y`: The conditioning information
    - `key`: A `jax.random.PRNGKey` for initialization

    **Returns**:
    A new layer with the parameters initialized.
    """
    assert x.shape[1:] == self.input_shape, 'Only works on batched data'
    out = self.net(t, x, y=y, key=key)
    assert out.shape == (1,)

  def __call__(self,
               t: Array,
               x: Array,
               y: Optional[Array] = None,
               **kwargs) -> Array:
    """**Arguments:**

    - `t`: A JAX array with shape `()`.
    - `x`: A JAX array with shape `(input_shape,)`.
    - `y`: A JAX array with shape `(cond_shape,)`.

    **Returns:**

    A JAX array with shape `(input_shape,)`.
    """
    assert x.shape == self.input_shape

    def net(x):
      net_out = self.net(t, x, y=y, **kwargs)
      if net_out.shape != (1,):
        raise ValueError(f'Expected net to return a scalar, but got {net_out.shape}')
      return net_out[0]
    return eqx.filter_grad(net)(x)

  @property
  def energy(self):
    return self.net
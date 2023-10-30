from collections.abc import Callable
from typing import Literal, Optional, Union, Tuple
import jax
import jax.random as random
from jaxtyping import Array, PRNGKeyArray
import equinox as eqx
import jax.numpy as jnp
import generax.util as util

__all__ = ['WeightNormDense']

class WeightNormDense(eqx.Module):
  """Weight normalization parametrized linear layer
  https://arxiv.org/pdf/1602.07868.pdf
  """

  in_size: int = eqx.field(static=True)
  out_size: int = eqx.field(static=True)
  W: Array
  b: Array
  g: Array

  def __init__(self,
               in_size: int,
               out_size: int,
               key: PRNGKeyArray,
               **kwargs):
    super().__init__(**kwargs)

    self.in_size = in_size
    self.out_size = out_size
    self.W = random.normal(key, shape=(out_size, in_size))*0.05
    self.g = jnp.array(1.0)
    self.b = jnp.zeros(out_size)

  def data_dependent_init(self,
                          x: Array,
                          key: PRNGKeyArray = None) -> eqx.Module:
    """Initialize the parameters of the layer based on the data.

    **Arguments**:

    - `x`: The data to initialize the parameters with.
    - `key`: A `jax.random.PRNGKey` for initialization

    **Returns**:
    A new layer with the parameters initialized.
    """
    assert x.shape[-1] == self.in_size, 'Only works on batched data'

    # Initialize g and b.
    W = self.W*jax.lax.rsqrt((self.W**2).sum(axis=1))[:, None]
    x = jnp.einsum('ij,bj->bi', W, x)

    std = jnp.std(x.reshape((-1, x.shape[-1])), axis=0) + 1e-5
    g = 1/std

    x *= g

    mean = jnp.mean(x.reshape((-1, x.shape[-1])), axis=0)
    b = -mean

    # Turn the new parameters into a new module
    get_g = lambda tree: tree.g
    get_b = lambda tree: tree.b
    updated_layer = eqx.tree_at(get_g, self, g)
    updated_layer = eqx.tree_at(get_b, updated_layer, b)

    return updated_layer

  def __call__(self, x: Array, y: Array = None) -> Array:
    W = self.W*jax.lax.rsqrt((self.W**2).sum(axis=1))[:, None]
    x = self.g*(W@x) + self.b
    return x


class WeightNormConv(eqx.Module):
  """Weight normalization parametrized convolutional layer
  https://arxiv.org/pdf/1602.07868.pdf
  """

  input_shape: int = eqx.field(static=True)
  out_size: int = eqx.field(static=True)
  filter_shape: Tuple[int] = eqx.field(static=True)
  W: Array
  b: Array
  g: Array

  def __init__(self,
               input_shape: Tuple[int], # in_channels
               filter_shape: Tuple[int],
               out_size: int,
               key: PRNGKeyArray,
               **kwargs):
    super().__init__(**kwargs)
    H, W, C = input_shape

    self.input_shape = input_shape
    self.filter_shape = filter_shape
    self.out_size = out_size
    self.W = random.normal(key, shape=self.filter_shape + (C, out_size))*0.05
    self.g = jnp.array(1.0)
    self.b = jnp.zeros(out_size)

  def data_dependent_init(self,
                          x: Array,
                          y: Optional[Array] = None,
                          key: PRNGKeyArray = None) -> eqx.Module:
    """Initialize the parameters of the layer based on the data.

    **Arguments**:

    - `x`: The data to initialize the parameters with.
    - `key`: A `jax.random.PRNGKey` for initialization

    **Returns**:
    A new layer with the parameters initialized.
    """
    assert x.shape[1:] == self.input_shape, 'Only works on batched data'

    # Initialize g and b.
    W = self.W*jax.lax.rsqrt((self.W**2).sum(axis=(0, 1, 2)))[None,None,None,:]
    x = util.conv(W, x)

    std = jnp.std(x.reshape((-1, x.shape[-1])), axis=0) + 1e-5
    g = 1/std

    x *= g

    mean = jnp.mean(x.reshape((-1, x.shape[-1])), axis=0)
    b = -mean

    # Turn the new parameters into a new module
    get_g = lambda tree: tree.g
    get_b = lambda tree: tree.b
    updated_layer = eqx.tree_at(get_g, self, g)
    updated_layer = eqx.tree_at(get_b, updated_layer, b)

    return updated_layer

  def __call__(self, x: Array, y: Array = None) -> Array:
    W = self.W*jax.lax.rsqrt((self.W**2).sum(axis=(0, 1, 2)))[None,None,None,:]
    x = self.g*util.conv(W, x) + self.b
    return x

################################################################################################################

if __name__ == '__main__':
  from debug import *
  import matplotlib.pyplot as plt
  from generax.flows.base import Sequential

  key = random.PRNGKey(0)
  x, y = random.normal(key, shape=(2, 10, 4, 4, 3))
  y, cond_size = None, None

  layer = WeightNormConv(input_shape=x.shape[1:],
                         filter_shape=(3, 3),
                         out_size=5,
                         key=key)

  out = eqx.filter_vmap(layer)(x, y)

  layer = layer.data_dependent_init(x, y, key=key)
  out = eqx.filter_vmap(layer)(x, y)
  import pdb; pdb.set_trace()





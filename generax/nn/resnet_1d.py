from collections.abc import Callable
from typing import Literal, Optional, Union
import jax
import jax.random as random
from jaxtyping import Array, PRNGKeyArray
import equinox as eqx
import jax.numpy as jnp

__all__ = ['WeightNormDense', 'ResNet1d']

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
    assert x.shape[-1] == self.in_size, 'Only works on unbatched data'

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


class GatedResBlock(eqx.Module):
  """Gated residual block for 1d data."""
  linear_cond: Union[eqx.nn.Linear, None]
  linear1: WeightNormDense
  linear2: WeightNormDense

  activation: Callable
  in_size: int = eqx.field(static=True)
  hidden_size: int = eqx.field(static=True)
  cond_size: int = eqx.field(static=True)

  def __init__(
      self,
      in_size: int,
      hidden_size: int,
      cond_size: Optional[int] = None,
      activation: Callable = jax.nn.swish,
      *,
      key: PRNGKeyArray,
      **kwargs,
  ):
    """**Arguments**:

    - `in_size`: The input size.  Output size is the same as in_size.
    - `hidden_size`: The hidden layer size.
    - `cond_size`: The size of the conditioning information.
    - `activation`: The activation function after each hidden layer.
    - `key`: A `jax.random.PRNGKey` for initialization
    """
    super().__init__(**kwargs)

    out_size = in_size
    self.in_size = in_size
    self.hidden_size = hidden_size
    self.cond_size = cond_size
    self.activation = activation

    k1, k2, k3 = random.split(key, 3)

    # Initialize the conditioning parameters
    if cond_size is not None:
      self.linear_cond = WeightNormDense(in_size=cond_size,
                                         out_size=2*hidden_size,
                                         key=k1)
    else:
      self.linear_cond = None

    self.linear1 = WeightNormDense(in_size=in_size,
                                   out_size=hidden_size,
                                   key=k2)

    self.linear2 = WeightNormDense(in_size=hidden_size,
                                   out_size=2*out_size,
                                   key=k3)

  def data_dependent_init(self,
                          x: Array,
                          y: Array = None,
                          key: PRNGKeyArray = None) -> eqx.Module:
    """Initialize the parameters of the layer based on the data.

    **Arguments**:

    - `x`: The data to initialize the parameters with.
    - `y`: The conditioning information
    - `key`: A `jax.random.PRNGKey` for initialization

    **Returns**:
    A new layer with the parameters initialized.
    """
    assert x.shape[-1] == self.in_size, 'Only works on unbatched data'

    k1, k2, k3 = random.split(key, 3)

    # Initialize the conditioning parameters
    if y is not None:
      linear_cond = self.linear_cond.data_dependent_init(y, key=k1)
      h = eqx.filter_vmap(linear_cond)(y)
      shift, scale = jnp.split(h, 2, axis=-1)
    else:
      linear_cond = None

    # Linear + shift/scale + activation
    linear1 = self.linear1.data_dependent_init(x, key=k2)
    x = eqx.filter_vmap(linear1)(x)
    if y is not None:
      x = shift + x*(1 + scale)
    x = eqx.filter_vmap(self.activation)(x)

    # Linear + gate
    linear2 = self.linear2.data_dependent_init(x, key=k3)

    # Turn the new parameters into a new module
    get_linear_cond = lambda tree: tree.linear_cond
    get_linear1 = lambda tree: tree.linear1
    get_linear2 = lambda tree: tree.linear2

    updated_layer = eqx.tree_at(get_linear_cond, self, linear_cond)
    updated_layer = eqx.tree_at(get_linear1, updated_layer, linear1)
    updated_layer = eqx.tree_at(get_linear2, updated_layer, linear2)

    return updated_layer

  def __call__(self, x: Array, y: Array = None) -> Array:
    """**Arguments:**

    - `x`: A JAX array with shape `(in_size,)`.
    - `y`: A JAX array to condition on with shape `(cond_size,)`.

    **Returns:**
    A JAX array with shape `(in_size,)`.
    """
    # The conditioning input will shift/scale x
    if y is not None:
      h = self.linear_cond(y)
      shift, scale = jnp.split(h, 2, axis=-1)

    # Linear + shift/scale + activation
    x = self.linear1(x)
    if y is not None:
      x = shift + x*(1 + scale)
    x = self.activation(x)

    # Linear + gate
    x = self.linear2(x)
    a, b = jnp.split(x, 2, axis=-1)
    return a*jax.nn.sigmoid(b)

class ResNet1d(eqx.Module):
  """ResNet for 1d data"""

  n_blocks: int = eqx.field(static=True)
  blocks: tuple[GatedResBlock, ...]
  in_projection: eqx.nn.Linear
  out_projection: eqx.nn.Linear

  in_size: int = eqx.field(static=True)
  working_size: int = eqx.field(static=True)
  hidden_size: int = eqx.field(static=True)
  out_size: int = eqx.field(static=True)

  def __init__(
      self,
      in_size: int,
      working_size: int,
      hidden_size: int,
      out_size: int,
      n_blocks: int,
      cond_size: Optional[int] = None,
      activation: Callable = jax.nn.swish,
      *,
      key: PRNGKeyArray,
      **kwargs,
  ):
    """**Arguments**:

    - `in_size`: The input size.  Output size is the same as in_size.
    - `hidden_size`: The size of each hidden layer.
    - `out_size`: The output size.
    - `n_blocks`: The number of residual blocks.
    - `cond_size`: The size of the conditioning information.
    - `activation`: The activation function in each residual block.
    - `key`: A `jax.random.PRNGKey` for initialization.
    """
    super().__init__(**kwargs)
    self.n_blocks = n_blocks
    self.in_size = in_size
    self.working_size = working_size
    self.hidden_size = hidden_size
    self.out_size = out_size

    k1, k2, k3 = random.split(key, 3)

    self.in_projection = WeightNormDense(in_size=in_size,
                                         out_size=working_size,
                                         key=k1)

    def make_resblock(k):
      return GatedResBlock(in_size=working_size,
                           hidden_size=hidden_size,
                           cond_size=cond_size,
                           activation=activation,
                           key=k)

    keys = random.split(k2, n_blocks)
    self.blocks = eqx.filter_vmap(make_resblock)(keys)

    self.out_projection = WeightNormDense(in_size=working_size,
                                          out_size=out_size,
                                          key=k3)

  def data_dependent_init(self,
                          x: Array,
                          y: Array = None,
                          key: PRNGKeyArray = None) -> eqx.Module:
    """Initialize the parameters of the layer based on the data.

    **Arguments**:

    - `x`: The data to initialize the parameters with.
    - `y`: The conditioning information
    - `key`: A `jax.random.PRNGKey` for initialization

    **Returns**:
    A new layer with the parameters initialized.
    """
    assert x.shape[-1] == self.in_size, 'Only works on unbatched data'

    k1, k2, k3 = random.split(key, 3)

    # Input projection
    in_proj = self.in_projection.data_dependent_init(x, key=k1)
    x = eqx.filter_vmap(in_proj)(x)

    # Scan over the vmapped blocks
    params, state = eqx.partition(self.blocks, eqx.is_array)
    def scan_body(x, inputs):
      key, block_params = inputs
      block = eqx.combine(block_params, state)
      new_block = block.data_dependent_init(x, y, key=key)
      new_x = eqx.filter_vmap(new_block)(x, y)
      new_params, _ = eqx.partition(block, eqx.is_array)
      return new_x, new_params

    keys = random.split(k2, self.n_blocks)
    x, params = jax.lax.scan(scan_body, x, (keys, params))
    blocks = eqx.combine(params, state)

    out_proj = self.out_projection.data_dependent_init(x, key=k3)

    # Turn the new parameters into a new module
    get_in_proj = lambda tree: tree.in_projection
    get_blocks = lambda tree: tree.blocks
    get_out_proj = lambda tree: tree.out_projection

    updated_layer = eqx.tree_at(get_in_proj, self, in_proj)
    updated_layer = eqx.tree_at(get_blocks, updated_layer, blocks)
    updated_layer = eqx.tree_at(get_out_proj, updated_layer, out_proj)

    return updated_layer

  def __call__(self,
                x: Array,
                y: Optional[Array] = None) -> Array:
    """**Arguments:**

    - `t`: A JAX array with shape `()`.
    - `x`: A JAX array with shape `(in_size,)`.
    - `y`: A JAX array with shape `(cond_size,)`.

    **Returns:**

    A JAX array with shape `(in_size,)`.
    """
    assert x.ndim == 1

    # Input projection
    x = self.in_projection(x)

    # Resnet blocks
    dynamic, static = eqx.partition(self.blocks, eqx.is_array)

    def f(x, params):
        block = eqx.combine(params, static)
        return block(x, y), None

    out, _ = jax.lax.scan(f, x, dynamic)

    # Output projection
    out = self.out_projection(out)
    return out

################################################################################################################

if __name__ == '__main__':
  from debug import *
  import matplotlib.pyplot as plt
  from generax.flows.base import Sequential

  key = random.PRNGKey(0)
  x, y = random.normal(key, shape=(2, 10, 3))
  y, cond_size = None, None

  layer = GatedResBlock(in_size=x.shape[-1],
                        cond_size=cond_size,
                        hidden_size=10,
                        key=key)

  layer = ResNet1d(in_size=x.shape[-1],
                   working_size=8,
                   hidden_size=16,
                   out_size=5,
                   n_blocks=4,
                   cond_size=cond_size,
                   key=key)

  out = eqx.filter_vmap(layer)(x, y)

  layer = layer.data_dependent_init(x, y, key=key)
  out = eqx.filter_vmap(layer)(x, y)
  import pdb; pdb.set_trace()



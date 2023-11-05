from collections.abc import Callable
from typing import Literal, Optional, Union, Tuple
import jax
import jax.random as random
from jaxtyping import Array, PRNGKeyArray
import equinox as eqx
import jax.numpy as jnp
from generax.nn.layers import *
from generax.nn.resnet_blocks import GatedResBlock

__all__ = ['ResNet',
           'TimeDependentResNet']

class ResNet(eqx.Module):
  """ResNet for 1d data"""

  n_blocks: int = eqx.field(static=True)
  blocks: tuple[GatedResBlock, ...]
  in_projection: eqx.nn.Linear
  out_projection: eqx.nn.Linear

  input_shape: int = eqx.field(static=True)
  cond_shape: int = eqx.field(static=True)
  working_size: int = eqx.field(static=True)
  hidden_size: int = eqx.field(static=True)
  out_size: int = eqx.field(static=True)
  filter_shape: Union[Tuple[int],None] = eqx.field(static=True)

  def __init__(self,
               input_shape: Tuple[int],
               working_size: int,
               hidden_size: int,
               out_size: int,
               n_blocks: int,
               filter_shape: Optional[Tuple[int]] = (3, 3),
               cond_shape: Optional[Tuple[int]] = None,
               activation: Callable = jax.nn.swish,
               *,
               key: PRNGKeyArray,
               **kwargs):
    """**Arguments**:

    - `input_shape`: The input size.  Output size is the same as input_shape.
    - `working_size`: The size (channels for images) of each hidden layer.
    - `hidden_size`: The size (channels for images) of each hidden layer.
    - `out_size`: The output size.  For images, this is the number of output
                  channels.
    - `n_blocks`: The number of residual blocks.
    - `cond_shape`: The size of the conditioning information.
    - `activation`: The activation function in each residual block.
    - `key`: A `jax.random.PRNGKey` for initialization.
    """
    super().__init__(**kwargs)

    if len(input_shape) not in [1, 3]:
      raise ValueError(f'Expected 1d or 3d input shape')

    image = False
    if len(input_shape) == 3:
      H, W, C = input_shape
      image = True
      assert filter_shape is not None, 'Must pass in filter shape when processing images'

    self.n_blocks = n_blocks
    self.working_size = working_size
    self.hidden_size = hidden_size
    self.filter_shape = filter_shape
    self.out_size = out_size

    k1, k2, k3 = random.split(key, 3)

    if isinstance(input_shape, int):
      input_shape = (input_shape,)
    self.input_shape = input_shape
    self.cond_shape = cond_shape

    if image == False:
      self.in_projection = WeightNormDense(in_size=input_shape[0],
                                          out_size=working_size,
                                          key=k1)
      working_shape = (working_size,)
    else:
      self.in_projection = ConvAndGroupNorm(input_shape=input_shape,
                                        out_size=working_size,
                                        filter_shape=filter_shape,
                                        groups=1,
                                        key=k1)
      working_shape = (H, W, working_size)

    def make_resblock(k):
      return GatedResBlock(input_shape=working_shape,
                          hidden_size=hidden_size,
                          cond_shape=cond_shape,
                          activation=activation,
                          filter_shape=filter_shape,
                          key=k)

    keys = random.split(k2, n_blocks)
    self.blocks = eqx.filter_vmap(make_resblock)(keys)

    if image == False:
      self.out_projection = WeightNormDense(in_size=working_size,
                                            out_size=out_size,
                                            key=k3)
    else:
      self.out_projection = ConvAndGroupNorm(input_shape=working_shape,
                                           out_size=out_size,
                                           filter_shape=filter_shape,
                                           groups=1,
                                           key=k3)

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

from generax.nn.time_condition import TimeFeatures

class TimeDependentResNet(ResNet):
  """A time dependent version of a 1d resnet
  """

  time_features: TimeFeatures

  def __init__(self,
               input_shape: Tuple[int],
               working_size: int,
               hidden_size: int,
               out_size: int,
               n_blocks: int,
               filter_shape: Optional[Tuple[int]] = (3, 3),
               cond_shape: Optional[Tuple[int]] = None,
               activation: Callable = jax.nn.swish,
               embedding_size: Optional[int] = 16,
               out_features: int=8,
               *,
               key: PRNGKeyArray,
               **kwargs):
    k1, k2 = random.split(key, 2)
    self.time_features = TimeFeatures(embedding_size=embedding_size,
                                      out_features=out_features,
                                      key=k1,
                                      **kwargs)

    total_cond_size = out_features
    if cond_shape is not None:
      if len(cond_shape) != 1:
        raise ValueError(f'Expected 1d conditional input.')
      total_cond_size += cond_shape[0]

    super().__init__(input_shape=input_shape,
                     working_size=working_size,
                     hidden_size=hidden_size,
                     out_size=out_size,
                     n_blocks=n_blocks,
                     filter_shape=filter_shape,
                     cond_shape=(total_cond_size,),
                     activation=activation,
                     key=k2,
                     **kwargs)

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
    assert t.ndim == 1
    h = eqx.filter_vmap(self.time_features)(t)
    if y is not None:
      h = jnp.concatenate([h, y], axis=-1)
    return super().data_dependent_init(x, y=h, key=key)

  def __call__(self,
               t: Array,
               x: Array,
               y: Optional[Array] = None,
               **kwargs) -> Array:
    assert t.shape == ()

    h = self.time_features(t)
    if y is not None:
      h = jnp.concatenate([h, y], axis=-1)

    return super().__call__(x, y=h)

################################################################################################################

if __name__ == '__main__':
  from debug import *
  import matplotlib.pyplot as plt
  from generax.flows.base import Sequential

  key = random.PRNGKey(0)
  x, y = random.normal(key, shape=(2, 10, 5, 5, 3))
  cond_shape = y.shape[1:]
  y, cond_shape = None, None

  layer = ResNet(input_shape=x.shape[1:],
                 working_size=8,
                 hidden_size=16,
                 out_size=5,
                 n_blocks=4,
                 cond_shape=cond_shape,
                 filter_shape=(3, 3),
                 key=key)

  out = eqx.filter_vmap(layer)(x, y)

  layer = layer.data_dependent_init(x, y, key=key)
  out = eqx.filter_vmap(layer)(x, y)
  import pdb; pdb.set_trace()



from collections.abc import Callable
from typing import Literal, Optional, Union, Tuple
import jax
import jax.random as random
from jaxtyping import Array, PRNGKeyArray
import equinox as eqx
import jax.numpy as jnp
from generax.nn.layers import *

__all__ = ['GatedResBlock',
           'Block',
           'ImageResBlock']

################################################################################################################

class GatedResBlock(eqx.Module):
  """Gated residual block for 1d data or images."""
  linear_cond: Union[Union[WeightNormDense,ConvAndGroupNorm], None]
  linear1: Union[WeightNormDense,ConvAndGroupNorm]
  linear2: Union[WeightNormDense,ConvAndGroupNorm]

  activation: Callable
  input_shape: Tuple[int] = eqx.field(static=True)
  hidden_size: int = eqx.field(static=True)
  cond_shape: Tuple[int] = eqx.field(static=True)
  filter_shape: Union[Tuple[int],None] = eqx.field(static=True)
  groups: Union[int,None] = eqx.field(static=True)

  def __init__(self,
               input_shape: Tuple[int],
               hidden_size: int,
               groups: Optional[int] = None,
               filter_shape: Optional[Tuple[int]] = None,
               cond_shape: Optional[Tuple[int]] = None,
               activation: Callable = jax.nn.swish,
               *,
               key: PRNGKeyArray,
               **kwargs):
    """**Arguments**:

    - `input_shape`: The input size.  Output size is the same as `input_shape`.
    - `hidden_size`: The hidden layer size.
    - `cond_shape`: The size of the conditioning information.
    - `activation`: The activation function after each hidden layer.
    - `key`: A `jax.random.PRNGKey` for initialization
    """
    super().__init__(**kwargs)

    if len(input_shape) not in [1, 3]:
      raise ValueError(f'Expected 1d or 3d input shape')

    image = False
    if len(input_shape) == 3:
      H, W, C = input_shape
      image = True
      assert filter_shape is not None, 'Must pass in filter shape when processing images'

    self.input_shape = input_shape
    self.hidden_size = hidden_size
    self.cond_shape = cond_shape
    self.filter_shape = filter_shape
    self.activation = activation

    if groups is not None:
      assert image
      if hidden_size % groups != 0:
        raise ValueError(f'Hidden size must be divisible by groups')
    self.groups = groups

    k1, k2, k3 = random.split(key, 3)

    # Initialize the conditioning parameters
    if cond_shape is not None:
        if len(cond_shape) == 1:
          self.linear_cond = WeightNormDense(in_size=cond_shape[0],
                                             out_size=2*hidden_size,
                                             key=k1)
        else:
          self.linear_cond = ConvAndGroupNorm(input_shape=cond_shape,
                                            out_size=2*hidden_size,
                                            filter_shape=filter_shape,
                                            groups=groups,
                                            key=k1)
    else:
      self.linear_cond = None

    if image:
      self.linear1 = ConvAndGroupNorm(input_shape=input_shape,
                                    out_size=hidden_size,
                                    filter_shape=filter_shape,
                                    groups=groups,
                                    key=k2)
      hidden_shape = (H, W, hidden_size)
      self.linear2 = WeightNormConv(input_shape=hidden_shape,
                                    out_size=2*C,
                                    filter_shape=filter_shape,
                                    key=k3)
    else:
      self.linear1 = WeightNormDense(in_size=input_shape[0],
                                    out_size=hidden_size,
                                    key=k2)

      self.linear2 = WeightNormDense(in_size=hidden_size,
                                    out_size=2*input_shape[0],
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
    assert x.shape[1:] == self.input_shape, 'Only works on batched data'

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

    - `x`: A JAX array with shape `input_shape`.
    - `y`: A JAX array to condition on with shape `cond_shape`.

    **Returns:**
    A JAX array with shape `input_shape`.
    """
    assert x.shape == self.input_shape, 'Only works on unbatched data'
    x_in = x
    # The conditioning input will shift/scale x
    if y is not None:
      h = self.linear_cond(self.activation(y))
      shift, scale = jnp.split(h, 2, axis=-1)

    # Linear + shift/scale + activation
    x = self.linear1(x)
    if y is not None:
      x = shift + x*(1 + scale)
    x = self.activation(x)

    # Linear + gate
    x = self.linear2(x)
    a, b = jnp.split(x, 2, axis=-1)
    return x_in + a*jax.nn.sigmoid(b)

################################################################################################################

class Block(eqx.Module):
  """Group norm, (shift+scale), activation, conv
  """
  input_shape: int = eqx.field(static=True)
  conv: WeightNormConv
  norm: eqx.nn.GroupNorm

  def __init__(self,
               input_shape: Tuple[int],
               out_size: int,
               groups: int,
               *,
               key: PRNGKeyArray,
               **kwargs):
    super().__init__(**kwargs)
    H, W, C = input_shape

    if C%groups != 0:
      raise ValueError("The number of groups must divide the number of channels.")

    self.norm = ChannelConvention(eqx.nn.GroupNorm(groups=groups, channels=C))
    self.conv = WeightNormConv(input_shape=input_shape,
                               filter_shape=(3, 3),
                               out_size=out_size,
                               key=key)
    self.input_shape = self.conv.input_shape

  def data_dependent_init(self,
                          x: Array,
                          y: Optional[Array] = None,
                          key: PRNGKeyArray = None) -> eqx.Module:
    return self

  def __call__(self,
               x: Array,
               y: Array = None,
               shift_scale: Optional[Array] = None) -> Array:
    assert x.shape == self.input_shape, 'Only works on unbatched data'
    H, W, C = self.input_shape

    h = self.norm(x)
    if shift_scale is not None:
      shift, scale = shift_scale
      h = shift + h*(1 + scale)

    h = jax.nn.silu(h)
    h = self.conv(h)
    return h

class ImageResBlock(eqx.Module):
  """Gated residual block for images."""
  linear_cond: Union[ConvAndGroupNorm, None]
  block1: Block
  block2: Block
  res_conv: Union[ConvAndGroupNorm,eqx.nn.Identity]
  gca: GatedGlobalContext

  input_shape: Tuple[int] = eqx.field(static=True)
  hidden_size: int = eqx.field(static=True)
  out_size: int = eqx.field(static=True)
  cond_shape: Tuple[int] = eqx.field(static=True)
  groups: Union[int,None] = eqx.field(static=True)

  def __init__(self,
               input_shape: Tuple[int],
               hidden_size: int,
               out_size: int,
               groups: Optional[int] = None,
               cond_shape: Optional[Tuple[int]] = None,
               *,
               key: PRNGKeyArray,
               **kwargs):
    """**Arguments**:

    - `input_shape`: The input size.  Output size is the same as `input_shape`.
    - `hidden_size`: The hidden layer size.
    - `cond_shape`: The size of the conditioning information.
    - `activation`: The activation function after each hidden layer.
    - `key`: A `jax.random.PRNGKey` for initialization
    """
    super().__init__(**kwargs)

    H, W, C = input_shape

    self.input_shape = input_shape
    self.hidden_size = hidden_size
    self.cond_shape = cond_shape
    self.out_size = out_size

    if hidden_size % groups != 0:
      raise ValueError(f'Hidden size must be divisible by groups')
    self.groups = groups

    k1, k2, k3, k4, k5 = random.split(key, 5)

    # Initialize the conditioning parameters
    if cond_shape is not None:
        if len(cond_shape) != 1:
          raise ValueError(f'Conditioning shape must be 1d')
        self.linear_cond = WeightNormDense(in_size=cond_shape[0],
                                           out_size=2*hidden_size,
                                           key=k1)
    else:
      self.linear_cond = None

    self.block1 = Block(input_shape=input_shape,
                        out_size=hidden_size,
                        groups=groups,
                        key=k2)
    self.block2 = Block(input_shape=(H, W, hidden_size),
                        out_size=out_size,
                        groups=groups,
                        key=k3)

    self.gca = GatedGlobalContext(input_shape=(H, W, out_size),
                                  key=k4)

    if out_size != C:
      self.res_conv = WeightNormConv(input_shape=input_shape,
                                     out_size=out_size,
                                     filter_shape=(3, 3),
                                     key=k5)
    else:
      self.res_conv = eqx.nn.Identity()

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
    return self

  def __call__(self, x: Array, y: Array = None) -> Array:
    """**Arguments:**

    - `x`: A JAX array with shape `input_shape`.
    - `y`: A JAX array to condition on with shape `cond_shape`.

    **Returns:**
    A JAX array with shape `input_shape`.
    """
    x_in = x

    h = self.block1(x)

    # The conditioning input will shift/scale x
    if y is not None:
      hh = self.linear_cond(jax.nn.silu(y))
      shift_scale = jnp.split(hh, 2, axis=-1)
    else:
      shift_scale = None

    h = self.block2(h, shift_scale=shift_scale)
    h = self.gca(h)
    return self.res_conv(x_in) + h

################################################################################################################

if __name__ == '__main__':
  from debug import *
  import matplotlib.pyplot as plt
  from generax.flows.base import Sequential

  key = random.PRNGKey(0)
  x, y = random.normal(key, shape=(2, 10, 5, 5, 16))
  y = random.normal(key, (x.shape[0], 13))
  cond_shape = y.shape[1:]

  layer = ImageResBlock(input_shape=x.shape[1:],
                        cond_shape=cond_shape,
                        hidden_size=16,
                        out_size=5,
                        groups=4,
                        key=key)

  out = eqx.filter_vmap(layer)(x, y)

  layer = layer.data_dependent_init(x, y, key=key)
  out = eqx.filter_vmap(layer)(x, y)
  import pdb; pdb.set_trace()



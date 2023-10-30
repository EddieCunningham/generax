from collections.abc import Callable
from typing import Literal, Optional, Union, Tuple
import jax
import jax.random as random
from jaxtyping import Array, PRNGKeyArray
import equinox as eqx
import jax.numpy as jnp
from generax.nn.layers import WeightNormDense, WeightNormConv

__all__ = ['GatedResBlock']

################################################################################################################

class GatedResBlock(eqx.Module):
  """Gated residual block for 1d data or images."""
  linear_cond: Union[Union[WeightNormDense,WeightNormConv], None]
  linear1: Union[WeightNormDense,WeightNormConv]
  linear2: Union[WeightNormDense,WeightNormConv]

  activation: Callable
  input_shape: Tuple[int] = eqx.field(static=True)
  hidden_size: int = eqx.field(static=True)
  cond_shape: Tuple[int] = eqx.field(static=True)
  filter_shape: Union[Tuple[int],None] = eqx.field(static=True)

  def __init__(self,
               input_shape: Tuple[int],
               hidden_size: int,
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

    k1, k2, k3 = random.split(key, 3)

    # Initialize the conditioning parameters
    if cond_shape is not None:
        if len(cond_shape) == 1:
          self.linear_cond = WeightNormDense(in_size=cond_shape[0],
                                             out_size=2*hidden_size,
                                             key=k1)
        else:
          self.linear_cond = WeightNormConv(input_shape=cond_shape,
                                            out_size=2*hidden_size,
                                            filter_shape=filter_shape,
                                            key=k1)
    else:
      self.linear_cond = None

    if image:
      self.linear1 = WeightNormConv(input_shape=input_shape,
                                    out_size=hidden_size,
                                    filter_shape=filter_shape,
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

################################################################################################################

if __name__ == '__main__':
  from debug import *
  import matplotlib.pyplot as plt
  from generax.flows.base import Sequential

  key = random.PRNGKey(0)
  x, y = random.normal(key, shape=(2, 10, 5, 5, 3))
  cond_shape = y.shape[1:]
  y, cond_shape = None, None

  layer = GatedResBlock(input_shape=x.shape[1:],
                        cond_shape=cond_shape,
                        hidden_size=10,
                        filter_shape=(3, 3),
                        key=key)

  out = eqx.filter_vmap(layer)(x, y)

  layer = layer.data_dependent_init(x, y, key=key)
  out = eqx.filter_vmap(layer)(x, y)
  import pdb; pdb.set_trace()



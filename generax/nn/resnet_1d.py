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
  W: Array
  b: Array
  g: Array

  def __init__(self,
               *_,
               out_size: int,
               x: Array,
               y: Optional[Array] = None,
               key: PRNGKeyArray,
               **kwargs):
     in_size = x.shape[-1]
     self.W = random.normal(key, shape=(out_size, in_size))*0.05
     W = self.W*jax.lax.rsqrt((self.W**2).sum(axis=1))[:, None]
     x_in = x
     x = jnp.einsum('ij,bj->bi', W, x)

     std = jnp.std(x.reshape((-1, x.shape[-1])), axis=0) + 1e-5
     self.g = 1/std

     x *= self.g

     mean = jnp.mean(x.reshape((-1, x.shape[-1])), axis=0)
     self.b = -mean
     x += self.b

  def __call__(self, x: Array, y: Array = None) -> Array:
    W = self.W*jax.lax.rsqrt((self.W**2).sum(axis=1))[:, None]
    x = self.g*(W@x) + self.b
    return x


class ResBlock(eqx.Module):
    """Residual block"""
    linear_cond: Union[eqx.nn.Linear, None]
    linear1: WeightNormDense
    linear2: WeightNormDense

    activation: Callable
    hidden_size: int = eqx.field(static=True)

    def __init__(
        self,
        hidden_size: int,
        activation: Callable = jax.nn.swish,
        *,
        x: Array,
        y: Optional[Array] = None,
        key: PRNGKeyArray,
        **kwargs,
    ):
      """**Arguments**:

      - `in_size`: The input size.  Output size is the same as in_size.
      - `hidden_size`: The hidden layer size.
      - `activation`: The activation function after each hidden layer.
      - `key`: A `jax.random.PRNGKey` for initialization
      """
      super().__init__(**kwargs)

      out_size = x.shape[-1]
      self.hidden_size = hidden_size
      self.activation = activation

      k1, k2, k3 = random.split(key, 3)

      # Initialize the conditioning parameters
      if y is not None:
        self.linear_cond = WeightNormDense(out_size=2*hidden_size,
                                          x=y,
                                          y=None,
                                          key=k1)
        h = eqx.filter_vmap(self.linear_cond)(y)
        shift, scale = jnp.split(h, 2, axis=-1)
      else:
        self.linear_cond = None

      # Linear + shift/scale + activation
      self.linear1 = WeightNormDense(out_size=hidden_size,
                                     x=x,
                                     y=y,
                                     key=k2)
      x = eqx.filter_vmap(self.linear1)(x)
      if y is not None:
        x = shift + x*(1 + scale)
      x = eqx.filter_vmap(self.activation)(x)

      # Linear + gate
      self.linear2 = WeightNormDense(out_size=2*out_size,
                                     x=x,
                                     y=y,
                                     key=k3)

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
    blocks: tuple[ResBlock, ...]
    in_projection: eqx.nn.Linear
    out_projection: eqx.nn.Linear

    def __init__(
        self,
        working_size: int,
        hidden_size: int,
        out_size: int,
        n_blocks: int,
        activation: Callable = jax.nn.swish,
        *,
        x: Array,
        y: Optional[Array] = None,
        key: PRNGKeyArray,
        **kwargs,
    ):
      """**Arguments**:

      - `in_size`: The input size.  Output size is the same as in_size.
      - `hidden_size`: The size of each hidden layer.
      - `out_size`: The output size.
      - `n_blocks`: The number of residual blocks.
      - `activation`: The activation function in each residual block.
      - `x`: A JAX array with shape `(in_size,)`.
      - `y`: A JAX array with shape `(cond_size,)`.
      - `key`: A `jax.random.PRNGKey` for initialization.
      """
      super().__init__(**kwargs)
      self.n_blocks = n_blocks

      k1, k2, k3 = random.split(key, 3)

      # Input projection
      self.in_projection = WeightNormDense(out_size=working_size,
                                           x=x,
                                           y=y,
                                           key=k1)
      x = eqx.filter_vmap(self.in_projection)(x)

      # Initialize in a scan loop so that we can call scan later
      def scan_body(x, k):
        block = ResBlock(hidden_size=hidden_size,
                         activation=activation,
                         x=x,
                         y=y,
                         key=k)
        x = eqx.filter_vmap(block)(x, y)
        params, _ = eqx.partition(block, eqx.is_array)
        return x, params

      keys = random.split(k2, n_blocks)
      x, params = jax.lax.scan(scan_body, x, keys)

      dummy_block = ResBlock(hidden_size=hidden_size,
                            activation=activation,
                            x=x,
                            y=y,
                            key=k2)
      _, static = eqx.partition(dummy_block, eqx.is_array)
      blocks = eqx.combine(params, static)
      self.blocks = blocks

      self.out_projection = WeightNormDense(out_size=out_size,
                                            x=x,
                                            y=y,
                                            key=k3)

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

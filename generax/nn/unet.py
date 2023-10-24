from collections.abc import Callable
from typing import Literal, Optional, Union
import jax
import jax.random as random
from jaxtyping import Array, PRNGKeyArray
import equinox as eqx
import jax.numpy as jnp
from generax.nn.time_condition import TimeFeatures

class ResBlock(eqx.Module):
    """Residual block"""
    linear_cond: eqx.nn.Linear
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    activation: Callable
    in_size: int = eqx.field(static=True)
    cond_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)

    def __init__(
        self,
        in_size: int,
        cond_size: int,
        hidden_size: int,
        activation: Callable = jax.nn.swish,
        *,
        key: PRNGKeyArray,
        **kwargs,
    ):
      """**Arguments**:

      - `in_size`: The input size.  Output size is the same as in_size.
      - `cond_size`: The size of the conditioning input.
      - `hidden_size`: The hidden layer size.
      - `activation`: The activation function after each hidden layer.
      - `key`: A `jax.random.PRNGKey` for initialization
      """
      super().__init__(**kwargs)

      out_size = in_size
      self.in_size = in_size
      self.cond_size = cond_size
      self.hidden_size = hidden_size

      k1, k2, k3 = random.split(key, 3)
      self.linear_cond = eqx.nn.Linear(cond_size, 2*hidden_size, use_bias=True, key=k1)
      self.linear1 = eqx.nn.Linear(in_size, hidden_size, use_bias=True, key=k2)
      self.linear2 = eqx.nn.Linear(hidden_size, out_size, use_bias=True, key=k3)

      self.activation = activation

    def __call__(self, x: Array, cond: Array) -> Array:
      """**Arguments:**

      - `x`: A JAX array with shape `(in_size,)`.
      - `cond`: A JAX array to condition on with shape `(cond_size,)`.

      **Returns:**
      A JAX array with shape `(in_size,)`.
      """
      # The conditioning input will shift/scale x
      h = self.linear_cond(cond)
      shift, scale = jnp.split(h, 2, axis=-1)

      # Linear + norm + shift/scale + activation
      x = self.linear1(x)
      x = shift + x*(1 + scale)
      x = self.activation(x)

      # Linear + norm + activation
      x = self.linear2(x)
      x = self.activation(x)
      return x

class TimeDependentUNet(eqx.Module):
    """ResNet that is conditioned on time"""

    blocks: tuple[ResBlock, ...]
    n_blocks: int = eqx.field(static=True)
    time_features: TimeFeatures

    def __init__(
        self,
        in_size: int,
        hidden_size: int,
        n_blocks: int,
        time_embedding_size: int,
        activation: Callable = jax.nn.swish,
        *,
        key: PRNGKeyArray,
        **kwargs,
    ):
      """**Arguments**:

      - `in_size`: The input size.  Output size is the same as in_size.
      - `hidden_size`: The size of each hidden layer.
      - `n_blocks`: The number of residual blocks.
      - `time_embedding_size`: The size of the time embedding.
      - `activation`: The activation function in each residual block.
      - `key`: A `jax.random.PRNGKey` for initialization.
      """
      super().__init__(**kwargs)
      self.n_blocks = n_blocks

      cond_size = 4*time_embedding_size
      self.time_features = TimeFeatures(embedding_size=time_embedding_size,
                                        out_features=cond_size,
                                        key=key)

      keys = random.split(key, n_blocks)

      make_block = lambda k: ResBlock(in_size=in_size,
                                      cond_size=cond_size,
                                      hidden_size=hidden_size,
                                      activation=activation,
                                      key=k)
      self.blocks = eqx.filter_vmap(make_block)(keys)

    def __call__(self, t: Array, x: Array) -> Array:
      """**Arguments:**

      - `t`: A JAX array with shape `()`.
      - `x`: A JAX array with shape `(in_size,)`.

      **Returns:**

      A JAX array with shape `(in_size,)`.
      """
      t = jnp.array(t)
      assert t.shape == ()
      assert x.ndim == 1

      # Featurize the time
      t_emb = self.time_features(t)

      dynamic, static = eqx.partition(self.blocks, eqx.is_array)
      def f(x, params):
          block = eqx.combine(params, static)
          return block(x, t_emb), None

      out, _ = jax.lax.scan(f, x, dynamic)
      return out

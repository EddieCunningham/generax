from collections.abc import Callable
from typing import Literal, Optional, Union
import jax
import jax.random as random
from jaxtyping import Array, PRNGKeyArray
import equinox as eqx
import jax.numpy as jnp
from generax.nn.time_condition import TimeFeatures

class WeightNormDense(eqx.Module):
  """Weight normalization parametrized linear layer
  https://arxiv.org/pdf/1602.07868.pdf
  """
  W: Array
  b: Array
  g: Array

  def __init__(self,
               in_size: int,
               out_size: int,
               *
               x: Array,
               y: Optional[Array] = None,
               key: PRNGKeyArray,
               **kwargs):
     self.W = random.normal(key, shape=(out_size, in_size))*0.05
     W = self.W*jax.lax.rsqrt((self.W**2).sum(axis=1))[:,None]
     x = jnp.einsum("ij,bj->bi", W, x)

     std = jnp.std(x.reshape((-1, x.shape[-1])), axis=0) + 1e-5
     self.g = 1/std

     x *= self.g

     mean = jnp.mean(x.reshape((-1, x.shape[-1])), axis=0)
     self.b = -mean
     x += self.b

  def __call__(self, x: Array, cond: Array = None) -> Array:
    W = self.W*jax.lax.rsqrt((self.W**2).sum(axis=1))[:, None]
    x = self.g*jnp.einsum("ij,bj->bi", self.W, x) + self.b
    return x

class ResBlock(eqx.Module):
    """Residual block"""
    linear_cond: Union[eqx.nn.Linear, None]
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
        x: Array,
        y: Optional[Array] = None,
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

      self.activation = activation

      k1, k2, k3 = random.split(key, 3)
      # if cond_size is not None:
      #   self.linear_cond = eqx.nn.Linear(cond_size, 2*hidden_size, use_bias=True, key=k1)
      # else:
      #    self.linear_cond = None
      # self.linear1 = eqx.nn.Linear(in_size, hidden_size, use_bias=True, key=k2)
      # self.linear2 = eqx.nn.Linear(hidden_size, 2*out_size, use_bias=True, key=k3)

      if cond_size is not None:
        self.linear_cond = WeightNormDense(in_size=cond_size,
                                           out_size=2*hidden_size,
                                           x=y,
                                           y=None,
                                           key=k1)
        h = eqx.filter_vmap(self.linear_cond)(y)
        shift, scale = jnp.split(h, 2, axis=-1)
      else:
         self.linear_cond = None
      self.linear1 = WeightNormDense(in_size=in_size,
                                     out_size=hidden_size,
                                     x=x,
                                     y=y,
                                     key=k2)
      x = eqx.filter_vmap(self.linear1)(x)
      if cond_size is not None:
        x = shift + x*(1 + scale)
      x = eqx.filter_vmap(self.activation)(x)

      self.linear2 = WeightNormDense(in_size=hidden_size,
                                     out_size=2*out_size,
                                     x=x,
                                     y=y,
                                     key=k3)
      x = eqx.filter_vmap(self.linear2)(x)
      import pdb; pdb.set_trace()

    def __call__(self, x: Array, cond: Array = None) -> Array:
      """**Arguments:**

      - `x`: A JAX array with shape `(in_size,)`.
      - `cond`: A JAX array to condition on with shape `(cond_size,)`.

      **Returns:**
      A JAX array with shape `(in_size,)`.
      """
      if cond is not None:
        # The conditioning input will shift/scale x
        h = self.linear_cond(cond)
        shift, scale = jnp.split(h, 2, axis=-1)

      # Linear + norm + shift/scale + activation
      x = self.linear1(x)
      if cond is not None:
        x = shift + x*(1 + scale)
      x = self.activation(x)

      # Linear + gate
      x = self.linear2(x)
      a, b = jnp.split(x, 2, axis=-1)
      return a*jax.nn.sigmoid(b)

class TimeDependentResNet(eqx.Module):
    """ResNet that is conditioned on time"""

    blocks: tuple[ResBlock, ...]
    n_blocks: int = eqx.field(static=True)
    time_features: Union[TimeFeatures,None]
    in_projection: eqx.nn.Linear
    out_projection: eqx.nn.Linear

    def __init__(
        self,
        in_size: int,
        working_size: int,
        hidden_size: int,
        out_size: int,
        n_blocks: int,
        time_embedding_size: int,
        activation: Callable = jax.nn.swish,
        *,
        t: Array,
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
      - `time_embedding_size`: The size of the time embedding.
      - `activation`: The activation function in each residual block.
      - `t`: A JAX array with shape `()`.
      - `x`: A JAX array with shape `(in_size,)`.
      - `y`: A JAX array with shape `(cond_size,)`.
      - `key`: A `jax.random.PRNGKey` for initialization.
      """
      super().__init__(**kwargs)
      self.n_blocks = n_blocks

      k1, k2, k3, k4 = random.split(key, 4)

      cond_size = 4*time_embedding_size
      self.time_features = TimeFeatures(embedding_size=time_embedding_size,
                                        out_features=cond_size,
                                        key=k1)

      self.in_projection = eqx.nn.Linear(in_features=in_size,
                                         out_features=working_size,
                                         use_bias=True,
                                         key=k2)

      blocks = []
      keys = random.split(k3, n_blocks)
      for k in keys:
        block = ResBlock(in_size=working_size,
                         cond_size=cond_size,
                         hidden_size=hidden_size,
                         activation=activation,
                         x=x,
                         y=y,
                         key=k)
        self.blocks.append(block)


      self.out_projection = eqx.nn.Linear(in_features=working_size,
                                          out_features=out_size,
                                          use_bias=True,
                                          key=k4)

    def __call__(self,
                 t: Array,
                 x: Array,
                 y: Optional[Array] = None) -> Array:
      """**Arguments:**

      - `t`: A JAX array with shape `()`.
      - `x`: A JAX array with shape `(in_size,)`.
      - `y`: A JAX array with shape `(cond_size,)`.

      **Returns:**

      A JAX array with shape `(in_size,)`.
      """
      t = jnp.array(t)
      assert t.shape == ()
      assert x.ndim == 1

      # Featurize the time
      t_emb = self.time_features(t)

      # Input projection
      x = self.in_projection(x)

      # Resnet blocks
      dynamic, static = eqx.partition(self.blocks, eqx.is_array)
      def f(x, params):
          block = eqx.combine(params, static)
          return block(x, t_emb), None

      out, _ = jax.lax.scan(f, x, dynamic)

      # Output projection
      out = self.out_projection(out)
      return out

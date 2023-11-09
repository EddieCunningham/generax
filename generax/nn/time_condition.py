import jax
import jax.numpy as jnp
from jax import random
from functools import partial
import einops
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable

__all__ = ['GaussianFourierProjection',
           'TimeFeatures']

################################################################################################################

class GaussianFourierProjection(eqx.Module):

  embedding_size: int = eqx.field(static=True)
  W: eqx.nn.Linear

  def __init__(self,
               embedding_size: Optional[int] = 16,
               *,
               key: PRNGKeyArray,
               **kwargs):
    """**Arguments**:

    - `embedding_size`: The size of the embedding.
    """
    super().__init__(**kwargs)

    self.embedding_size = embedding_size
    self.W = eqx.nn.Linear(in_features=1,
                           out_features=embedding_size,
                           use_bias=False,
                           key=key)

  def __call__(self, t: Array) -> Array:
    """**Arguments:**

    - `t`: A JAX array with shape `()`.

    **Returns:**

    A JAX array with shape `(2*embedding_size,)`.
    """
    assert t.shape == ()
    t = jnp.expand_dims(t, axis=-1)
    t_proj = self.W(t*2*jnp.pi)
    return jnp.concatenate([jnp.sin(t_proj), jnp.cos(t_proj)], axis=-1)

class TimeFeatures(eqx.Module):

  out_features: int = eqx.field(static=True)
  projection: GaussianFourierProjection
  W1: Array
  W2: Array
  activation: Callable

  def __init__(self,
               embedding_size: Optional[int] = 16,
               out_features: int=8,
               activation: Callable = jax.nn.gelu,
               *,
               key: PRNGKeyArray,
               **kwargs):
    """**Arguments**:

    - `embedding_size`: The size of the embedding.
    - `out_features`: The number of output features.
    - `activation`: The activation function.
    """
    super().__init__(**kwargs)
    self.out_features = out_features

    k1, k2, k3 = random.split(key, 3)
    self.projection = GaussianFourierProjection(embedding_size=embedding_size,
                                                key=k1)
    self.W1 = eqx.nn.Linear(in_features=2*embedding_size,
                            out_features=4*embedding_size,
                            key=k2)
    self.activation = activation
    self.W2 = eqx.nn.Linear(in_features=4*embedding_size,
                            out_features=self.out_features,
                            key=k3)

  def __call__(self, t: Array) -> Array:
    """**Arguments:**

    - `t`: A JAX array with shape `()`.

    **Returns:**

    A JAX array with shape `(out_features,)`.
    """
    assert t.shape == ()
    x = self.projection(t)
    x = self.W1(x)
    x = self.activation(x)
    return self.W2(x)

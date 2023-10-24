import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
import einops
import equinox as eqx
from abc import ABC, abstractmethod
from jaxtyping import Array, PRNGKeyArray
import generax.nn.util as util
from generax.flow.base import BijectiveTransform
import numpy as np

__all__ = ['Reverse',]

class Reverse(BijectiveTransform):
  """Reverse an input
  """

  def __init__(self,
               *_,
               x: Array,
               y: Optional[Array] = None,
               key: PRNGKeyArray,
               **kwargs):
    """**Arguments**:

    - `x`: A JAX array with shape `shape`. This is *required*
           to be batched!
    - `y`: A JAX array with shape `shape` representing conditioning
          information.  Should also be batched.
    - `key`: A `jax.random.PRNGKey` for initialization
    """
    super().__init__(x=x,
                     y=y,
                     key=key,
                     **kwargs)

  def __call__(self,
               x: Array,
               y: Optional[Array] = None,
               inverse: bool=False) -> Array:
    """**Arguments**:

    - `x`: The input to the transformation
    - `y`: The conditioning information
    - `inverse`: Whether to inverse the transformation

    **Returns**:
    The transformed input and 0
    """
    assert x.shape == self.input_shape
    z = x[..., ::-1]
    log_det = jnp.array(0.0)
    return z, log_det

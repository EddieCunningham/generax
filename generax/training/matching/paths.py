import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
import einops
import equinox as eqx
from abc import ABC, abstractmethod

__all__ = ['Path',
           'StraightPath',
           'VariancePreservingPath']

class Path(eqx.Module, ABC):
  """This represents a path that we can construct between
  two points.  We will be able to get points along this path"""
  x0: jax.Array
  x1: jax.Array

  def __init__(self, x0: jax.Array, x1: jax.Array):
    """Initialize the path

    Args:
      x0: The starting point of the path
      x1: The ending point of the path
    """
    self.x0 = x0
    self.x1 = x1

  @abstractmethod
  def __call__(self, t: jax.Array) -> jax.Array:
    """Get a point along the path

    Args:
      t: The point along the path

    Returns:
      The point along the path
    """
    pass

  def get_point_and_tangent_vector(self, t: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Get a point along the path and the tangent vector at that point

    Args:
      t: The point along the path

    Returns:
      A tuple containing the point along the path and the tangent vector at that point
    """
    return jax.jvp(self.__call__, (t,), (jnp.ones_like(t),))

class StraightPath(Path):
  """This represents a straight path between two points"""

  def __call__(self, t: jax.Array) -> jax.Array:
    """Get a point along the path

    Args:
      t: The point along the path

    Returns:
      The point along the path
    """
    return self.x0 + jnp.einsum("b,b...->b...", t, self.x1 - self.x0)

class VariancePreservingPath(Path):
  """This is the path for the ODE formulation of a VP diffusion model"""

  def __init__(self,
               x0: jax.Array,
               x1: jax.Array,
               beta_min: Optional[float]=0.1,
               beta_max: Optional[float]=20.0):
    """Initialize the path

    Args:
      x0: The starting point of the path
      x1: The ending point of the path
      beta_min: The minimum value of beta
      beta_max: The maximum value of beta
    """
    self.x0 = x0
    self.x1 = x1
    self.beta_min = beta_min
    self.beta_max = beta_max

  def alpha(self, t: jax.Array) -> jax.Array:
    """Get the alpha parameter of the diffusion process at a point along the path

    Args:
      t: The point along the path

    Returns:
      The alpha parameter of the diffusion process at a point along the path
    """
    T = t*self.beta_min + 0.5*t**2*(self.beta_max - self.beta_min)
    return jnp.exp(-0.5*T)

  def __call__(self, t: jax.Array) -> jax.Array:
    """Get a point along the path

    Args:
      t: The point along the path

    Returns:
      The point along the path
    """
    alpha = self.alpha(1.0 - t)
    return alpha*self.x1 + jnp.sqrt(1 - alpha**2)*self.x0

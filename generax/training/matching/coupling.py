import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
import einops
import equinox as eqx
from abc import ABC, abstractmethod

__all__ = ['Coupling',
           'UniformCoupling',
           'OTTCoupling']

class Coupling(eqx.Module, ABC):
  """Given two batches of samples from two distributions, this
  will compute a discrete distribution q(x_0,x_1) = pi(i,j)delta(x_0 - x_0^i)delta(x_1 - x_1^j))"""
  batch_size: int
  x0: jax.Array
  x1: jax.Array
  logits: jax.Array

  def __init__(self, x0: jax.Array, x1: jax.Array):
    """Initialize the coupling

    Args:
      x0: A batch of samples from p(x_0)
      x1: A batch of samples from p(x_1)
    """
    self.batch_size = x0.shape[0]
    self.x0 = x0
    self.x1 = x1
    self.logits = self.compute_logits()
    assert self.logits.shape == (self.batch_size, self.batch_size)

  @abstractmethod
  def compute_logits(self):
    pass

  def sample_x0_given_x1(self, rng_key: random.PRNGKey) -> jax.Array:
    """Resample from the coupling

    Args:
      rng_key: The random number generator key

    Returns:
      A sample from q(x_0|x_1)
    """
    idx = jax.random.categorical(rng_key, self.logits, axis=0)
    return self.x0[idx]

class UniformCoupling(Coupling):
  """This is a uniform coupling between two distributions"""

  def compute_logits(self) -> jax.Array:
    """Compute the logits for the coupling"""
    return jnp.ones((self.batch_size, self.batch_size))/self.batch_size

  def sample_x0_given_x1(self, rng_key: random.PRNGKey) -> jax.Array:
    return self.x0


# Optimal transport library
import ott
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

class OTTCoupling(Coupling):

  def compute_logits(self) -> jax.Array:
    """Solve for the optimal transport couplings"""

    # Create a point cloud object
    geom = pointcloud.PointCloud(self.x0, self.x1)

    # Define the loss function
    ot_prob = linear_problem.LinearProblem(geom)

    # Create a sinkhorn solver
    solver = sinkhorn.Sinkhorn(ot_prob)

    # Solve the OT problem
    ot = solver(ot_prob)

    # Return the coupling
    mat = ot.matrix
    return jnp.log(mat + 1e-8)


class BatchOTCoupling(Coupling):
  pass

class BatchEOTCoupling(Coupling):
  pass

class StableCoupling(Coupling):
  pass

class HeuristicCoupling(Coupling):
  pass

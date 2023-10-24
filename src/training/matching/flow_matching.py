import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
import einops
from src.training.paths import *
from src.training.coupling import *
from src.distributions import *

class FlowMatching():

  def __init__(self,
               prior: ProbabilityDistribution,
               path_type: Optional[str]="straight",
               coupling_type: Optional[str]="uniform"):
    """Initialize the flow matcher"""
    self.prior = prior

    assert path_type in ["straight"]
    assert coupling_type in ["uniform", "ott"]

    self.path_type = path_type
    self.coupling_type = coupling_type

    paths = dict(straight=StraightPath,
                 vp=VariancePreservingPath)

    couplings = dict(uniform=UniformCoupling,
                     ott=OTTCoupling)

    self.path = paths[path_type]
    self.coupling = couplings[coupling_type]

  def get_training_batch(self,
                         data: jax.Array,
                         rng_key: random.PRNGKey) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Get the arrays that we'll use to train our network

    Args:
      data: The data.
      rng_key: The random number generator key.

    Returns:
      A tuple of (t, x, u) where t is the time, x is the point, and u is the tangent vector.
    """
    x1 = data
    k1, k2, k3 = random.split(rng_key, 3)

    # Randomly sample from our prior
    x0 = self.prior.sample(k1, x1.shape)

    # Apply a coupling in order to sample from q(x_0, x_1)
    coupling = self.coupling(x0, x1)
    x0 = coupling.sample_x0_given_x1(k2)

    # Construct a path between the prior samples and the x1
    path = self.path(x0, x1)

    # Get a random time point that we want to sample
    t = random.uniform(k3, (x1.shape[0],), minval=0.0, maxval=1.0)

    # Get the point and tangent vector
    xt, ut = path.get_point_and_tangent_vector(t)

    return t, xt, ut

  def loss_function(self,
                    V: Callable[[jax.Array, jax.Array], jax.Array],
                    data: jax.Array,
                    rng_key: random.PRNGKey) -> jax.Array:
    """Compute the flow matching objective.

    Args:
      data: The data.
      V: The vector field function.  Must accept two arguments: t and x.
      rng_key: The random number generator key.

    Returns:
      The flow matching objective.
    """
    # Given the data, get the training batch
    t, xt, ut = self.get_training_batch(data, rng_key)

    # Evaluate the vector field at the point
    vt = V(t, xt)

    # Compute the flow matching objective
    objective = jnp.mean((vt - ut)**2)

    return objective



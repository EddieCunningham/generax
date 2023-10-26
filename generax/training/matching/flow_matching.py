import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
import einops
from generax.training.matching.paths import *
from generax.training.matching.coupling import *
from generax.distributions.base import *
from generax.distributions.flow_models import ContinuousNormalizingFlow

class FlowMatching():

  def __init__(self,
               path_type: Optional[str]="straight",
               coupling_type: Optional[str]="ot",
               t0: float = 0.0,
               t1: float = 1.0):
    """Initialize the flow matcher"""
    self.t0 = t0
    self.t1 = t1

    assert path_type in ["straight"]
    assert coupling_type in ["uniform", "ot"]

    self.path_type = path_type
    self.coupling_type = coupling_type

    paths = dict(straight=StraightPath,
                 vp=VariancePreservingPath)

    couplings = dict(uniform=UniformCoupling,
                     ot=OTTCoupling)

    self.path = paths[path_type]
    self.coupling = couplings[coupling_type]

  def get_training_batch(self,
                         x0: jax.Array,
                         x1: jax.Array,
                         rng_key: random.PRNGKey) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Get the arrays that we'll use to train our network

    Args:
      x0: Source samples.
      x1: Target samples.
      rng_key: The random number generator key.

    Returns:
      A tuple of (t, x, u) where t is the time, x is the point, and u is the tangent vector.
    """
    assert x0.shape == x1.shape
    k1, k2 = random.split(rng_key, 2)

    # Apply a coupling in order to sample from q(x_0, x_1)
    coupling = self.coupling(x0, x1)
    x0 = coupling.sample_x0_given_x1(k1)

    # Construct a path between the prior samples and the x1
    path = self.path(x0, x1)

    # Get a random time point that we want to sample
    t = random.uniform(k2, (x1.shape[0],), minval=self.t0, maxval=self.t1)

    # Get the point and tangent vector
    xt, ut = path.get_point_and_tangent_vector(t)

    return t, xt, ut

  def loss_function(self,
                    flow: ContinuousNormalizingFlow,
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



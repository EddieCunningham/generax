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
from jaxtyping import Array, PRNGKeyArray
import equinox as eqx

__all__ = ['FlowMatching']

class FlowMatching():

  def __init__(self,
               path_type: Optional[str]="straight",
               coupling_type: Optional[str]="ot"):
    """Initialize the flow matcher"""

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
                         key: PRNGKeyArray) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Get the arrays that we'll use to train our network

    Args:
      x0: Source samples.
      x1: Target samples.
      key: The random number generator key.

    Returns:
      A tuple of (t, x, u) where t is the time, x is the point, and u is the tangent vector.
    """
    assert x0.shape == x1.shape
    k1, k2 = random.split(key, 2)

    # Apply a coupling in order to sample from q(x_0, x_1)
    coupling = self.coupling(x0, x1)
    x0 = coupling.sample_x0_given_x1(k1)

    # Construct a path between the prior samples and the x1
    path = self.path(x0, x1)

    # Get a random time point that we want to sample
    t = random.uniform(k2, (x1.shape[0],), minval=0.0, maxval=1.0)

    # Get the point and tangent vector
    xt, ut = path.get_point_and_tangent_vector(t)

    return t, xt, ut

  def initialize_vector_field(self,
                              flow: ContinuousNormalizingFlow,
                              data: jax.Array,
                              key: PRNGKeyArray):
    k1, k2, k3 = random.split(key, 3)
    # Sample from the prior
    x1 = data['x']
    x0 = flow.prior.sample(k1, n_samples=x1.shape[0])

    # Given the data, get the training batch
    t, xt, _ = self.get_training_batch(x0, x1, k2)

    # Pass this through the vector field to initialize
    if 'y' in data:
      y = data['y']
      assert y.shape[0] == data.shape[0]
      vt = flow.vector_field.data_dependent_init(t, xt, y, key=k3)
    else:
      vt = flow.vector_field.data_dependent_init(t, xt, key=k3)

    new_flow = eqx.tree_at(lambda tree: tree.vector_field, flow, vt)
    return new_flow

  def loss_function(self,
                    flow: ContinuousNormalizingFlow,
                    data: jax.Array,
                    key: PRNGKeyArray) -> jax.Array:
    """Compute the flow matching objective.

    Args:
      data: The data.
      V: The vector field function.  Must accept two arguments: t and x.
      key: The random number generator key.

    Returns:
      The flow matching objective.
    """
    assert isinstance(flow, ContinuousNormalizingFlow)

    # Sample from the prior
    x1 = data['x']
    x0 = flow.prior.sample(key, n_samples=x1.shape[0])

    # Given the data, get the training batch
    t, xt, ut = self.get_training_batch(x0, x1, key)

    # Evaluate the vector field at the point
    if 'y' in data:
      y = data['y']
      assert y.shape[0] == data.shape[0]
      vt = eqx.filter_vmap(flow.vector_field)(t, xt, y)
    else:
      vt = eqx.filter_vmap(flow.vector_field)(t, xt)

    # Compute the flow matching objective
    objective = jnp.mean((vt - ut)**2)

    aux = dict(objective = objective)
    return objective, aux



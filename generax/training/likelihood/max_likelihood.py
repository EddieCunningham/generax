import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
import einops
import equinox as eqx
from abc import ABC, abstractmethod
from generax.distributions.base import ProbabilityDistribution
from jaxtyping import Array, PRNGKeyArray
import generax.util.misc as misc
from generax.distributions.flow_models import NormalizingFlow

__all__ = ['max_likelihood']

def max_likelihood(flow: NormalizingFlow,
                   data: Array,
                   key: PRNGKeyArray) -> Tuple[Array, Mapping[str, Any]]:
  """Compute the maximum likelihood objective for a flow.

  **Arguments**:

  - `flow`: A `NormalizingFlow` object to optimize
  - `data`: A batch of data.  This is expected to
            be a dictionary where the keys correspond to
            different data types

  **Returns**:

  - `objective`: The objective to minimize
  - `aux`: A dictionary of auxiliary information
  """
  x = data['x']
  keys = random.split(key, x.shape[0])

  if 'y' in data:
    y = data['y']
    def log_prob(x, y, key):
      return flow.log_prob(x, y=y, key=key)
  else:
    def log_prob(x, key):
      return flow.log_prob(x, key=key)

  log_px = eqx.filter_vmap(log_prob)(x, keys).mean()
  objective = -log_px
  aux = dict(log_px = log_px)
  return objective, aux

################################################################################################################

if __name__ == '__main__':
  from debug import *
  import matplotlib.pyplot as plt
  from generax.flows.models import RealNVP

  key = random.PRNGKey(0)
  x = random.normal(key, shape=(10, 2))

  flow = RealNVP(x=x,
                 n_layers=3,
                 n_res_blocks=3,
                 hidden_size=32,
                 key=key)

  @eqx.filter_jit
  def update_step(model, data):
    (objective, aux), grads = eqx.filter_value_and_grad(max_likelihood, has_aux=True)(model, data)
    model = eqx.apply_updates(model, grads)
    return model, objective, aux

  # jit_grad = eqx.filter_jit(eqx.filter_value_and_grad(max_likelihood, has_aux=True))


  import tqdm
  pbar = tqdm.tqdm(jnp.arange(1000))
  for i in pbar:
    x = random.normal(key, shape=(10, 2))
    key, _ = random.split(key)

    flow, out, aux = update_step(flow, dict(x=x))

    # (out, aux), grads = jit_grad(flow, dict(x=x))
    # flow = eqx.apply_updates(flow, grads)


  import pdb
  pdb.set_trace()
import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
import einops
import equinox as eqx
from abc import ABC, abstractmethod
from jaxtyping import Array, PRNGKeyArray
import generax.util.misc as misc
from generax.flows.base import BijectiveTransform

def run_test(layer_init, key, x):
  layer = layer_init(input_shape=x.shape[1:], key=key)

  z, log_det = layer(x[0])
  x_inv, log_det_inv = layer.inverse(z)

  z, log_det = eqx.filter_vmap(layer)(x)
  eqx.filter_vmap(layer.inverse)(z)

  # Data dependent init
  layer.data_dependent_init(x, key=key)

def inverse_test(layer_init, key, x):
  layer = layer_init(input_shape=x.shape[1:], key=key)

  z, log_det = eqx.filter_vmap(layer)(x)
  x_reconstr, log_det2 = eqx.filter_vmap(layer.inverse)(z)
  assert jnp.allclose(x, x_reconstr)
  assert jnp.allclose(log_det, -log_det2)

def log_det_test(layer_init, key, x):
  layer = layer_init(input_shape=x.shape[1:], key=key)

  def jacobian(x):
    return jax.jacobian(lambda x: layer(x)[0])(x)

  _, log_det = eqx.filter_vmap(layer)(x)
  G = eqx.filter_vmap(jacobian)(x)
  if x.ndim == 4:
    G = einops.rearrange(G, 'b h1 w1 c1 h2 w2 c2 -> b (h1 w1 c1) (h2 w2 c2)')
  assert jnp.allclose(log_det, jnp.linalg.slogdet(G)[1])

if __name__ == '__main__':
  from debug import *
  # Turn on x64
  from jax.config import config
  config.update("jax_enable_x64", True)

  # Get all of the layers from the flows library
  import generax.flows as flows

  # Get a list of of the layers defined in the module
  layer_inits = []
  for name in dir(flows):
    t = getattr(flows, name)
    if isinstance(t, eqx._module._ModuleMeta) and (name != 'BijectiveTransform'):
      layer_inits.append(t)

  key = random.PRNGKey(0)

  # Run the tests
  for i, flow_init in enumerate(layer_inits):
    if i < 14:
      continue
    if 'Sequential' in str(flow_init):
      continue
    x = random.normal(key, shape=(10, 2))
    x = jax.nn.sigmoid(x)
    x = jnp.clip(x, 1e-4, 1.0 - 1e-4)
    run_test(flow_init, key, x)
    inverse_test(flow_init, key, x)
    log_det_test(flow_init, key, x)
    print(f'Passed tests for {flow_init}')

    x = random.normal(key, shape=(10, 5, 5, 3))
    x = jax.nn.sigmoid(x)
    x = jnp.clip(x, 1e-4, 1.0 - 1e-4)
    run_test(flow_init, key, x)
    inverse_test(flow_init, key, x)
    log_det_test(flow_init, key, x)
    print(f'Passed tests image for {flow_init}')

  import pdb; pdb.set_trace()
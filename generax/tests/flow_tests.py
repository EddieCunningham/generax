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

def run_test(layer_init, key, x, y):
  cond_shape = y.shape[1:] if y is not None else None
  layer = layer_init(input_shape=x.shape[1:],
                     key=key,
                     cond_shape=cond_shape)

  # Test that the model runs
  if y is None:
    z, log_det = layer(x[0])
    x_inv, log_det_inv = layer.inverse(z, y=y)
  else:
    z, log_det = layer(x[0], y=y[0])
    x_inv, log_det_inv = layer.inverse(z, y=y[0])

  # Test vmap
  z, log_det = eqx.filter_vmap(layer)(x, y)
  eqx.filter_vmap(layer.inverse)(z, y)

  # Test data dependent init
  layer.data_dependent_init(x, y=y, key=key)

  # Test reconstruction
  z, log_det = eqx.filter_vmap(layer)(x, y)
  x_reconstr, log_det2 = eqx.filter_vmap(layer.inverse)(z, y)
  assert jnp.allclose(x, x_reconstr)
  assert jnp.allclose(log_det, -log_det2)

  # Test log determinant
  def jacobian(x, y):
    return jax.jacobian(lambda x, y: layer(x, y=y)[0])(x, y)

  _, log_det = eqx.filter_vmap(layer)(x, y)
  G = eqx.filter_vmap(jacobian)(x, y)
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
    # if i < 13:
    #   continue
    if 'Sequential' in str(flow_init):
      continue
    if 'ConditionalOptionalTransport' in str(flow_init):
      continue
    x, y = random.normal(key, shape=(2, 10, 2))
    x = jax.nn.sigmoid(x)
    x = jnp.clip(x, 1e-4, 1.0 - 1e-4)
    run_test(flow_init, key, x, None)
    print(f'Passed 2d tests for {flow_init}')

    run_test(flow_init, key, x, y)
    print(f'Passed conditional 2d tests for {flow_init}')

    x_im = random.normal(key, shape=(10, 5, 5, 3))
    x_im = jax.nn.sigmoid(x_im)
    x_im = jnp.clip(x_im, 1e-4, 1.0 - 1e-4)
    run_test(flow_init, key, x_im, None)
    print(f'Passed image tests for {flow_init}')

    run_test(flow_init, key, x_im, y)
    print(f'Passed conditional image tests for {flow_init}')
    print()

  import pdb; pdb.set_trace()
import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
import einops
import equinox as eqx
from abc import ABC, abstractmethod
import diffrax
from jaxtyping import Array, PRNGKeyArray
from generax.flows.base import BijectiveTransform, Sequential

__all__ = ['RealNVPTransform',
           'NeuralSplineTransform']

################################################################################################################

from generax.flows.coupling import Coupling
from generax.flows.affine import ShiftScale, PLUAffine
from generax.flows.reshape import Reverse
from generax.distributions import Gaussian
from generax.flows.spline import RationalQuadraticSpline
from generax.nn.resnet_1d import ResNet1d

class RealNVPTransform(Sequential):

  def __init__(self,
               input_shape: Tuple[int],
               n_flow_layers: int = 3,
               working_size: int = 16,
               hidden_size: int = 32,
               n_blocks: int = 4,
               cond_shape: Optional[Tuple[int]] = None,
               *,
               key: PRNGKeyArray,
               **kwargs):

    # Build a dummy transfom to get the input and output shapes
    transform = ShiftScale(input_shape=input_shape, key=key)
    transform_input_shape, net_input_shape, net_output_size = Coupling.get_net_input_and_output_shapes(input_shape, transform)

    if len(input_shape) == 1:
      def create_net(key):
        return ResNet1d(in_size=net_input_shape[-1],
                      working_size=working_size,
                      hidden_size=hidden_size,
                      out_size=net_output_size,
                      n_blocks=n_blocks,
                      cond_size=cond_shape,
                      key=key)
    else:
      raise NotImplementedError(f'Only implemented for 1d inputs')

    layers = []
    keys = random.split(key, n_flow_layers)
    for i, k in enumerate(keys):
      k1, k2, k3, k4 = random.split(k, 4)
      transform = ShiftScale(transform_input_shape, key=k1)
      layer = Coupling(transform,
               create_net(k2),
               input_shape=input_shape,
               cond_shape=cond_shape,
               key=k2)
      layers.append(layer)
      layers.append(PLUAffine(input_shape=input_shape, key=k3))
      layers.append(ShiftScale(input_shape=input_shape, key=k4))

    super().__init__(*layers, **kwargs)

class NeuralSplineTransform(Sequential):

  def __init__(self,
               input_shape: Tuple[int],
               n_flow_layers: int = 3,
               working_size: int = 16,
               hidden_size: int = 32,
               n_blocks: int = 4,
               n_spline_knots: int = 8,
               cond_shape: Optional[Tuple[int]] = None,
               *,
               key: PRNGKeyArray,
               **kwargs):

    # Build a dummy transfom to get the input and output shapes
    transform = RationalQuadraticSpline(input_shape=input_shape,
                                        K=n_spline_knots,
                                        key=key)
    transform_input_shape, net_input_shape, net_output_size = Coupling.get_net_input_and_output_shapes(input_shape, transform)

    if len(input_shape) == 1:
      def create_net(key):
        return ResNet1d(in_size=net_input_shape[-1],
                      working_size=working_size,
                      hidden_size=hidden_size,
                      out_size=net_output_size,
                      n_blocks=n_blocks,
                      cond_size=cond_shape,
                      key=key)
    else:
      raise NotImplementedError(f'Only implemented for 1d inputs')

    layers = []
    keys = random.split(key, n_flow_layers)
    for i, k in enumerate(keys):
      k1, k2, k3, k4 = random.split(k, 4)
      transform = RationalQuadraticSpline(transform_input_shape,
                                          K=n_spline_knots,
                                          key=k1)
      layer = Coupling(transform,
               create_net(k2),
               input_shape=input_shape,
               cond_shape=cond_shape,
               key=k2)
      layers.append(layer)
      layers.append(PLUAffine(input_shape=input_shape, key=k3))
      layers.append(ShiftScale(input_shape=input_shape, key=k4))

    super().__init__(*layers, **kwargs)

################################################################################################################

if __name__ == '__main__':
  from debug import *
  import matplotlib.pyplot as plt

  key = random.PRNGKey(0)
  x = random.normal(key, shape=(10, 2))

  layer = NeuralSplineTransform(input_shape=x.shape[1:],
                           n_flow_layers=3,
                           working_size=16,
                           hidden_size=32,
                           n_blocks=3,
                           n_spline_knots=8,
                           key=key)

  layer(x[0])
  layer = layer.data_dependent_init(x, key=key)

  z, log_det = eqx.filter_vmap(layer)(x)

  z, log_det = layer(x[0])
  x_reconstr, log_det2 = layer(z, inverse=True)

  G = jax.jacobian(lambda x: layer(x)[0])(x[0])
  log_det_true = jnp.linalg.slogdet(G)[1]

  assert jnp.allclose(log_det, log_det_true)
  assert jnp.allclose(x[0], x_reconstr)

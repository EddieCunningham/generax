import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
import einops
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray
from generax.flows.base import BijectiveTransform, Sequential

__all__ = ['GeneralTransform',
           'NICETransform',
           'RealNVPTransform',
           'NeuralSplineTransform',
           'GeneralImageTransform',
           'NICEImageTransform',
           'RealNVPImageTransform',
           'NeuralSplineImageTransform']

################################################################################################################

from generax.flows.coupling import Coupling
from generax.flows.affine import Shift, ShiftScale, PLUAffine
from generax.flows.conv import OneByOneConv
from generax.flows.reshape import Reverse
from generax.flows.spline import RationalQuadraticSpline
from generax.nn.resnet import ResNet
from generax.nn.unet import UNet, Encoder

class GeneralTransform(Sequential):

  def __init__(self,
               TransformType: type,
               input_shape: Tuple[int],
               n_flow_layers: int = 3,
               working_size: int = 16,
               hidden_size: int = 32,
               n_blocks: int = 4,
               filter_shape: Optional[Tuple[int]] = (3, 3),
               cond_shape: Optional[Tuple[int]] = None,
               *,
               key: PRNGKeyArray,
               **kwargs):

    # Build a dummy transfom to get the input and output shapes
    transform_input_shape, net_input_shape = Coupling.get_split_shapes(input_shape)
    transform = TransformType(input_shape=transform_input_shape,
                              key=key)
    net_output_size = Coupling.get_net_output_shapes(input_shape, transform)

    if len(net_input_shape) == 3:
      H, W, C = net_input_shape
      assert net_output_size%(H*W) == 0
      net_output_size = net_output_size // (H*W)

    def create_net(key):
      return ResNet(input_shape=net_input_shape,
                    working_size=working_size,
                    hidden_size=hidden_size,
                    out_size=net_output_size,
                    n_blocks=n_blocks,
                    filter_shape=filter_shape,
                    cond_shape=cond_shape,
                    key=key)

    layers = []
    keys = random.split(key, n_flow_layers)
    for i, k in enumerate(keys):
      k1, k2, k3, k4 = random.split(k, 4)
      transform = TransformType(input_shape=transform_input_shape,
                                cond_shape=cond_shape,
                                key=k1)
      layer = Coupling(transform,
                       create_net(k2),
                       input_shape=input_shape,
                       cond_shape=cond_shape,
                       key=k2)
      layers.append(layer)
      layers.append(PLUAffine(input_shape=input_shape,
                              cond_shape=cond_shape,
                              key=k3))
      layers.append(ShiftScale(input_shape=input_shape,
                               cond_shape=cond_shape,
                               key=k4))

    super().__init__(*layers, **kwargs)

class NICETransform(GeneralTransform):
  def __init__(self,
               *args,
               **kwargs):
    super().__init__(TransformType=Shift,
                     *args,
                     **kwargs)

class RealNVPTransform(GeneralTransform):
  def __init__(self,
               *args,
               **kwargs):
    super().__init__(TransformType=ShiftScale,
                     *args,
                     **kwargs)

class NeuralSplineTransform(GeneralTransform):
  def __init__(self,
               *args,
               n_spline_knots: int = 8,
               **kwargs):
    super().__init__(TransformType=partial(RationalQuadraticSpline, K=n_spline_knots),
                     *args,
                     **kwargs)

################################################################################################################

class GeneralImageTransform(Sequential):

  def __init__(self,
               TransformType: type,
               input_shape: Tuple[int],
               n_flow_layers: int = 3,
               cond_shape: Optional[Tuple[int]] = None,
               unet: Optional[bool] = True,
               *,
               key: PRNGKeyArray,
               **kwargs):

    # Build a dummy transfom to get the input and output shapes
    transform_input_shape, net_input_shape = Coupling.get_split_shapes(input_shape)
    transform = TransformType(input_shape=transform_input_shape,
                              key=key)
    net_output_size = Coupling.get_net_output_shapes(input_shape, transform)

    if unet:
      H, W, C = net_input_shape
      def create_net(key):
        return UNet(input_shape=net_input_shape,
                      dim=kwargs.pop('dim', 32),
                      out_channels=net_output_size//(H*W),
                      dim_mults=kwargs.pop('dim_mults', (1, 2, 4)),
                      resnet_block_groups=kwargs.pop('resnet_block_groups', 8),
                      attn_heads=kwargs.pop('attn_heads', 4),
                      attn_dim_head=kwargs.pop('attn_dim_head', 32),
                      cond_shape=cond_shape,
                      time_dependent=False,
                      key=key)
    else:
      def create_net(key):
        return Encoder(input_shape=net_input_shape,
                      dim=kwargs.pop('dim', 32),
                      dim_mults=kwargs.pop('dim_mults', (1, 2, 4)),
                      resnet_block_groups=kwargs.pop('resnet_block_groups', 8),
                      attn_heads=kwargs.pop('attn_heads', 4),
                      attn_dim_head=kwargs.pop('attn_dim_head', 32),
                      out_size=net_output_size,
                      cond_shape=cond_shape,
                      key=key)

    layers = []
    keys = random.split(key, n_flow_layers)
    for i, k in enumerate(keys):
      k1, k2, k3, k4 = random.split(k, 4)
      transform = TransformType(input_shape=transform_input_shape,
                                cond_shape=cond_shape,
                                key=k1)
      layer = Coupling(transform,
                       create_net(k2),
                       input_shape=input_shape,
                       cond_shape=cond_shape,
                       key=k2)
      layers.append(layer)
      layers.append(OneByOneConv(input_shape=input_shape,
                              key=k3))
      layers.append(ShiftScale(input_shape=input_shape,
                               cond_shape=cond_shape,
                               key=k4))

    super().__init__(*layers, **kwargs)

class NICEImageTransform(GeneralImageTransform):
  def __init__(self,
               *args,
               **kwargs):
    super().__init__(TransformType=Shift,
                     *args,
                     **kwargs)

class RealNVPImageTransform(GeneralImageTransform):
  def __init__(self,
               *args,
               **kwargs):
    super().__init__(TransformType=ShiftScale,
                     *args,
                     **kwargs)

class NeuralSplineImageTransform(GeneralImageTransform):
  def __init__(self,
               *args,
               n_spline_knots: int = 8,
               **kwargs):
    super().__init__(TransformType=partial(RationalQuadraticSpline, K=n_spline_knots),
                     *args,
                     **kwargs)

################################################################################################################

if __name__ == '__main__':
  from debug import *
  import matplotlib.pyplot as plt
  jax.config.update("jax_enable_x64", True)

  key = random.PRNGKey(0)
  x = random.normal(key, shape=(4, 8, 8, 2))

  layer = NeuralSplineImageTransform(input_shape=x.shape[1:],
                           n_flow_layers=3,
                           dim=16,
                           dim_mults=(1, 2),
                           attn_heads=4,
                           attn_dim_head=8,
                           key=key)

  layer(x[0])
  layer = layer.data_dependent_init(x, key=key)

  z, log_det = eqx.filter_vmap(layer)(x)

  z, log_det = layer(x[0])
  x_reconstr, log_det2 = layer(z, inverse=True)

  G = jax.jacobian(lambda x: layer(x)[0])(x[0])
  G = einops.rearrange(G, 'h1 w1 c1 h2 w2 c2 -> (h1 w1 c1) (h2 w2 c2)')
  log_det_true = jnp.linalg.slogdet(G)[1]

  assert jnp.allclose(log_det, log_det_true)
  assert jnp.allclose(x[0], x_reconstr)

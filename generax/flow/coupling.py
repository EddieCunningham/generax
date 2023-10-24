import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
import einops
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray
import generax.nn.util as util
from generax.flow.base import BijectiveTransform
import numpy as np
from generax.nn.resnet_1d import ResNet1d
from generax.nn.util import RavelParameters

__all__ = ['Coupling']

class Coupling(BijectiveTransform):
  """Parametrize a flow over half of the inputs using the other half.
  The conditioning network will be fixed

  ```python
  # Intended usage:
  layer = Coupling(BijectiveTransformInit,
                   hidden_size=16,
                   n_blocks=2,
                   activation=jax.nn.swish,
                   x=x,
                   y=y,
                   key=key)
  z, log_det = layer(x, y)
  ```

  **Attributes**:
  - `transform`: The bijective transformation to use.
  - `zero_init`: A zero initialization for the parameters.
  - `net`: The neural network to use.
  """

  net: eqx.Module
  zero_init: eqx.Module
  params_to_transform: RavelParameters

  def __init__(self,
               transform_type,
               *,
               x: Array,
               y: Optional[Array] = None,
               key: PRNGKeyArray,
               working_size: int = 16,
               hidden_size: int = 32,
               n_blocks: int = 2,
               activation: Callable = jax.nn.swish,
               **kwargs):
    """**Arguments**:

    - `x`: A JAX array with shape `shape`. This is *required*
           to be batched!
    - `y`: A JAX array with shape `shape` representing conditioning
    - `key`: A `jax.random.PRNGKey` for initialization
    """
    k1, k2 = random.split(key, 2)
    super().__init__(x=x,
                     y=y,
                     key=k1,
                     **kwargs)

    # Split the input into two halves
    x1, x2 = self.split(x)

    # Transform x1 through the bijection in order to figure
    # out the shapes of the parameters.
    transform = transform_type(x=x1,
                               y=y,
                               key=k2,
                               **kwargs)
    self.params_to_transform = RavelParameters(transform)
    out_size = self.params_to_transform.flat_params_size

    self.net = ResNet1d(working_size=working_size,
                       hidden_size=hidden_size,
                       out_size=out_size,
                       n_blocks=n_blocks,
                       activation=activation,
                       x=x2,
                       y=y,
                       key=k2)

    params = eqx.filter_vmap(self.net)(x2, y)
    self.zero_init = util.ZeroInit(x=params,
                                   y=y,
                                   key=k2)

  def split(self, x: Array) -> Tuple[Array, Array]:
    """Split the input into two halves."""
    split_dim = x.shape[-1]//2
    x1, x2 = x[..., :split_dim], x[..., split_dim:]
    return x1, x2

  def __call__(self,
               x: Array,
               y: Optional[Array] = None,
               inverse: bool = False) -> Array:
    """**Arguments**:

    - `x`: The input to the transformation
    - `y`: The conditioning information
    - `inverse`: Whether to inverse the transformation

    **Returns**:
    (z, log_det)
    """
    assert x.shape == self.input_shape, 'Only works on unbatched data'

    # Split the input into two halves
    x1, x2 = self.split(x)
    params = self.net(x2, y=y)
    params = self.zero_init(params)

    # Apply the transformation to x1 given x2
    transform = self.params_to_transform(params)
    z1, log_det = transform(x1, y=y, inverse=inverse)

    z = jnp.concatenate([z1, x2], axis=-1)
    return z, log_det

################################################################################################################

if __name__ == '__main__':
  from debug import *
  import matplotlib.pyplot as plt
  from generax.flow.base import Sequential
  from generax.flow.affine import DenseAffine, ShiftScale
  from generax.flow.reshape import Reverse
  from generax.flow.models import NormalizingFlow
  from generax.distributions import Gaussian

  # Turn on x64
  from jax.config import config
  config.update("jax_enable_x64", True)

  key = random.PRNGKey(0)
  x = random.normal(key, shape=(10, 5))

  layer = Coupling(ShiftScale,
                   x=x,
                   key=key)

  layer = Sequential(partial(Coupling, ShiftScale),
             Reverse,
             partial(Coupling, ShiftScale),
             Reverse,
             partial(Coupling, ShiftScale),
             x=x,
             key=key)

  z, log_det = eqx.filter_vmap(layer)(x)

  z, log_det = layer(x[0])
  x_reconstr, log_det2 = layer(z, inverse=True)

  G = jax.jacobian(lambda x: layer(x)[0])(x[0])
  log_det_true = jnp.linalg.slogdet(G)[1]

  assert jnp.allclose(log_det, log_det_true)
  assert jnp.allclose(x[0], x_reconstr)

  flow = NormalizingFlow(transform=layer,
                         prior=Gaussian(data_shape=layer.input_shape))

  log_px = eqx.filter_vmap(flow.log_prob)(x)

  def loss(flow, x):
    return 0.001*eqx.filter_vmap(flow.log_prob)(x).mean()

  out = eqx.filter_grad(loss)(flow, x)
  new_flow = eqx.apply_updates(flow, out)
  log_px2 = eqx.filter_vmap(new_flow.log_prob)(x)

  import pdb; pdb.set_trace()
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
from generax.flow.base import BijectiveTransform, Sequential
from generax.distributions import ProbabilityDistribution

__all__ = ['NormalizingFlow',
           'RealNVP',
           'NeuralSpline']

################################################################################################################

class NormalizingFlow(ProbabilityDistribution, ABC):
  """A normalizing flow is a model that we use to represent probability
  distributions https://arxiv.org/pdf/1912.02762.pdf

  **Atributes**:

  - `transform`: A `BijectiveTransform` object that transforms a variable
                  from the base space to the data space and also computes
                  the change is log pdf.
    - `transform(x) -> (z,log_det)`: Apply the transformation to the input.
  - `prior`: The prior probability distribution.

  **Methods**:

  - `to_base_space(x) -> z`: Transform a point from the data space to the base space.
  - `sample_and_log_prob(key) -> (x,log_px)`: Sample from the distribution and compute the log probability.
  - `sample(key) -> x`: Pull a single sample from the model
  - `log_prob(x) -> log_px`: Compute the log probability of a point under the model
  """
  transform: BijectiveTransform
  prior: ProbabilityDistribution

  def __init__(self,
               transform: BijectiveTransform,
               prior: ProbabilityDistribution,
               **kwargs):
    """**Arguments**:

    - `transform`: A bijective transformation
    - `prior`: The prior distribution
    """
    self.transform = transform
    self.prior = prior
    data_shape = self.transform.input_shape
    super().__init__(data_shape=data_shape, **kwargs)

  def to_base_space(self,
                    x: Array,
                    y: Optional[Array] = None) -> Array:
    """**Arguments**:

    - `x`: A JAX array with shape `(dim,)`.
    - `y`: The conditioning information

    **Returns**:
    A JAX array with shape `(dim,)`.
    """
    return self.transform(x, y=y)[0]

  def to_data_space(self,
                    z: Array,
                    y: Optional[Array] = None) -> Array:
    """**Arguments**:

    - `z`: A JAX array with shape `(dim,)`.
    - `y`: The conditioning information

    **Returns**:
    A JAX array with shape `(dim,)`.
    """
    return self.transform(z, y=y, inverse=True)[0]

  def sample_and_log_prob(self,
                          key: PRNGKeyArray,
                          y: Optional[Array] = None) -> Array:
    """**Arguments**:

    - `key`: The random number generator key.
    - `y`: The conditioning information

    **Returns**:
    A single sample from the model.  Use vmap to get more samples.
    """
    z, log_pz = self.prior.sample_and_log_prob(key)
    x, log_det = self.transform(z, y=y, inverse=True)
    return x, log_pz + log_det

  def log_prob(self,
               x: Array,
               y: Optional[Array] = None) -> Array:
    """**Arguments**:

    - `x`: The point we want to compute logp(x) at.
    - `y`: The conditioning information

    **Returns**:
    The log likelihood of x under the model.
    """
    z, log_det = self.transform(x, y=y)
    log_pz = self.prior.log_prob(z)
    return log_pz + log_det

################################################################################################################

from generax.flow.coupling import Coupling
from generax.flow.affine import ShiftScale, PLUAffine
from generax.flow.reshape import Reverse
from generax.distributions import Gaussian
from generax.flow.spline import RationalQuadraticSpline

class RealNVP(NormalizingFlow):

  def __init__(self,
               *_,
               x: Array,
               y: Optional[Array] = None,
               key: PRNGKeyArray,
               n_layers: int,
               n_res_blocks: int,
               working_size: int,
               hidden_size: int,
               activation=jax.nn.swish,
               **kwargs):

    make_coupling_layer_type = lambda : partial(Coupling,
                                                ShiftScale,
                                                working_size=working_size,
                                                hidden_size=hidden_size,
                                                n_blocks=n_res_blocks,
                                                activation=activation)

    layers = []
    for i in range(n_layers):
      layers.append(make_coupling_layer_type())
      layers.append(PLUAffine)
      layers.append(ShiftScale)

    transform = Sequential(*layers, x=x, y=y, key=key, **kwargs)

    z = transform(x[0])[0]
    prior = Gaussian(data_shape=z.shape, **kwargs)
    super().__init__(transform=transform, prior=prior, **kwargs)

class NeuralSpline(NormalizingFlow):

  def __init__(self,
               *_,
               x: Array,
               y: Optional[Array] = None,
               key: PRNGKeyArray,
               n_layers: int,
               n_res_blocks: int,
               working_size: int,
               hidden_size: int,
               activation=jax.nn.swish,
               **kwargs):

    make_coupling_layer_type = lambda : partial(Coupling,
                                                eqx.Partial(RationalQuadraticSpline, K=8),
                                                working_size=working_size,
                                                hidden_size=hidden_size,
                                                n_blocks=n_res_blocks,
                                                activation=activation)

    layers = []
    for i in range(n_layers):
      layers.append(make_coupling_layer_type())
      layers.append(PLUAffine)
      layers.append(ShiftScale)

    transform = Sequential(*layers, x=x, y=y, key=key, **kwargs)

    z = transform(x[0])[0]
    prior = Gaussian(data_shape=z.shape, **kwargs)
    super().__init__(transform=transform, prior=prior, **kwargs)

################################################################################################################

if __name__ == '__main__':
  from debug import *
  import matplotlib.pyplot as plt

  key = random.PRNGKey(0)
  x = random.normal(key, shape=(10, 2))

  flow = RealNVP(x=x,
                 n_layers=3,
                 n_res_blocks=3,
                 hidden_size=32,
                 key=key)

  log_px = eqx.filter_vmap(flow.log_prob)(x)

  jit_layer = eqx.filter_jit(eqx.filter_vmap(flow.log_prob))

  import tqdm
  pbar = tqdm.tqdm(jnp.arange(1000))
  for i in pbar:
    x = random.normal(key, shape=(10, 2))
    key, _ = random.split(key)
    out = jit_layer(x)

  import pdb
  pdb.set_trace()


  def loss(flow, x):
    return 0.001*eqx.filter_vmap(flow.log_prob)(x).mean()

  out = eqx.filter_grad(loss)(flow, x)
  new_flow = eqx.apply_updates(flow, out)
  log_px2 = eqx.filter_vmap(new_flow.log_prob)(x)

  import pdb; pdb.set_trace()
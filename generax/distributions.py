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

__all__ = ['ProbabilityDistribution',
           'Gaussian']

class ProbabilityDistribution(eqx.Module, ABC):
  """An object that we can sample from and use to evaluate log probabilities.

  **Atributes**:

  - `data_shape`: The dimension of the sampling space.

  **Methods**:

  - `sample_and_log_prob(key) -> (x,log_px)`: Sample from the distribution and compute the log probability.
  - `sample(key) -> x`: Pull a single sample from the model
  - `log_prob(x) -> log_px`: Compute the log probability of a point under the model
  """

  data_shape: int = eqx.field(static=True)

  def __init__(self,
               *,
               data_shape: Union[int, Tuple[int]],
               **kwargs):
    """**Arguments**:

    - `data_shape`: The dimension of the space.  This can be either
            an integer or a tuple of integers to represent images
    """
    super().__init__(**kwargs)
    assert isinstance(data_shape, tuple) or isinstance(data_shape, list)
    self.data_shape = data_shape

  @abstractmethod
  def sample_and_log_prob(self,
                          key: PRNGKeyArray) -> Array:
    """**Arguments**:

    - `key`: The random number generator key.

    **Returns**:
    A single sample from the model with its log probability.
    """
    pass

  def sample(self,
             key: PRNGKeyArray,
             n_samples: Optional[int] = None) -> Array:
    """**Arguments**:

    - `key`: The random number generator key.
    - `n_samples`: The number of samples to draw.  If `None`,
                   then we just draw a single sample.

    **Returns**:
    Samples from the model
    """
    if n_samples is None:
      return self.sample_and_log_prob(key)[0]
    keys = random.split(key, n_samples)
    return eqx.filter_vmap(self.sample)(keys)

  @abstractmethod
  def log_prob(self,
               x: Array) -> Array:
    """**Arguments**:

    - `x`: The point we want to compute logp(x) at.

    **Returns**:
    The log likelihood of x under the model.
    """
    pass

################################################################################################################

class Gaussian(ProbabilityDistribution):
  """This represents a Gaussian distribution"""

  def sample(self,
             key: PRNGKeyArray) -> Array:
    """**Arguments**:

    - `key`: The random number generator key.

    **Returns**:
    A single sample from the model.  Use vmap to get more samples.
    """
    return random.normal(key, shape=self.data_shape)

  def log_prob(self,
               x: Array) -> Array:
    """**Arguments**:

    - `x`: The point we want to compute logp(x) at.

    **Returns**:
    The log likelihood of x under the model.
    """
    return jax.scipy.stats.norm.logpdf(x).sum()

  def sample_and_log_prob(self,
                          key: PRNGKeyArray) -> Array:
    """**Arguments**:

    - `key`: The random number generator key.

    **Returns**:
    A single sample from the model with its log probability.
    """
    x = self.sample(key)
    log_px = self.log_prob(x)
    return x, log_px

################################################################################################################
from generax.flows.base import BijectiveTransform

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

  def data_dependent_init(self,
                          x: Array,
                          y: Optional[Array] = None,
                          key: PRNGKeyArray = None) -> BijectiveTransform:
    """Initialize the parameters of the layer based on the data.

    **Arguments**:

    - `x`: The data to initialize the parameters with.
    - `y`: The conditioning information
    - `key`: A `jax.random.PRNGKey` for initialization

    **Returns**:
    A new flow with the parameters initialized.
    """
    new_layer = self.transform.data_dependent_init(x, y=y, key=key)

    # Turn the new parameters into a new module
    get_transform = lambda tree: tree.transform
    return eqx.tree_at(get_transform, self, new_layer)


from generax.flows.models import RealNVPTransform, NeuralSplineTransform

class RealNVP(NormalizingFlow):

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
    """**Arguments**:

    - `input_shape`: The shape of the input data.
    - `n_flow_layers`: The number of layers in the flow.
    - `working_size`: The size of the working space.
    - `hidden_size`: The size of the hidden layers.
    - `n_blocks`: The number of blocks in the coupling layers.
    - `cond_shape`: The shape of the conditioning information.
    - `key`: A `jax.random.PRNGKey` for initialization
    """
    transform = RealNVPTransform(input_shape=input_shape,
                                 n_flow_layers=n_flow_layers,
                                 working_size=working_size,
                                 hidden_size=hidden_size,
                                 n_blocks=n_blocks,
                                 cond_shape=cond_shape,
                                 key=key)
    prior = Gaussian(data_shape=input_shape)
    super().__init__(transform=transform,
                     prior=prior,
                     **kwargs)

class NeuralSpline(NormalizingFlow):

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
    """**Arguments**:

    - `input_shape`: The shape of the input data.
    - `n_flow_layers`: The number of layers in the flow.
    - `working_size`: The size of the working space.
    - `hidden_size`: The size of the hidden layers.
    - `n_blocks`: The number of blocks in the coupling layers.
    - `cond_shape`: The shape of the conditioning information.
    - `n_splice_knots`: The number of knots in the spline.
    - `key`: A `jax.random.PRNGKey` for initialization
    """
    transform = NeuralSplineTransform(input_shape=input_shape,
                                 n_flow_layers=n_flow_layers,
                                 working_size=working_size,
                                 hidden_size=hidden_size,
                                 n_blocks=n_blocks,
                                 n_spline_knots=n_spline_knots,
                                 cond_shape=cond_shape,
                                 key=key)
    prior = Gaussian(data_shape=input_shape)
    super().__init__(transform=transform,
                     prior=prior,
                     **kwargs)

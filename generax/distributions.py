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

# class ContinuousNormalizingFlow(GenerativeModel):

#   def __init__(self, path: str):
#     """**Arguments**:

#     - `path`: The path to the model.
#     """
#     pass

# class DiffusionModel(GenerativeModel):
#   pass

################################################################################################################

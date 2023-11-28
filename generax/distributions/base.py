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
           'ProbabilityPath',
           'Gaussian']

class ProbabilityDistribution(eqx.Module, ABC):
  """An object that we can sample from and use to evaluate log probabilities.  This is an abstract base class.

  **Atributes**:

  - `input_shape`: The shape of samples.

  **Methods**:

  - `sample_and_log_prob(key) -> (x,log_px)`: Sample from the distribution and compute the log probability.
  - `sample(key) -> x`: Pull a single sample from the model
  - `log_prob(x) -> log_px`: Compute the log probability of a point under the model
  """

  input_shape: int = eqx.field(static=True)

  def __init__(self,
               *,
               input_shape: Union[int, Tuple[int]],
               **kwargs):
    """**Arguments**:

    - `input_shape`: The dimension of the space.  This can be either
            an integer or a tuple of integers to represent images
    """
    assert isinstance(input_shape, tuple) or isinstance(input_shape, list)
    self.input_shape = input_shape

  @abstractmethod
  def sample_and_log_prob(self,
                          key: PRNGKeyArray,
                          y: Optional[Array] = None) -> Array:
    """**Arguments**:

    - `key`: The random number generator key.

    **Returns**:
    A single sample from the model with its log probability.

    Use eqx.filter_vmap to get more samples!  For example,
    ```python
    keys = random.split(key, n_samples)
    x, log_px = eqx.filter_vmap(self.sample_and_log_prob)(keys)
    ```
    """
    pass

  def sample(self,
             key: PRNGKeyArray,
             y: Optional[Array] = None) -> Array:
    """
    **Arguments**:

    - `key`: The random number generator key.

    **Returns**:
    Samples from the model

    Use eqx.filter_vmap to get more samples!  For example,
    ```python
    keys = random.split(key, n_samples)
    samples = eqx.filter_vmap(self.sample)(keys)
    ```
    """
    return self.sample_and_log_prob(key, y)[0]

  @abstractmethod
  def log_prob(self,
               x: Array,
               y: Optional[Array] = None,
               key: Optional[PRNGKeyArray] = None) -> Array:
    """**Arguments**:

    - `x`: The point we want to compute logp(x) at.
    - `y`: The (optional) conditioning information.
    - `key`: The random number generator key.  Can be passed in the event
             that we're getting a stochastic estimate of the log prob.

    **Returns**:
    The log likelihood of x under the model.
    """
    pass

  def score(self,
            x: Array,
            y: Optional[Array] = None,
            key: Optional[PRNGKeyArray] = None) -> Array:
    """**Arguments**:

    - `x`: The point we want to compute grad logp(x) at.
    - `y`: The (optional) conditioning information.
    - `key`: The random number generator key.  Can be passed in the event
             that we're getting a stochastic estimate of the log prob.

    **Returns**:
    The log likelihood of x under the model.
    """
    return eqx.filter_grad(self.log_prob)(x, y=y, key=key)

################################################################################################################

class ProbabilityPath(ProbabilityDistribution):
  """A time dependent probability distribution.

  **Atributes**:

  - `input_shape`: The dimension of the sampling space.

  **Methods**:

  - `sample_and_log_prob(key) -> (x,log_px)`: Sample from the distribution and compute the log probability.
  - `sample(key) -> x`: Pull a single sample from the model
  - `log_prob(x) -> log_px`: Compute the log probability of a point under the model
  """

  input_shape: int = eqx.field(static=True)

  def __init__(self,
               *,
               input_shape: Union[int, Tuple[int]],
               **kwargs):
    """**Arguments**:

    - `input_shape`: The dimension of the space.  This can be either
            an integer or a tuple of integers to represent images
    """
    assert isinstance(input_shape, tuple) or isinstance(input_shape, list)
    self.input_shape = input_shape

  @abstractmethod
  def sample_and_log_prob(self,
                          t: Array,
                          key: PRNGKeyArray) -> Array:
    """**Arguments**:

    - `t`: The time at which we want to sample.
    - `key`: The random number generator key.

    **Returns**:
    A single sample from the model with its log probability.
    """
    pass

  @abstractmethod
  def log_prob(self,
               t: Array,
               xt: Array,
               y: Optional[Array] = None,
               key: Optional[PRNGKeyArray] = None) -> Array:
    """**Arguments**:

    - `t`: The time at which we want to sample.
    - `xt`: The point we want to compute logp(x) at.
    - `y`: The (optional) conditioning information.
    - `key`: The random number generator key.  Can be passed in the event
             that we're getting a stochastic estimate of the log prob.

    **Returns**:
    The log likelihood of x under the model.
    """
    pass

  def sample(self,
             t: Array,
             key: PRNGKeyArray,
             y: Optional[Array] = None) -> Array:
    """
    Use eqx.filter_vmap to get more samples!  For example,
    keys = random.split(key, n_samples)
    samples = eqx.filter_vmap(self.sample, in_axes=(None, 0))(t, keys)

    **Arguments**:

    - `t`: The time at which we want to sample.
    - `key`: The random number generator key.
    - `n_samples`: The number of samples to draw.  If `None`,
                   then we just draw a single sample.

    **Returns**:
    Samples from the model
    """
    return self.sample_and_log_prob(t, key, y)[0]

  def score(self,
            t: Array,
            xt: Array,
            y: Optional[Array] = None,
            key: Optional[PRNGKeyArray] = None) -> Array:
    """**Arguments**:

    - `x`: The point we want to compute grad logp(x) at.
    - `y`: The (optional) conditioning information.
    - `key`: The random number generator key.  Can be passed in the event
             that we're getting a stochastic estimate of the log prob.

    **Returns**:
    The log likelihood of x under the model.
    """
    def log_prob(xt):
      return self.log_prob(t, xt, y=y, key=key)
    return eqx.filter_grad(log_prob)(xt)

  @abstractmethod
  def transform_and_vector_field(self,
                                 t: Array,
                                 x0: Array,
                                 y: Optional[Array] = None,
                                 **kwargs) -> Array:
    """The vector field that samples evolve on as t changes

    **Arguments**:

    - `t`: Time.
    - `x0`: A point in the base space.
    - `y`: The (optional) conditioning information.

    **Returns**:
    (xt, dxt/dt)
    """
    pass

    @abstractmethod
    def vector_field(self,
                     t: Array,
                     xt: Array,
                     y: Optional[Array] = None,
                     **kwargs) -> Array:
      """The vector field that samples evolve on as t changes

      **Arguments**:

      - `t`: Time.
      - `xt`: A point in the base space.
      - `y`: The (optional) conditioning information.

      **Returns**:
      dxt/dt
      """
      pass

################################################################################################################

class Gaussian(ProbabilityDistribution):
  """This represents a Gaussian distribution"""

  def sample(self,
             key: PRNGKeyArray,
             y: Optional[Array] = None) -> Array:
    """**Arguments**:

    - `key`: The random number generator key.

    **Returns**:
    A single sample from the model.  Use eqx.filter_vmap to get more samples.
    """
    return random.normal(key, shape=self.input_shape)

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

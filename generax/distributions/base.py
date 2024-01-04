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
           'BoltzmannDistribution',
           'ProductDistribution',
           'ProbabilityPath',
           'EmpiricalDistribution',
           'Gaussian',
           'Uniform',
           'Logistic']

class ProbabilityDistribution(eqx.Module, ABC):
  """An object that we can sample from and use to evaluate log probabilities.  This is an abstract base class.

  **Atributes**:

  - `input_shape`: The shape of samples.

  **Methods**:

  - `sample_and_log_prob(key) -> (x,log_px)`: Sample from the distribution and compute the log probability.
  - `sample(key) -> x`: Pull a single sample from the model
  - `log_prob(x) -> log_px`: Compute the log probability of a point under the model
  - `score(x) -> dlog_px/dx`: Compute the gradient of the log probability of a point under the model
  """

  input_shape: int = eqx.field(static=True)

  def __init__(self,
               *,
               input_shape: Tuple[int],
               **kwargs):
    """**Arguments**:

    - `input_shape`: The dimensions of the samples
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

  def energy(self,
             x: Array,
             y: Optional[Array] = None,
             key: Optional[PRNGKeyArray] = None) -> Array:
    return -self.log_prob(x, y=y, key=key)

class BoltzmannDistribution(ProbabilityDistribution):
  """An unnormalized probability density function.  p(x) = 1/Z*exp(-E(x))

  **Atributes**:

  - `input_shape`: The shape of samples.

  **Methods**:

  - `energy(x) -> E`: Compute the energy of a point under the model
  - `score(x) -> grad log_px = -grad E`: Compute the gradient of the log probability of a point under the model
  """
  def sample_and_log_prob(self,
                          key: PRNGKeyArray,
                          y: Optional[Array] = None) -> Array:
    raise AssertionError("Can't sample from a Boltzmann distribution")

  def log_prob(self,
               x: Array,
               y: Optional[Array] = None,
               key: Optional[PRNGKeyArray] = None) -> Array:
    raise AssertionError("Can't compute log prob of a Boltzmann distribution")

  @abstractmethod
  def energy(self,
             x: Array,
             y: Optional[Array] = None,
             key: Optional[PRNGKeyArray] = None) -> Array:
    """**Arguments**:

    - `x`: The point we want to compute E(x) at.
    - `y`: The (optional) conditioning information.
    - `key`: The random number generator key.  Can be passed in the event
             that we're getting a stochastic estimate of the energy.

    **Returns**:
    The energy of x under the model.
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
    return -eqx.filter_grad(self.energy)(x, y=y, key=key)

################################################################################################################

class ProductDistribution(ProbabilityDistribution):
  """A product of probability distributions
  """

  dists: Tuple[ProbabilityDistribution]

  def __init__(self,
               *distributions: Tuple[ProbabilityDistribution],
               **kwargs):
    """**Arguments**:

    - `distributions`: The distributions to take the product of.
    """
    self.dists = distributions

    # Check that the input shapes are all the same on all but the
    # first axis and construct the total input shape
    input_shape = list(self.dists[0].input_shape)
    input_shape_end = self.dists[0].input_shape[1:]
    for dist in self.dists[1:]:
      assert dist.input_shape[1:] == input_shape_end
      input_shape[0] += dist.input_shape[0]
    input_shape = tuple(input_shape)

    super().__init__(input_shape=input_shape, **kwargs)

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
    # Sample from each of our distributions
    keys = random.split(key, len(self.dists))
    xs = []
    log_px = 0.0
    for i, key in enumerate(keys):
      x, _log_px = self.dists[i].sample_and_log_prob(key, y=y)
      xs.append(x)
      log_px += _log_px

    # Concatenate the samples along the first axis
    x = jnp.concatenate(xs, axis=0)
    return x, log_px

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
    assert x.shape == self.input_shape

    # Figure out how to split the input
    split_indices = jnp.cumsum(jnp.array([0] + [dist.input_shape[0] for dist in self.dists]))
    splits = list(zip(split_indices[:-1], split_indices[1:]))

    # Compute the log prob of each sample
    log_px = 0.0
    for i, (start, end) in enumerate(splits):
      _x = x[start:end]
      log_px += self.dists[i].log_prob(_x, y=y, key=key)

    return log_px

################################################################################################################

class EmpiricalDistribution(ProbabilityDistribution):
  """An empirical distribution.  This can be used as a wrapper around data
  """

  data: Array

  def __init__(self,
               data: Array,
               **kwargs):
    """**Arguments**:

    - `input_shape`: The dimensions of the samples
    """
    self.data = data
    input_shape = data.shape[1:]
    super().__init__(input_shape=input_shape, **kwargs)

  def sample_and_log_prob(self):
    assert 0, "Can't compute"

  def log_prob(self):
    assert 0, "Can't compute"

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
    return random.choice(key, self.data, shape=(1,))[0]

  def train_iterator(self,
                     key: PRNGKeyArray,
                     batch_size: int) -> Mapping[str, Array]:
    """An iterator over the training data.  This is compatible with the
    Trainer class in this package.  Use like:
    ```python
    train_iter = empirical_dist.train_iterator(key, batch_size=128)
    data_batch = next(train_iter)
    ```

    **Arguments**:

    - `key`: The random number generator key.
    - `batch_size`: The batch size.

    **Returns**:
    An iterator over the training data that yields a dictionary
    with the key `x` and the value the training data.
    """

    total_choices = jnp.arange(self.data.shape[0])
    while True:
      key, _ = random.split(key, 2)
      idx = random.choice(key,
                          total_choices,
                          shape=(batch_size,),
                          replace=True)
      yield dict(x=self.data[idx])

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
               input_shape: Tuple[int],
               **kwargs):
    """**Arguments**:

    - `input_shape`: The dimensions of the samples
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
               x: Array,
               **kwargs) -> Array:
    """**Arguments**:

    - `x`: The point we want to compute logp(x) at.

    **Returns**:
    The log likelihood of x under the model.
    """
    assert x.shape == self.input_shape
    return jax.scipy.stats.norm.logpdf(x).sum()

  def sample_and_log_prob(self,
                          key: PRNGKeyArray,
                          **kwargs) -> Array:
    """**Arguments**:

    - `key`: The random number generator key.

    **Returns**:
    A single sample from the model with its log probability.
    """
    x = self.sample(key)
    log_px = self.log_prob(x)
    return x, log_px

################################################################################################################

class Uniform(ProbabilityDistribution):
  """This represents a Uniform distribution between 0 and 1"""

  def sample(self,
             key: PRNGKeyArray,
             y: Optional[Array] = None) -> Array:
    """**Arguments**:

    - `key`: The random number generator key.

    **Returns**:
    A single sample from the model.  Use eqx.filter_vmap to get more samples.
    """
    return random.uniform(key, shape=self.input_shape)

  def log_prob(self,
               x: Array,
               **kwargs) -> Array:
    """**Arguments**:

    - `x`: The point we want to compute logp(x) at.

    **Returns**:
    The log likelihood of x under the model.
    """
    assert x.shape == self.input_shape
    return jnp.array(0.0)

  def sample_and_log_prob(self,
                          key: PRNGKeyArray,
                          **kwargs) -> Array:
    """**Arguments**:

    - `key`: The random number generator key.

    **Returns**:
    A single sample from the model with its log probability.
    """
    x = self.sample(key)
    log_px = self.log_prob(x)
    return x, log_px

################################################################################################################

class Logistic(ProbabilityDistribution):
  """This represents a Logistic distribution"""

  def sample(self,
             key: PRNGKeyArray,
             y: Optional[Array] = None) -> Array:
    """**Arguments**:

    - `key`: The random number generator key.

    **Returns**:
    A single sample from the model.  Use eqx.filter_vmap to get more samples.
    """
    return random.logistic(key, shape=self.input_shape)

  def log_prob(self,
               x: Array,
               **kwargs) -> Array:
    """**Arguments**:

    - `x`: The point we want to compute logp(x) at.

    **Returns**:
    The log likelihood of x under the model.
    """
    assert x.shape == self.input_shape
    return jax.scipy.stats.logistic.logpdf(x).sum()

  def sample_and_log_prob(self,
                          key: PRNGKeyArray,
                          **kwargs) -> Array:
    """**Arguments**:

    - `key`: The random number generator key.

    **Returns**:
    A single sample from the model with its log probability.
    """
    x = self.sample(key)
    log_px = self.log_prob(x)
    return x, log_px

################################################################################################################

if __name__ == '__main__':
  from debug import *
  import matplotlib.pyplot as plt
  import generax.util as util
  # switch to x64
  jax.config.update("jax_enable_x64", True)

  key = random.PRNGKey(0)
  x = random.normal(key, shape=(20, 10, 3, 4))

  # Test the product distribution
  p1 = Gaussian(input_shape=(5, 3, 4))
  p2 = Gaussian(input_shape=(3, 3, 4))
  p3 = Gaussian(input_shape=(2, 3, 4))
  p_prod = ProductDistribution(p1, p2, p3)

  p_comp = Gaussian(input_shape=(10, 3, 4))

  x, log_px1 = p_prod.sample_and_log_prob(key)
  log_px2 = p_comp.log_prob(x)
  log_px3 = p_prod.log_prob(x)
  assert jnp.allclose(log_px1, log_px2)
  assert jnp.allclose(log_px3, log_px2)


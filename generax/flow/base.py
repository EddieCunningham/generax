import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
import einops
import equinox as eqx
from abc import ABC, abstractmethod
from generax.distributions import ProbabilityDistribution
from jaxtyping import Array, PRNGKeyArray
import generax.nn.util as util

__all__ = ['BijectiveTransform',
           'Sequential']

class BijectiveTransform(eqx.Module, ABC):
  """This represents a bijective transformation.  Every bijective transformation
  only `(x, key)` on initialization in order to be able to do data
  dependent initialization.

  *** x MUST be a batched array in order for the layers to work correctly. ***

  **Atributes**:

  - `input_shape`: The input shape.  Output shape will have the same dimensionality
                  as the input.
  - `cond_shape`: The shape of the conditioning information.  If there is no
                  conditioning information, this is None.

  **Methods**:

  - `__call__(x,y,inverse) -> (z,log_det)`: Apply the transformation to the input.
    - `x`: Input array of shape `input_shape`
    - `y`: Conditioning information of shape `cond_shape`
    - `inverse`: Whether to invert the transformation
    - `z`: The transformed input
    - `log_det`: The log determinant of the Jacobian
  """

  input_shape: Union[int, Tuple[int]] = eqx.field(static=True)
  cond_shape: Union[None, Union[int,Tuple[int]]] = eqx.field(static=True)

  def __init__(self,
               *_,
               x: Array,
               y: Optional[Array] = None,
               key: PRNGKeyArray,
               **kwargs):
    """**Arguments**:

    - `input_shape`: The input shape.  Output size is the same as shape.
    """
    self.input_shape = x.shape[1:]
    self.cond_shape = y.shape[1:] if y is not None else None

  @abstractmethod
  def __call__(self,
               x: Array,
               y: Optional[Array] = None,
               inverse: bool=False) -> Array:
    """**Arguments**:

    - `x`: The input to the transformation
    - `y`: The conditioning information
    - `inverse`: Whether to inverse the transformation

    **Returns**:
    (z, log_det)
    """
    pass

################################################################################################################

class Sequential(BijectiveTransform):
  """A sequence of bijective transformations.  Accepts a sequence
   of `BijectiveTransform` initializers.

  ```python
  # Intented usage:
  composition = Sequential(LayerInit1,
                           LayerInit2,
                           x=x,
                           key=key)
  z = composition(x)

  # Equicalent to the following:
  layer1 = LayerInit1(**init_kwargs)
  u = layer1(x)
  **init_kwargs = dict(input_shape=input_shape, x=u, key=key)
  layer2 = LayerInit2(**init_kwargs)
  z = layer2(u)
  ```

  **Attributes**:
  - `n_layers`: The number of layers in the composition
  - `layers`: A tuple of the layers in the composition
  """

  n_layers: int = eqx.field(static=True)
  layers: Tuple[BijectiveTransform]

  def __init__(self,
               *layer_types: Sequence[type],
               x: Array,
               y: Optional[Array] = None,
               key: PRNGKeyArray,
               **kwargs):
    """**Arguments**:

    - `layer_types`: A sequence of `BijectiveTransform` initializers.
    """
    super().__init__(x=x, y=y, key=key, **kwargs)
    self.n_layers = len(layer_types)

    # We need to initialize each of the layers
    keys = random.split(key, self.n_layers)

    layers = [None for _ in range(self.n_layers)]
    for i, (k, layer_type) in enumerate(zip(keys, layer_types)):
      layers[i] = layer_type(x=x, y=y, key=k, **kwargs)
      x, log_det = eqx.filter_vmap(layers[i])(x)
      assert log_det.shape == (x.shape[0],)

    self.layers = tuple(layers)

  @jax.named_scope("generx.flows.Sequential")
  def __call__(self,
               x: Array,
               y: Optional[Array] = None,
               inverse: bool=False) -> Array:
    """**Arguments**:

    - `x`: The input to the transformation
    - `y`: The conditioning information
    - `inverse`: Whether to inverse the transformation

    **Returns**:
    (z, log_det)
    """
    delta_logpx = 0.0
    layers = reversed(self.layers) if inverse else self.layers
    for layer in layers:
      x, log_det = layer(x, y=y, inverse=inverse)
      delta_logpx += log_det
    return x, delta_logpx

  def __getitem__(self, i: Union[int, slice]) -> Callable:
      if isinstance(i, int):
          return self.layers[i]
      elif isinstance(i, slice):
          return Sequential(self.layers[i])
      else:
          raise TypeError(f"Indexing with type {type(i)} is not supported")

  def __iter__(self):
      yield from self.layers

  def __len__(self):
      return len(self.layers)

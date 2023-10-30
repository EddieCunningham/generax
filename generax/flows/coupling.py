import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
import einops
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray
import generax.util.misc as misc
from generax.flows.base import BijectiveTransform
import numpy as np
from generax.nn.resnet import ResNet

__all__ = ['Coupling']

class RavelParameters(eqx.Module):
  """Flatten and concatenate the parameters of a eqx.Module
  """

  shapes_and_sizes: Sequence[Tuple[Tuple[int], int]] = eqx.field(static=True)
  flat_params_size: Tuple[int] = eqx.field(static=True)
  static: Any = eqx.field(static=True)
  treedef: Any = eqx.field(static=True)
  indices: np.ndarray = eqx.field(static=True)

  def __init__(self, module):

    # Split the parameters into dynamic and static
    params, self.static = eqx.partition(module, eqx.is_array)

    # Flatten the parameters so that we can extract its sizes
    leaves, self.treedef = jax.tree_util.tree_flatten(params)

    # Get the shape and size of each leaf
    self.shapes_and_sizes = [(leaf.shape, leaf.size) for leaf in leaves]

    # Flatten the parameters
    flat_params = jnp.concatenate([leaf.ravel() for leaf in leaves])

    # Keep track of the size of the flattened parameters
    self.flat_params_size = flat_params.size

    # Keep track of the split points for each paramter in the flattened array
    self.indices = np.cumsum(np.array([0] + [size for _, size in self.shapes_and_sizes]))

  def __call__(self, flat_params: Array) -> eqx.Module:
    flat_params = flat_params.ravel() # Flatten the parameters completely
    leaves = []
    for i, (shape, size) in enumerate(self.shapes_and_sizes):

      # Extract each leaf from the flattened parameters and reshape it
      buffer = flat_params[self.indices[i]: self.indices[i + 1]]
      if buffer.size != misc.list_prod(shape):
        raise ValueError(f'Expected total size of {misc.list_prod(shape)} but got {buffer.size}')
      leaf = buffer.reshape(shape)
      leaves.append(leaf)

    # Turn the leaves back into a tree
    params = jax.tree_util.tree_unflatten(self.treedef, leaves)

    return eqx.combine(params, self.static)

################################################################################################################

class Coupling(BijectiveTransform):
  """Parametrize a flow over half of the inputs using the other half.
  The conditioning network will be fixed

  ```python
  # Intended usage:
  layer = Coupling(BijectiveTransform,
                   eqx.Module)
  z, log_det = layer(x, y)
  ```

  **Attributes**:
  - `transform`: The bijective transformation to use.
  - `scale`: A scalar that we'll use to start with small parameter values
  - `net`: The neural network to use.
  """

  net: eqx.Module
  scale: Array
  params_to_transform: RavelParameters

  def __init__(self,
               transform: BijectiveTransform,
               net: eqx.Module,
               input_shape: Tuple[int],
               cond_shape: Optional[Tuple[int]] = None,
               *,
               key: PRNGKeyArray,
               **kwargs):
    """**Arguments**:

    - `transform`: The bijective transformation to use.
    - `net`: The neural network to generate the transform parameters.
    - `input_shape`: The shape of the input
    - `cond_shape`: The shape of the conditioning information
    - `key`: A `jax.random.PRNGKey` for initialization
    """
    super().__init__(input_shape=input_shape,
                     cond_shape=cond_shape,
                     **kwargs)

    # Check the input shapes of the transform and network
    x1_shape, x2_shape = self.get_split_shapes(input_shape)
    assert transform.input_shape == x1_shape

    self.net = net

    # Use this to turn an eqx module into an array and vice-versa
    self.params_to_transform = RavelParameters(transform)

    # Also initialize the parameters to be close to 0
    self.scale = random.normal(key, (1,))*0.01

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
    A new layer with the parameters initialized.
    """
    assert x.shape[1:] == self.input_shape, 'Only works on batched data'
    x1, x2 = self.split(x)
    net = self.net.data_dependent_init(x2, y=y, key=key)

    # Turn the new parameters into a new module
    get_net = lambda tree: tree.net
    updated_layer = eqx.tree_at(get_net, self, net)
    return updated_layer

  def split(self, x: Array) -> Tuple[Array, Array]:
    """Split the input into two halves."""
    split_dim = x.shape[-1]//2
    x1, x2 = x[..., :split_dim], x[..., split_dim:]
    return x1, x2

  @classmethod
  def get_split_shapes(cls, input_shape: Tuple[int]) -> Tuple[Tuple[int]]:
    split_dim = input_shape[-1]//2
    x1_dim, x2_dim = split_dim, input_shape[-1] - split_dim
    x1_shape = input_shape[:-1] + (x1_dim,)
    x2_shape = input_shape[:-1] + (x2_dim,)
    return x1_shape, x2_shape

  @classmethod
  def get_net_output_shapes(cls,
                            input_shape: Tuple[int],
                            transform: BijectiveTransform) -> Tuple[Tuple[int],int]:
    """
    **Arguments**:
    - `input_shape`: The shape of the input
    - `transform`: The bijective transformation to use.

    **Returns**:
    - `net_output_size`: The size of the output of the neural network.  This is a single integer
                         because the network is expected to produce a single vector.
    """
    x1_shape, x2_shape = cls.get_split_shapes(input_shape)
    if x1_shape != transform.input_shape:
      raise ValueError(f'The transform {transform} needs to have an input shape equal to {x1_shape}.  Use Coupling.get_input_shapes to get this shape.')
    params_to_transform = RavelParameters(transform)
    net_output_size = params_to_transform.flat_params_size
    return net_output_size


  @classmethod
  def get_net_input_and_output_shapes(cls,
                                      input_shape: Tuple[int],
                                      transform: BijectiveTransform) -> Tuple[Tuple[int],int]:
    """
    **Arguments**:
    - `input_shape`: The shape of the input
    - `transform`: The bijective transformation to use.

    **Returns**:
    - `net_input_shape`: The shape of the input to the neural network.  This is a tuple of ints
    - `net_output_size`: The size of the output of the neural network.  This is a single integer
                         because the network is expected to produce a single vector.
    """
    x1_shape, x2_shape = cls.get_split_shapes(input_shape)
    net_input_shape = x2_shape
    params_to_transform = RavelParameters(transform)
    net_output_size = params_to_transform.flat_params_size
    return x1_shape, net_input_shape, net_output_size

  def __call__(self,
               x: Array,
               y: Optional[Array] = None,
               inverse: bool = False,
               **kwargs) -> Array:
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
    params = self.net(x2, y=y, **kwargs)
    params *= self.scale
    assert params.size == self.params_to_transform.flat_params_size

    # Apply the transformation to x1 given x2
    transform = self.params_to_transform(params)
    z1, log_det = transform(x1, y=y, inverse=inverse, **kwargs)

    z = jnp.concatenate([z1, x2], axis=-1)
    return z, log_det

################################################################################################################

if __name__ == '__main__':
  from debug import *
  import matplotlib.pyplot as plt
  from generax.flows.base import Sequential
  from generax.flows.affine import DenseAffine, ShiftScale
  from generax.flows.reshape import Reverse
  from generax.distributions.base import Gaussian

  # Turn on x64
  from jax.config import config
  config.update("jax_enable_x64", True)

  key = random.PRNGKey(0)
  x, y = random.normal(key, shape=(2, 10, 5))
  cond_shape = y.shape[1:]
  y, cond_shape = None, None

  input_shape = x.shape[1:]
  transform_input_shape, net_input_shape = Coupling.get_split_shapes(input_shape)

  transform = ShiftScale(input_shape=transform_input_shape,
                            key=key)
  net_output_size = Coupling.get_net_output_shapes(input_shape, transform)

  net = ResNet(input_shape=net_input_shape,
                   working_size=4,
                    hidden_size=4,
                    out_size=net_output_size,
                    n_blocks=2,
                    filter_shape=(3, 3),
                    cond_shape=cond_shape,
                    key=key)

  transform = ShiftScale(input_shape=transform_input_shape,
                            key=key)
  layer = Coupling(transform,
                   net,
                   input_shape=input_shape,
                    cond_shape=cond_shape,
                   key=key)

  layer(x[0])
  layer = layer.data_dependent_init(x, y=y, key=key)

  z, log_det = eqx.filter_vmap(layer)(x)

  z, log_det = layer(x[0])
  x_reconstr, log_det2 = layer(z, inverse=True)

  G = jax.jacobian(lambda x: layer(x)[0])(x[0])
  log_det_true = jnp.linalg.slogdet(G)[1]

  assert jnp.allclose(log_det, log_det_true)
  assert jnp.allclose(x[0], x_reconstr)

  import pdb; pdb.set_trace()
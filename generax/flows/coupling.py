import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
import einops
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray
import generax.util.misc as misc
from generax.flows.base import BijectiveTransform, TimeDependentBijectiveTransform
import numpy as np
from generax.nn.resnet import ResNet

__all__ = ['Coupling',
           'TimeDependentCoupling',
           'RavelParameters',
           'TimeDependentWrapper']

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

  def flatten_params(self, module: eqx.Module) -> Array:
    # Split the parameters into dynamic and static
    params, _ = eqx.partition(module, eqx.is_array)

    # Flatten the parameters so that we can extract its sizes
    leaves, _ = jax.tree_util.tree_flatten(params)

    # Flatten the parameters
    flat_params = jnp.concatenate([leaf.ravel() for leaf in leaves])
    return flat_params

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
  # Example of intended usage:

  def initialize_scale(transform_input_shape, key):
    return ShiftScale(input_shape=transform_input_shape,
                      key=key,
                      **kwargs)

  def initialize_network(net_input_shape, net_output_size, key):
    return ResNet(input_shape=net_input_shape,
                  out_size=net_output_size,
                  key=key,
                  **kwargs)

  layer = Coupling(transform_init=initialize_scale,
                   net_init=initialize_network,
                   input_shape=input_shape,
                   cond_shape=cond_shape,
                   key=key,
                   reverse_conditioning=True,
                   split_dim=1)

  z, log_det = layer(x, y)
  ```

  **Attributes**:
  - `params_to_transform`: A module that turns an array of parameters into an eqx.Module.
  - `scale`: A scalar that we'll use to start with small parameter values
  - `net`: The neural network to use.
  """

  net: eqx.Module
  scale: Array
  params_to_transform: RavelParameters

  split_dim: Optional[int] = eqx.field(static=True)
  reverse_conditioning: bool = eqx.field(static=True)

  def __init__(self,
               transform_init: Callable[[Tuple[int]],BijectiveTransform],
               net_init: Callable[[Tuple[int],int],eqx.Module],
               input_shape: Tuple[int],
               cond_shape: Optional[Tuple[int]] = None,
               split_dim: Optional[int] = None,
               reverse_conditioning: Optional[bool] = False,
               *,
               key: PRNGKeyArray,
               **kwargs):
    """**Arguments**:

    - `transform`: The bijective transformation to use.
    - `net`: The neural network to generate the transform parameters.
    - `input_shape`: The shape of the input
    - `cond_shape`: The shape of the conditioning information
    - `split_dim`: The number of dimension to split the last axis on.  If `None`, defaults to `dim//2`.
    - `reverse_conditioning`: If `True`, condition on the first part of the input instead of the second part.
    - `key`: A `jax.random.PRNGKey` for initialization
    """
    super().__init__(input_shape=input_shape,
                     cond_shape=cond_shape,
                     **kwargs)

    k1, k2 = random.split(key, 2)

    self.split_dim = split_dim if split_dim is not None else input_shape[-1]//2
    self.reverse_conditioning = reverse_conditioning

    # Get the shapes of the input to the transform and the network
    transform_input_shape, net_input_shape = self.get_split_shapes(input_shape)
    transform = transform_init(transform_input_shape, key=k1)

    net_output_size = self.get_net_output_shapes(input_shape, transform)
    net = net_init(net_input_shape, net_output_size, key=k2)

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
    x1, x2 = x[..., :self.split_dim], x[..., self.split_dim:]
    if self.reverse_conditioning:
      return x2, x1
    return x1, x2

  def combine(self, x1: Array, x2: Array) -> Array:
    """Combine the two halves of the input."""
    if self.reverse_conditioning:
      return jnp.concatenate([x2, x1], axis=-1)
    return jnp.concatenate([x1, x2], axis=-1)

  def get_split_shapes(self,
                       input_shape: Tuple[int]) -> Tuple[Tuple[int]]:
    x1_dim, x2_dim = self.split_dim, input_shape[-1] - self.split_dim
    x1_shape = input_shape[:-1] + (x1_dim,)
    x2_shape = input_shape[:-1] + (x2_dim,)
    if self.reverse_conditioning:
      return x2_shape, x1_shape
    return x1_shape, x2_shape

  def get_net_output_shapes(self,
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
    x1_shape, x2_shape = self.get_split_shapes(input_shape)
    if x1_shape != transform.input_shape:
      raise ValueError(f'The transform {transform} needs to have an input shape equal to {x1_shape}.  Use `get_input_shapes` to get this shape.')
    params_to_transform = RavelParameters(transform)
    net_output_size = params_to_transform.flat_params_size
    return net_output_size

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

    z = self.combine(z1, x2)
    return z, log_det

################################################################################################################

class TimeDependentCoupling(Coupling, TimeDependentBijectiveTransform):
  """Time dependent coupling transform.  At t=0, this will pass parameters of 0s
  to the transform.
  ```

  **Attributes**:
  - `params_to_transform`: A module that turns an array of parameters into an eqx.Module.
  - `scale`: A scalar that we'll use to start with small parameter values
  - `net`: The neural network to use.
  """

  def data_dependent_init(self,
                          t: Array,
                          x: Array,
                          y: Optional[Array] = None,
                          key: PRNGKeyArray = None) -> BijectiveTransform:
    """Initialize the parameters of the layer based on the data.

    **Arguments**:

    - `t`: The time to initialize the parameters with.
    - `x`: The data to initialize the parameters with.
    - `y`: The conditioning information
    - `key`: A `jax.random.PRNGKey` for initialization

    **Returns**:
    A new layer with the parameters initialized.
    """
    assert x.shape[1:] == self.input_shape, 'Only works on batched data'
    x1, x2 = self.split(x)
    net = self.net.data_dependent_init(t, x2, y=y, key=key)

    # Turn the new parameters into a new module
    def get_net(tree): return tree.net
    updated_layer = eqx.tree_at(get_net, self, net)
    return updated_layer

  def __call__(self,
               t: Array,
               xt: Array,
               y: Optional[Array] = None,
               inverse: bool=False,
               **kwargs) -> Array:
    """**Arguments**:

    - `xt`: The input to the transformation.  If inverse=True, then should be x0
    - `y`: The conditioning information
    - `inverse`: Whether to inverse the transformation

    **Returns**:
    (x0, log_det)
    """
    assert xt.shape == self.input_shape, 'Only works on unbatched data'

    # Split the input into two halves
    x1, x2 = self.split(xt)
    params = self.net(t, x2, y=y, **kwargs)
    params *= self.scale*t
    assert params.size == self.params_to_transform.flat_params_size

    # Apply the transformation to x1 given x2
    transform = self.params_to_transform(params)
    z1, log_det = transform(x1, y=y, inverse=inverse, **kwargs)

    z = self.combine(z1, x2)
    return z, log_det

################################################################################################################

class TimeDependentWrapper(TimeDependentBijectiveTransform):
  """Turn a bijective transform into a time dependent bijective transformation
  where t is multiplied by the parameters.  If the transform is initialized correctly,
  then when t=0, the transform should be equal to the identity transform.
  """

  transform: BijectiveTransform
  params_to_transform: RavelParameters

  def __init__(self, transform: BijectiveTransform):
    super().__init__(input_shape=transform.input_shape,
                     cond_shape=transform.cond_shape)
    self.transform = transform
    eqx.module_update_wrapper(self)

    # Use this to turn an eqx module into an array and vice-versa
    self.params_to_transform = RavelParameters(transform)

  @property
  def __wrapped__(self):
    return self.transform

  def __call__(self,
               t: Array,
               x: Array,
               y: Optional[Array] = None,
               inverse: bool = False,
               **kwargs) -> Array:
    """**Arguments**:

    - `t`: The time to evaluate at
    - `x`: The input to the transformation
    - `y`: The conditioning information
    - `inverse`: Whether to inverse the transformation

    **Returns**:
    (z, log_det)
    """
    assert x.shape == self.input_shape, 'Only works on unbatched data'

    # Flatten the parameters of the base transform and multiply by t
    params, static = eqx.partition(self.transform, eqx.is_inexact_array)
    leaves, _ = jax.tree_util.tree_flatten(params)
    flat_params = jnp.concatenate([leaf.ravel() for leaf in leaves])
    transform = self.params_to_transform(t*flat_params)

    return transform(x, y=y, inverse=inverse, **kwargs)

################################################################################################################

if __name__ == '__main__':
  from debug import *
  import matplotlib.pyplot as plt
  from generax.flows.base import Sequential
  from generax.flows.affine import DenseAffine, ShiftScale
  from generax.flows.reshape import Reverse
  from generax.distributions.base import Gaussian
  import generax as gx

  # Turn on x64
  from jax.config import config
  config.update("jax_enable_x64", True)

  key = random.PRNGKey(0)
  x, y = random.normal(key, shape=(2, 10, 5))
  t = random.uniform(key, shape=(x.shape[0],))
  cond_shape = y.shape[1:]
  y, cond_shape = None, None

  input_shape = x.shape[1:]

  def initialize_scale(transform_input_shape, key):
    return ShiftScale(input_shape=transform_input_shape,
                      key=key)

  def initialize_network(net_input_shape, net_output_size, key):
    return gx.TimeDependentResNet(input_shape=net_input_shape,
                  working_size=4,
                  hidden_size=4,
                  out_size=net_output_size,
                  n_blocks=2,
                  filter_shape=(3, 3),
                  cond_shape=cond_shape,
                  key=key)

  layer = TimeDependentCoupling(transform_init=initialize_scale,
                   net_init=initialize_network,
                   input_shape=input_shape,
                   cond_shape=cond_shape,
                   key=key,
                   reverse_conditioning=True,
                   split_dim=1)

  layer(t[0], x[0])
  layer = layer.data_dependent_init(t, x, y=y, key=key)

  z, log_det = eqx.filter_vmap(layer)(t, x)

  z, log_det = layer(t[0], x[0])
  x_reconstr, log_det2 = layer(t[0], z, inverse=True)

  G = jax.jacobian(lambda x: layer(t[0], x)[0])(x[0])
  log_det_true = jnp.linalg.slogdet(G)[1]

  assert jnp.allclose(log_det, log_det_true)
  assert jnp.allclose(x[0], x_reconstr)

  import pdb; pdb.set_trace()
import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
import einops
import equinox as eqx
from abc import ABC, abstractmethod
from jaxtyping import Array, PRNGKeyArray
import generax.util.misc as misc
import generax.util as util
import lineax as lx

__all__ = ['BijectiveTransform',
           'InjectiveTransform',
           'TimeDependentBijectiveTransform',
           'Sequential',
           'InjectiveSequential']

class BijectiveTransform(eqx.Module, ABC):
  """This represents a bijective transformation.

  **Atributes**:

  - `input_shape`: The input shape.  Output shape will have the same dimensionality as the input.
  - `cond_shape`: The shape of the conditioning information.  If there is no
                  conditioning information, this is None.
  """

  input_shape: Tuple[int] = eqx.field(static=True)
  cond_shape: Union[None, Tuple[int]] = eqx.field(static=True)

  def __init__(self,
               *_,
               input_shape: Tuple[int],
               cond_shape: Union[None, Tuple[int]] = None,
               **kwargs):
    """**Arguments**:

    - `input_shape`: The input shape.  Output size is the same as shape.
    - `cond_shape`: The shape of the conditioning information.  If there is no
    """
    super().__init__(**kwargs)

    assert isinstance(input_shape, tuple) or isinstance(input_shape, list)
    self.input_shape = tuple(input_shape)
    if cond_shape is not None:
      assert isinstance(cond_shape, tuple) or isinstance(cond_shape, list)
      self.cond_shape = tuple(cond_shape)
    else:
       self.cond_shape = None

  def data_dependent_init(self,
                          x: Array,
                          y: Optional[Array] = None,
                          key: PRNGKeyArray = None):
    """Initialize the parameters of the layer based on the data.

    **Arguments**:

    - `x`: The data to initialize the parameters with.
    - `y`: The conditioning information
    - `key`: A `jax.random.PRNGKey` for initialization

    **Returns**:
    A new layer with the parameters initialized.
    """
    return self

  @abstractmethod
  def __call__(self,
               x: Array,
               y: Optional[Array] = None,
               inverse: bool=False,
               **kwargs) -> Array:
    """**Arguments**:

    - `x`: The input to the transformation
    - `y`: The conditioning information
    - `inverse`: Whether to inverse the transformation

    **Returns**:
    (z, log_det)
    """
    pass

  def to_base_space(self,
                    x: Array,
                    y: Optional[Array] = None,
                    **kwargs) -> Array:
    """Apply the inverse transformation.

    **Arguments**:

    - `x`: The input to the transformation
    - `y`: The conditioning information

    **Returns**:
    z
    """
    return self(x, y=y, **kwargs)[0]

  def to_data_space(self,
                    z: Array,
                    y: Optional[Array] = None,
                    **kwargs) -> Array:
    """Apply the inverse transformation.

    **Arguments**:

    - `z`: The input to the transformation
    - `y`: The conditioning information

    **Returns**:
    x
    """
    return self(z, y=y, inverse=True, **kwargs)[0]

  def inverse(self,
              x: Array,
              y: Optional[Array] = None,
              **kwargs) -> Array:
    """Apply the inverse transformation.

    **Arguments**:

    - `x`: The input to the transformation
    - `y`: The conditioning information

    **Returns**:
    (z, log_det)
    """
    return self(x, y=y, inverse=True, **kwargs)

################################################################################################################

class InjectiveTransform(BijectiveTransform, ABC):
  """This represents an injective transformation.  This is a special case of a bijective
  transformation.

  **Atributes**:

  - `input_shape`: The input shape.
  - `output_shape`: The output shape.
  - `cond_shape`: The shape of the conditioning information.  If there is no
                  conditioning information, this is None.
  """
  output_shape: Tuple[int] = eqx.field(static=True)

  def __init__(self,
               *_,
               input_shape: Tuple[int],
               output_shape: Tuple[int],
               cond_shape: Union[None, Tuple[int]] = None,
               **kwargs):
    """**Arguments**:

    - `input_shape`: The input shape.
    - `output_shape`: The output shape.
    - `cond_shape`: The shape of the conditioning information.  If there is no
    """
    super().__init__(input_shape=input_shape,
                     cond_shape=cond_shape,
                     **kwargs)
    assert isinstance(output_shape, tuple) or isinstance(output_shape, list)
    self.output_shape = output_shape

  def project(self,
              x: Array,
              y: Optional[Array] = None,
              **kwargs) -> Array:
    """Project a point onto the image of the transformation.

    **Arguments**:

    - `x`: The input to the transformation
    - `y`: The conditioning information

    **Returns**:
    z
    """
    z, _ = self(x, y=y, **kwargs)
    x_proj, _ = self(z, y=y, inverse=True, **kwargs)
    return x_proj

  def log_determinant(self,
                      z: Array,
                      y: Optional[Array] = None,
                      **kwargs) -> Array:
    """Compute -0.5*log(det(J^TJ))

    **Arguments**:

    - `z`: An element of the base space

    **Returns**:
    The log determinant of (J^TJ)^-0.5
    """

    def jvp(v_flat):
      v = v_flat.reshape(z.shape)
      _, (Jv) = jax.jvp(self.to_data_space, (z,), (v,))
      return Jv.ravel()

    z_dim = util.list_prod(z.shape)
    eye = jnp.eye(z_dim)
    J = jax.vmap(jvp, in_axes=1, out_axes=1)(eye)
    return -0.5*jnp.linalg.slogdet(J.T@J)[1]

  def log_determinant_surrogate(z: Array,
                                transform: eqx.Module,
                                method: str = 'brute_force',
                                key: PRNGKeyArray = None,
                                **kwargs) -> Array:
    """Compute a term that has the same expected gradient as `-0.5*log_det(J^TJ))`.

    If `method='brute_force'`, then this is just -0.5*log(det(J^TJ)).
    If `method='iterative'`, then this is a term that has the same expected gradient.

    **Arguments**:

    - `z`: An element of the base space
    - `method`: How to compute the log determinant.  Options are:
      - `brute_force`: Compute the entire Jacobian
      - `iterative`: Use conjugate gradient (https://arxiv.org/pdf/2106.01413.pdf)
    - `key`: A `jax.random.PRNGKey` for initialization.  Needed for some methods

    **Returns**:
    The log determinant of J^TJ or a term that has the same gradient
    """

    def jvp(v_flat):
      v = v_flat.reshape(z.shape)
      _, (Jv) = jax.jvp(transform, (z,), (v,))
      return Jv.ravel()

    if method == 'brute_force':
      z_dim = util.list_prod(z.shape)
      eye = jnp.eye(z_dim)
      J = jax.vmap(jvp, in_axes=1, out_axes=1)(eye)
      return -0.5*jnp.linalg.slogdet(J.T@J)[1]

    elif method == 'iterative':

      def vjp(v_flat):
        x, vjp = jax.vjp(transform, z)
        v = v_flat.reshape(x.shape)
        return vjp(v)[0].ravel()

      def vjp_jvp(v_flat):
        return vjp(jvp(v_flat))

      v = random.normal(key, shape=z.shape)

      operator = lx.FunctionLinearOperator(vjp_jvp, v, tags=lx.positive_semidefinite_tag)
      solver = lx.CG(rtol=1e-6, atol=1e-6)
      JTJinv_v = lx.linear_solve(operator, v, solver).value
      JTJ_v = vjp_jvp(v)
      return -0.5*jnp.vdot(jax.lax.stop_gradient(JTJinv_v), JTJ_v)

################################################################################################################

class TimeDependentBijectiveTransform(BijectiveTransform):
  """Time dependent bijective transform.  This will help us build simple probability paths.
  Non-inverse mode goes t -> 0 while inverse mode goes t -> 1.

  **Atributes**:

  - `input_shape`: The input shape.  Output shape will have the same dimensionality
                  as the input.
  - `cond_shape`: The shape of the conditioning information.  If there is no
                  conditioning information, this is None.
  """

  def data_dependent_init(self,
                          t: Array,
                          xt: Array,
                          y: Optional[Array] = None,
                          key: PRNGKeyArray = None):
    """Initialize the parameters of the layer based on the data.

    **Arguments**:

    - `t`: Time.
    - `x`: The data to initialize the parameters with.
    - `y`: The conditioning information
    - `key`: A `jax.random.PRNGKey` for initialization

    **Returns**:
    A new layer with the parameters initialized.
    """
    return self

  @abstractmethod
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
    pass

  def inverse(self,
              t: Array,
              x0: Array,
              y: Optional[Array] = None,
              **kwargs) -> Array:
    """Apply the inverse transformation.

    **Arguments**:

    - `x`: The input to the transformation
    - `y`: The conditioning information

    **Returns**:
    (xt, log_det)
    """
    return self(t, x0, y=y, inverse=True, **kwargs)

  def vector_field(self,
                   t: Array,
                   xt: Array,
                   y: Optional[Array] = None,
                   **kwargs) -> Array:
    """The vector field that samples evolve on as t changes

    **Arguments**:

    - `t`: Time.
    - `x0`: A point in the base space.
    - `y`: The conditioning information.

    **Returns**:
    The vector field that samples evolve on at (t, x).
    """
    raise NotImplementedError

################################################################################################################

class Sequential(BijectiveTransform):
  """A sequence of bijective transformations.  Accepts a sequence
   of `BijectiveTransform` initializers.

  ```python
  # Intented usage:
  layer1 = MyTransform(...)
  layer2 = MyTransform(...)
  transform = Sequential(layer1, layer2)
  ```

  **Attributes**:
  - `n_layers`: The number of layers in the composition
  - `layers`: A tuple of the layers in the composition
  """

  n_layers: int = eqx.field(static=True)
  layers: Tuple[BijectiveTransform]

  def __init__(self,
               *layers: Sequence[BijectiveTransform],
               **kwargs):
    """**Arguments**:

    - `layers`: A sequence of `BijectiveTransform`.
    """
    input_shape = layers[0].input_shape
    cond_shape = layers[0].cond_shape
    # Check that all of the layers have the same cond shape
    for layer in layers:
      assert layer.cond_shape == cond_shape

    super().__init__(input_shape=input_shape,
                     cond_shape=cond_shape,
                     **kwargs)

    self.layers = tuple(layers)
    self.n_layers = len(layers)

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

    # We need to initialize each of the layers
    keys = random.split(key, self.n_layers)

    new_layers = []
    for i, (layer, key) in enumerate(zip(self.layers, keys)):
      new_layer = layer.data_dependent_init(x=x, y=y, key=key)
      new_layers.append(new_layer)
      x, _ = eqx.filter_vmap(new_layer)(x)
    new_layers = tuple(new_layers)

    # Turn the new parameters into a new module
    get_layers = lambda tree: tree.layers
    updated_layer = eqx.tree_at(get_layers, self, new_layers)
    return updated_layer

  def __call__(self,
               x: Array,
               y: Optional[Array] = None,
               inverse: bool=False,
               **kwargs) -> Array:
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
      x, log_det = layer(x, y=y, inverse=inverse, **kwargs)
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

################################################################################################################

class InjectiveSequential(Sequential, InjectiveTransform):
  """A sequence of injective or bijective transformations.
  """

  def __init__(self,
               *layers: Sequence[BijectiveTransform],
               **kwargs):
    """**Arguments**:

    - `layers`: A sequence of `BijectiveTransform`.
    """
    input_shape = layers[0].input_shape
    cond_shape = layers[0].cond_shape

    # Check that all of the layers have the same cond shape
    # and that the output shape of each layer matches the input shape of the next layer
    layer_iter = iter(zip(layers[:-1], layers[1:]))
    for l1, l2 in layer_iter:
      assert l1.cond_shape == cond_shape
      if isinstance(l1, InjectiveTransform):
        assert l1.output_shape == l2.input_shape
    assert l2.cond_shape == cond_shape

    if isinstance(l2, InjectiveTransform):
      output_shape = l2.output_shape
    else:
      output_shape = l2.input_shape

    InjectiveTransform.__init__(self,
                                     input_shape=input_shape,
                                     output_shape=output_shape,
                                     cond_shape=cond_shape,
                                     **kwargs)

    self.layers = tuple(layers)
    self.n_layers = len(layers)

################################################################################################################

if __name__ == '__main__':
  from debug import *
  import matplotlib.pyplot as plt
  from generax.flows.affine import ShiftScale

  key = random.PRNGKey(0)
  x = random.normal(key, shape=(10, 2, 2, 2))

  # composition = Sequential(ShiftScale,
  #                          ShiftScale,
  #                          ShiftScale,
  #                          x=x,
  #                          key=key)
  # z, log_det = eqx.filter_vmap(composition)(x)

  # import pdb; pdb.set_trace()

  input_shape = x.shape[1:]
  k1, k2, k3 = random.split(key, 3)
  layer1 = ShiftScale(input_shape=x.shape[1:], key=key)
  layer2 = ShiftScale(input_shape=x.shape[1:], key=key)
  layer3 = ShiftScale(input_shape=x.shape[1:], key=key)

  layer = Sequential(layer1, layer2, layer3)

  x = random.normal(key, shape=(10, 2, 2, 2))
  layer = layer.data_dependent_init(x, key=key)

  # layer = PLUAffine(x=x, key=key)
  z, log_det = eqx.filter_vmap(layer)(x)

  z, log_det = layer(x[0])
  x_reconstr, log_det2 = layer(z, inverse=True)

  G = jax.jacobian(lambda x: layer(x)[0])(x[0])
  G = einops.rearrange(G, 'h1 w1 c1 h2 w2 c2 -> (h1 w1 c1) (h2 w2 c2)')
  log_det_true = jnp.linalg.slogdet(G)[1]

  assert jnp.allclose(log_det, log_det_true)
  assert jnp.allclose(x[0], x_reconstr)


  import pdb; pdb.set_trace()



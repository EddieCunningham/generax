import jax.numpy as jnp
from jax import jit, random
from functools import partial, reduce
import numpy as np
import jax
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable, Iterator
import jax.lax as lax
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray, PyTree
import einops

def tree_shapes(pytree):
  return jax.tree_util.tree_map(lambda x: x.shape, pytree)

def square_plus(x, gamma=0.5):
  # https://arxiv.org/pdf/1901.08431.pdf
  out = 0.5*(x + jnp.sqrt(x**2 + 4*gamma))
  out = jnp.maximum(out, 0.0)
  return out

def square_sigmoid(x, gamma=0.5):
  # Derivative of proximal relu.  Basically sigmoid without saturated gradients.
  return 0.5*(1 + x*jax.lax.rsqrt(x**2 + 4*gamma))

def square_swish(x, gamma=0.5):
  x2 = x**2
  out = 0.5*(x + x2*jax.lax.rsqrt(x2 + 4*gamma))
  return out

def mean_and_std(x, axis=-1, keepdims=False):
  mean = jnp.mean(x, axis=axis, keepdims=keepdims)
  std = jnp.std(x, axis=axis, keepdims=keepdims)
  return mean, std

def mean_and_inverse_std(x, axis=-1, keepdims=False):
  mean = jnp.mean(x, axis=axis, keepdims=keepdims)
  mean_sq = jnp.mean(lax.square(x), axis=axis, keepdims=keepdims)
  var = mean_sq - lax.square(mean)
  inv_std = lax.rsqrt(var + 1e-6)
  return mean, inv_std

def list_prod(x):
  # We might run into JAX tracer issues if we do something like multiply the elements of a shape tuple with jnp
  return np.prod(x)

def whiten(x):
  U, s, VT = jnp.linalg.svd(x, full_matrices=False)
  return jnp.dot(U, VT)


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
    leaves = []
    for i, (shape, size) in enumerate(self.shapes_and_sizes):

      # Extract each leaf from the flattened parameters and reshape it
      buffer = flat_params[self.indices[i]: self.indices[i + 1]]
      leaf = buffer.reshape(shape)
      leaves.append(leaf)

    # Turn the leaves back into a tree
    params = jax.tree_util.tree_unflatten(self.treedef, leaves)

    return eqx.combine(params, self.static)

def extract_multiple_batches_from_iterator(it: Iterator,
                                           n_batches: int,
                                           single_batch=False):
  data = [None for _ in range(n_batches)]
  for i in range(n_batches):
    data[i] = next(it)
  out = jax.tree_util.tree_map(lambda *xs: jnp.array(xs), *data)
  if single_batch:
    out = jax.tree_util.tree_map(lambda x: einops.rearrange(x, 'n b ... -> (n b) ...'), out)
  return out

class ZeroInit(eqx.Module):

  w: Array

  def __init__(self,
               *_,
               x: Array,
               y: Optional[Array] = None,
               key: PRNGKeyArray,
               **kwargs):
    self.w = random.normal(key, (1,))*0.01

  def __call__(self, x: Array, **kwargs) -> Array:
    return x*self.w



import pickle
from pathlib import Path
from typing import Union

def ensure_path_exists(path):
  Path(path).mkdir(parents=True, exist_ok=True)

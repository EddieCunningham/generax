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

__all__ = ['broadcast_to_first_axis',
           'last_axes',
           'get_reduce_axes',
           'index_list',
           'tree_shapes',
           'square_plus',
           'square_sigmoid',
           'square_swish',
           'only_gradient',
           'mean_and_std',
           'mean_and_inverse_std',
           'list_prod',
           'whiten',
           'extract_multiple_batches_from_iterator',
           'ensure_path_exists',
           'conv',
           'unbatch']

################################################################################################################

def unbatch(pytree):
  return jax.tree_util.tree_map(lambda x: x[0], pytree)

def broadcast_to_first_axis(x, ndim):
  if x.ndim == 0:
    return x
  return jnp.expand_dims(x, axis=tuple(range(1, ndim)))

def last_axes(shape):
  return tuple(range(-1, -1 - len(shape), -1))

def get_reduce_axes(axes, ndim, offset=0):
  if isinstance(axes, int):
    axes = (axes,)
  keep_axes = [ax%ndim for ax in axes]
  reduce_axes = tuple([ax + offset for ax in range(ndim) if ax not in keep_axes])
  return reduce_axes

def index_list(shape, axis):
  ndim = len(shape)
  axis = [ax%ndim for ax in axis]
  shapes = [s for i, s in enumerate(shape) if i in axis]
  return tuple(shapes)

################################################################################################################

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

def only_gradient(x):
  return x - jax.lax.stop_gradient(x)

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


import pickle
from pathlib import Path
from typing import Union

def ensure_path_exists(path):
  Path(path).mkdir(parents=True, exist_ok=True)

def conv(w,
         x,
         stride=1,
         padding='SAME'):
  no_batch = False
  if x.ndim == 3:
    no_batch = True
    x = x[None]

  if isinstance(padding, int):
    padding = ((padding, padding), (padding, padding))

  out = jax.lax.conv_general_dilated(x,
                                     w,
                                     window_strides=(stride, stride),
                                     padding=padding,
                                     lhs_dilation=(1, 1),
                                     rhs_dilation=(1, 1),
                                     dimension_numbers=("NHWC", "HWIO", "NHWC"))
  if no_batch:
    out = out[0]
  return out


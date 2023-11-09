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
from generax.flows.base import BijectiveTransform
import numpy as np
import generax.util as util

__all__ = ['RationalQuadraticSpline',]

################################################################################################################

def forward_spline(x,
                   mask,
                   knot_x_k,
                   knot_y_k,
                   delta_k,
                   knot_x_kp1,
                   knot_y_kp1,
                   delta_kp1):

  delta_y = (knot_y_kp1 - knot_y_k)
  delta_x = (knot_x_kp1 - knot_x_k)
  delta_x = jnp.where(mask, delta_x, 1.0)
  s_k = delta_y/delta_x

  zeta = (x - knot_x_k)/delta_x
  onemz = (1 - zeta)
  z1mz = zeta*onemz

  # Return the output
  alpha = delta_y*(s_k*zeta**2 + delta_k*z1mz)
  beta = s_k + (delta_kp1 + delta_k - 2*s_k)*z1mz
  gamma = alpha/beta
  z = knot_y_k + gamma

  z = jnp.where(mask, z, x)

  dzdx = (s_k/beta)**2 * (delta_kp1*zeta**2 + 2*s_k*z1mz + delta_k*onemz**2)
  dzdx = jnp.where(mask, dzdx, 1.0)
  return z, dzdx

def inverse_spline(x,
                   mask,
                   knot_x_k,
                   knot_y_k,
                   delta_k,
                   knot_x_kp1,
                   knot_y_kp1,
                   delta_kp1):

  delta_y = (knot_y_kp1 - knot_y_k)
  delta_x = (knot_x_kp1 - knot_x_k)
  delta_x = jnp.where(mask, delta_x, 1.0)
  s_k = delta_y/delta_x

  knot_y_diff = x - knot_y_k
  term = knot_y_diff*(delta_kp1 + delta_k - 2*s_k)

  # Solve the quadratic
  b = delta_y*delta_k - term
  a = delta_y*s_k - b
  c = -s_k*knot_y_diff
  argument = b**2 - 4*a*c
  argument = jnp.where(mask, argument, 1.0) # Avoid nans
  d = -b - jnp.sqrt(argument)
  zeta = 2*c/d
  z1mz = zeta*(1 - zeta)

  # Solve for x
  z = zeta*delta_x + knot_x_k

  z = jnp.where(mask, z, x)

  beta = s_k + (delta_kp1 + delta_k - 2*s_k)*z1mz
  dzdx = (s_k/beta)**2 * (delta_kp1*zeta**2 + 2*s_k*z1mz + delta_k*(1 - zeta)**2)
  dzdx = jnp.where(mask, dzdx, 1.0)
  return z, dzdx

################################################################################################################

def find_knots(x, knot_x, knot_y, knot_derivs, inverse):

  # Need to find the knots for each dimension of x
  searchsorted = partial(jnp.searchsorted, side="right")
  take = jnp.take
  for i in range(len(x.shape)):
    searchsorted = jax.vmap(searchsorted)
    take = jax.vmap(take)

  if inverse == False:
    indices = searchsorted(knot_x, x) - 1
  else:
    indices = searchsorted(knot_y, x) - 1

  # Find the corresponding knots and derivatives
  knot_x_k = take(knot_x, indices)
  knot_y_k = take(knot_y, indices)
  delta_k = take(knot_derivs, indices)

  # We need the next indices too
  knot_x_kp1 = take(knot_x, indices + 1)
  knot_y_kp1 = take(knot_y, indices + 1)
  delta_kp1 = take(knot_derivs, indices + 1)
  args = knot_x_k, knot_y_k, delta_k, knot_x_kp1, knot_y_kp1, delta_kp1

  return args

################################################################################################################

def get_knot_params(settings, theta):
  K, min_width, min_height, min_derivative, bounds = settings

  # Get the individual parameters
  tw, th, td = theta[...,:K], theta[...,K:2*K], theta[...,2*K:]

  # Make the parameters fit the discription of knots
  tw, th = jax.nn.softmax(tw, axis=-1), jax.nn.softmax(th, axis=-1)
  tw = min_width + (1.0 - min_width*K)*tw
  th = min_height + (1.0 - min_height*K)*th
  td = min_derivative + misc.square_plus(td)
  knot_x, knot_y = jnp.cumsum(tw, axis=-1), jnp.cumsum(th, axis=-1)

  # Pad the knots so that the first element is 0
  pad = [(0, 0)]*(len(td.shape) - 1) + [(1, 0)]
  knot_x = jnp.pad(knot_x, pad)
  knot_y = jnp.pad(knot_y, pad)

  # Scale by the bounds
  knot_x = (bounds[0][1] - bounds[0][0])*knot_x + bounds[0][0]
  knot_y = (bounds[1][1] - bounds[1][0])*knot_y + bounds[1][0]

  # Set the max and min values exactly
  knot_x = knot_x.at[...,0].set(bounds[0][0])
  knot_x = knot_x.at[...,-1].set(bounds[0][1])
  knot_y = knot_y.at[...,0].set(bounds[1][0])
  knot_y = knot_y.at[...,-1].set(bounds[1][1])

  # Pad the derivatives so that the first and last elts are 1
  pad = [(0, 0)]*(len(td.shape) - 1) + [(1, 1)]
  knot_derivs = jnp.pad(td, pad, constant_values=1)

  return knot_x, knot_y, knot_derivs

################################################################################################################

class RationalQuadraticSpline(BijectiveTransform):
  """Splines from https://arxiv.org/pdf/1906.04032.pdf.  This is the best overall choice to use in flows.


  **Attributes**:
  - `theta`: The parameters of the spline.
  """

  theta: Array

  K: int = eqx.field(static=True)
  min_width: float = eqx.field(static=True)
  min_height: float = eqx.field(static=True)
  min_derivative: float = eqx.field(static=True)
  bounds: Sequence[float] = eqx.field(static=True)

  def __init__(self,
               input_shape: Tuple[int],
               K: int = 8,
               min_width: Optional[float] = 1e-3,
               min_height: Optional[float] = 1e-3,
               min_derivative: Optional[float] = 1e-3,
               bounds: Sequence[float] = ((-10.0, 10.0), (-10.0, 10.0)),
               *,
               key: PRNGKeyArray,
               **kwargs):
    """**Arguments**:

    - `input_shape`: The input shape.  Output size is the same as shape.
    - `key`: A `jax.random.PRNGKey` for initialization
    - `K`: The number of knots to use.
    - `min_width`: The minimum width of the knots.
    - `min_height`: The minimum height of the knots.
    - `min_derivative`: The minimum derivative of the knots.
    - `bounds`: The bounds of the splines.
    """
    super().__init__(input_shape=input_shape,
                     **kwargs)
    self.K = K
    self.min_width = min_width
    self.min_height = min_height
    self.min_derivative = min_derivative
    self.bounds = bounds

    x_dim = util.list_prod(input_shape)
    self.theta = random.normal(key, shape=(x_dim*(3*self.K - 1),))*0.1

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
    `(z, log_det)`
    """
    assert x.shape == self.input_shape, 'Only works on unbatched data'

    # Flatten x
    x = x.ravel()

    # Get the parameters
    settings = self.K, self.min_width, self.min_height, self.min_derivative, self.bounds
    theta = jnp.broadcast_to(self.theta, x.shape + self.theta.shape)
    knot_x, knot_y, knot_derivs = get_knot_params(settings, theta)

    # The relevant knot depends on if we are inverting or not
    if inverse == False:
      mask = (x > self.bounds[0][0] + 1e-5) & (x < self.bounds[0][1] - 1e-5)
      apply_fun = forward_spline
    else:
      mask = (x > self.bounds[1][0] + 1e-5) & (x < self.bounds[1][1] - 1e-5)
      apply_fun = inverse_spline

    args = find_knots(x, knot_x, knot_y, knot_derivs, inverse)

    z, dzdx = apply_fun(x, mask, *args)
    elementwise_log_det = jnp.log(dzdx)

    log_det = elementwise_log_det.sum()
    if inverse:
      log_det = -log_det

    # Unflatten the output
    z = z.reshape(self.input_shape)

    return z, log_det

################################################################################################################

if __name__ == '__main__':
  from debug import *
  import matplotlib.pyplot as plt
  from generax.flows.base import Sequential

  key = random.PRNGKey(0)
  x = random.normal(key, shape=(10, 2))

  layer = RationalQuadraticSpline(x=x, key=key)

  z, log_det = layer(x[0])
  x_reconstr, log_det2 = layer(z, inverse=True)

  G = jax.jacobian(lambda x: layer(x)[0])(x[0])
  log_det_true = jnp.linalg.slogdet(G)[1]

  assert jnp.allclose(log_det, log_det_true)
  assert jnp.allclose(x[0], x_reconstr)

  z, log_det = eqx.filter_vmap(layer)(x)

  import pdb
  pdb.set_trace()

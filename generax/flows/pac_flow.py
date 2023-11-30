from jax.config import config
# config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
import generax.util as util
from jax import random, vmap
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
from jax.flatten_util import ravel_pytree
import einops
from generax.flows.base import *
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray
import numpy as np

__all__ = ['PACFlow',
           'EmergingConv']

################################################################################################################

def im2col_fun(x,
               filter_shape: Sequence[int],
               stride: Sequence[int],
               padding: Union[str, Sequence[Tuple[int, int]]],
               lhs_dilation: Sequence[int],
               rhs_dilation: Sequence[int],
               dimension_numbers: Sequence[str]):
  # Turn (H,W,C_in) into (H,W,C_in*Kx*Ky) so that we can just do
  # a matrix multiply with filter.reshape((C_in*Kx*Ky,C_out)) to
  # get the convolution result of shape (H,W,C_out).
  three_dim = False
  if x.ndim == 3:
    x = x[None]
    three_dim = True
  out = jax.lax.conv_general_dilated_patches(x,
                                             filter_shape=filter_shape,
                                             window_strides=stride,
                                             padding=padding,
                                             lhs_dilation=lhs_dilation,
                                             rhs_dilation=rhs_dilation,
                                             dimension_numbers=dimension_numbers)
  if three_dim:
    out = out[0]
  return out

################################################################################################################

def im2col_conv(x_i2c, k_i2c, mask, w):
  x_i2c = x_i2c*mask
  if k_i2c is not None:
    return jnp.einsum("...hwiuv,...hwouv,uvio->...hwo", x_i2c, k_i2c, w, optimize=True)
  return jnp.einsum("...hwiuv,uvio->...hwo", x_i2c, w, optimize=True)

def pac_features_se(im2col, f, s, t):
  f_i2c = im2col(f)
  f_diff = f_i2c - f[...,None,None]
  summed = jnp.sum(f_diff**2, axis=-3)
  k_i2c = jnp.exp(-0.5*s[...,None,None,:]*summed[...,None])
  k_i2c *= t[...,None,None,:]
  k_i2c = einops.rearrange(k_i2c, "... h w u v c -> ... h w c u v")
  return k_i2c

def pac_base(im2col, theta, w):
  Kx, _, _, C = w.shape
  pad = Kx//2

  f, s, t = theta[...,:-2*C], theta[...,-2*C:-C], theta[...,-C:]
  f = util.square_sigmoid(f, gamma=1.0)*2 - 1.0
  t = util.square_sigmoid(t, gamma=1.0)*1.0
  s = util.square_sigmoid(s, gamma=1.0)*1.0

  # Generate the convolutional kernel from the features
  k_i2c = pac_features_se(im2col, f, s, t)

  # Get the diagonal of the transformation
  diag_jacobian = k_i2c[...,pad,pad]*w[pad,pad,jnp.arange(C),jnp.arange(C)]
  return k_i2c, diag_jacobian

def pac_ldu_mvp(x, theta, w, order, inverse=False, **im2col_kwargs):
  H, W, C = x.shape[-3:]
  Kx, Ky = w.shape[:2]
  assert Kx == Ky
  pad = Kx//2

  def im2col(x):
    return im2col_fun(x, **im2col_kwargs).reshape(x.shape + (Kx, Ky))

  # Convert the imageto the patches view
  x_i2c = im2col(x)
  if theta is not None:
    k_i2c, diag_jacobian = pac_base(im2col, theta, w)
  else:
    diag_jacobian = w[pad,pad,jnp.arange(C),jnp.arange(C)]
    diag_jacobian = jnp.broadcast_to(diag_jacobian, x.shape)
    k_i2c = None

  # For autoregressive convs
  order_i2c = im2col(order)
  upper_mask = order[...,None,None] >= order_i2c
  lower_mask = ~upper_mask

  # Compute the output using the LDU decomposition
  if inverse == False:
    z = im2col_conv(x_i2c, k_i2c, upper_mask, jnp.triu(w))
    z_i2c = im2col(z)
    z = im2col_conv(z_i2c, k_i2c, lower_mask, w) + z
  else:
    def mvp(mask, x):
      x_i2c = im2col(x)
      return im2col_conv(x_i2c, k_i2c, mask, w) + x
    z = util.weighted_jacobi(partial(mvp, lower_mask), x)

    def mvp(mask, x):
      x_i2c = im2col(x)
      return im2col_conv(x_i2c, k_i2c, mask, jnp.triu(w))
    z = util.weighted_jacobi(partial(mvp, upper_mask), z, diagonal=diag_jacobian)

  return z, diag_jacobian

################################################################################################################

class PACFlow(BijectiveTransform):
  """Pixel adaptive convolutions.  Gets too numerically unstable to use in practice...
  https://eddiecunningham.github.io/pdfs/PAC_Flow.pdf
  """
  kernel_shape: Tuple[int] = eqx.field(static=True)
  feature_dim: int = eqx.field(static=True)
  order_type: str = eqx.field(static=True)
  pixel_adaptive: bool = eqx.field(static=True)
  im2col_kwargs: Any = eqx.field(static=True)

  order: Array = eqx.field(static=True)

  w: Array
  theta: Union[Array, None]

  def __init__(self,
               input_shape: Tuple[int],
               feature_dim: int = 8,
               kernel_size: int=5,
               order_type: str="s_curve",
               pixel_adaptive: bool=True,
               zero_init: bool = True,
               *,
               key: PRNGKeyArray,
               **kwargs):
    """
    **Arguments**:

    - `input_shape`: The input shape.  Output size is the same as shape.
    - `feature_dim`: The dimension of the features
    - `kernel_size`: Height and width for the convolutional filter, (Kx, Ky).  The full
                     kernel will have shape (Kx, Ky, C, C)
    - `order_type`: The order to convolve in.  Either "raster" or "s_curve"
    - `pixel_adaptive`: Whether to use pixel adaptive convolutions
    - `zero_init`: Whether to initialize the weights to zero
    """
    super().__init__(input_shape=input_shape,
                     **kwargs)

    assert kernel_size%2 == 1
    self.kernel_shape   = (kernel_size, kernel_size)
    self.feature_dim    = feature_dim
    self.order_type     = order_type
    self.pixel_adaptive = pixel_adaptive

    H, W, C = input_shape

    # Extract the im2col kwargs
    Kx, Ky = self.kernel_shape
    pad = Kx//2
    self.im2col_kwargs = dict(filter_shape=self.kernel_shape,
                              stride=(1, 1),
                              padding=((pad, Kx - pad - 1),
                                       (pad, Ky - pad - 1)),
                              lhs_dilation=(1, 1),
                              rhs_dilation=(1, 1),
                              dimension_numbers=("NHWC", "HWIO", "NHWC"))

    # Determine the order to convolve
    order_shape = H, W, 1

    if self.order_type == "raster":
      order = np.arange(1, 1 + util.list_prod(order_shape)).reshape(order_shape)
    elif self.order_type == "s_curve":
      order = np.arange(1, 1 + util.list_prod(order_shape)).reshape(order_shape)
      order[::2, :, :] = order[::2, :, :][:, ::-1]
    order = order*1.0 # Turn into a float

    self.order = order

    # Initialize the weights
    k1, k2 = random.split(key, 2)
    w = random.normal(k1, shape=self.kernel_shape + (C, C))
    if zero_init:
      pad = Kx//2
      w = w.at[pad,pad,jnp.arange(C),jnp.arange(C)].set(1.0)
    self.w = w

    if self.pixel_adaptive == True:
      self.theta = random.normal(k2, shape=(H, W, 2*C + self.feature_dim))
    else:
      self.theta = None

  def __call__(self,
               x: Array,
               y: Optional[Array] = None,
               inverse: bool = False,
               **kwargs) -> Array:
    """
    **Arguments**:

    - `x`: The input to the transformation
    - `y`: The conditioning information
    - `inverse`: Whether to inverse the transformation

    **Returns**:
    `(z, log_det)`
    """
    assert x.shape == self.input_shape, 'Only works on unbatched data'
    H, W, C = x.shape

    # Apply the linear function
    z, diag_jacobian = pac_ldu_mvp(x,
                                   self.theta,
                                   self.w,
                                   self.order,
                                   inverse=inverse,
                                   **self.im2col_kwargs)

    # Get the log det
    if self.theta is not None:
      flat_diag = diag_jacobian.reshape(self.theta.shape[:-3] + (-1,))
    else:
      flat_diag = diag_jacobian.reshape(x.shape[:1] + (-1,))

    log_det = jnp.log(jnp.abs(flat_diag)).sum()

    if inverse:
      log_det = -log_det

    assert z.shape == self.input_shape
    assert log_det.shape == ()

    return z, log_det

class EmergingConv(PACFlow):
  """Emerging convolutions https://arxiv.org/pdf/1901.11137.pdf
  This is a special case of PAC flows
  """

  def __init__(self,
               input_shape: Tuple[int],
               kernel_size: int = 5,
               order_type: str = "s_curve",
               zero_init: bool = True,
               *,
               key: PRNGKeyArray,
               **kwargs):
    super().__init__(input_shape=input_shape,
                     feature_dim=None,
                     kernel_size=kernel_size,
                     order_type=order_type,
                     pixel_adaptive=False,
                     zero_init=zero_init,
                     key=key,
                     **kwargs)

################################################################################################################

if __name__ == "__main__":

  from debug import *
  import matplotlib.pyplot as plt
  jax.config.update("jax_enable_x64", True)

  key = random.PRNGKey(0)
  x, y = random.normal(key, shape=(2, 10, 10, 10, 2))

  layer = PACFlow(input_shape=x.shape[1:],
                  feature_dim=8,
                  kernel_size=5,
                  order_type="s_curve",
                  pixel_adaptive=True,
                  zero_init=False,
                  key=key)

  z, log_det = layer(x[0])
  x_reconstr, log_det2 = layer(z, inverse=True)

  G = jax.jacobian(lambda x: layer(x)[0])(x[0])
  G = einops.rearrange(G, 'h1 w1 c1 h2 w2 c2 -> (h1 w1 c1) (h2 w2 c2)')
  log_det_true = jnp.linalg.slogdet(G)[1]

  assert jnp.allclose(log_det, log_det_true)
  assert jnp.allclose(x[0], x_reconstr)

  import pdb
  pdb.set_trace()

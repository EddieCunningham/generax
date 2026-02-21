import jax
import jax.numpy as jnp
from jax import random
from typing import Optional, Tuple, Union, Sequence, Any
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray
import einops
import generax.util as util

__all__ = ['PAC']

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

class PAC(eqx.Module):
  """Pixel adaptive convolution.
  https://arxiv.org/pdf/1904.05373.pdf
  """
  in_channels: int = eqx.field(static=True)
  out_channels: int = eqx.field(static=True)
  feature_dim: int = eqx.field(static=True)
  kernel_size: int = eqx.field(static=True)

  W: Array
  log_s: Array
  log_t: Array

  def __init__(self,
               in_channels: int,
               out_channels: int,
               feature_dim: int,
               kernel_size: int = 5,
               *,
               key: PRNGKeyArray,
               **kwargs):
    """**Arguments**:

    - `in_channels`: The number of input channels.
    - `out_channels`: The number of output channels.
    - `feature_dim`: The dimension of the guidance features.
    - `kernel_size`: The size of the convolutional kernel.
    - `key`: A `jax.random.PRNGKey` for initialization.
    """
    super().__init__(**kwargs)
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.feature_dim = feature_dim
    self.kernel_size = kernel_size

    k1, k2, k3 = random.split(key, 3)

    # Initialize weights
    w_init = jax.nn.initializers.he_uniform(in_axis=-2, out_axis=-1)
    self.W = w_init(k1, shape=(kernel_size, kernel_size, in_channels, out_channels))

    # log_s and log_t as parameters, initialized to 0 (s=1, t=1)
    self.log_s = jnp.zeros((out_channels,))
    self.log_t = jnp.zeros((out_channels,))

  @property
  def im2col_kwargs(self) -> dict:
    pad = self.kernel_size // 2
    return dict(
                filter_shape=(self.kernel_size, self.kernel_size),
                stride=(1, 1),
                padding=((pad, self.kernel_size - pad - 1),
                         (pad, self.kernel_size - pad - 1)),
                lhs_dilation=(1, 1),
                rhs_dilation=(1, 1),
                dimension_numbers=("NHWC", "HWIO", "NHWC"))

  def __call__(self,
               x: Array,
               f: Array,
               **kwargs) -> Array:
    """**Arguments**:

    - `x`: Input image of shape `(H, W, in_channels)`.
    - `f`: Guidance features of shape `(H, W, feature_dim)`.

    **Returns**:
    Output image of shape `(H, W, out_channels)`.
    """
    H, W, C_in = x.shape
    K = self.kernel_size

    def im2col(img):
      return im2col_fun(img, **self.im2col_kwargs).reshape(img.shape[:-1] + (-1, K, K))

    # Get patches for x and f
    x_i2c = im2col(x) # (H, W, C_in, K, K)
    f_i2c = im2col(f) # (H, W, feature_dim, K, K)

    # Compute Gaussian kernel
    # f_diff = f_i2c - f(i,j)
    f_diff = f_i2c - f[..., None, None] # (H, W, feature_dim, K, K)
    summed = jnp.sum(f_diff**2, axis=-3) # (H, W, K, K)

    s = jnp.exp(self.log_s)
    t = jnp.exp(self.log_t)

    # K(i,j) = t*exp(-0.5*s*||f_i - f_j||^2)
    # s and t are (C_out,)
    # summed is (H, W, K, K)
    # We want k_i2c to be (H, W, C_out, K, K)
    k_i2c = jnp.exp(-0.5*s[None, None, :, None, None]*summed[..., None, :, :])
    k_i2c = t[None, None, :, None, None]*k_i2c

    # Apply adaptive convolution
    # x_i2c: (H, W, C_in, K, K)
    # k_i2c: (H, W, C_out, K, K)
    # W: (K, K, C_in, C_out)
    out = jnp.einsum("hwiuv,hwouv,uvio->hwo", x_i2c, k_i2c, self.W, optimize=True)

    return out

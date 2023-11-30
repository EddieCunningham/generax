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
from generax.flows.base import *
import generax.util as util

__all__ = ['CircularConv',
           'CaleyOrthogonalConv',
           'OneByOneConv']

fft_channel_vmap = jax.vmap(jnp.fft.fftn, in_axes=(2,), out_axes=2)
ifft_channel_vmap = jax.vmap(jnp.fft.ifftn, in_axes=(2,), out_axes=2)
fft_double_channel_vmap = jax.vmap(fft_channel_vmap, in_axes=(2,), out_axes=2)

inv_height_vmap = jax.vmap(jnp.linalg.inv)
inv_height_width_vmap = jax.vmap(inv_height_vmap)

def complex_slogdet(x):
    D = jnp.block([[x.real, -x.imag], [x.imag, x.real]])
    return 0.25*jnp.linalg.slogdet(D@D.T)[1]
slogdet_height_width_vmap = jax.vmap(jax.vmap(complex_slogdet))


class CircularConv(BijectiveTransform):
  """Circular convolution.  Equivalent to a regular convolution with circular padding.
        https://papers.nips.cc/paper/2019/file/b1f62fa99de9f27a048344d55c5ef7a6-Paper.pdf
  """
  filter_shape: Tuple[int] = eqx.field(static=True)
  w: Array

  def __init__(self,
               input_shape: Tuple[int],
               filter_shape: Tuple[int]=(3, 3),
               *,
               key: PRNGKeyArray,
               **kwargs):
    """**Arguments**:

    - `input_shape`: The input shape.  Output size is the same as shape.
    - `key`: A `jax.random.PRNGKey` for initialization
    - `filter_shape`: Height and width for the convolutional filter, (Kx, Ky).  The full
                      kernel will have shape (Kx, Ky, C, C)
    """
    assert len(filter_shape) == 2
    self.filter_shape = filter_shape
    super().__init__(input_shape=input_shape,
                     **kwargs)

    H, W, C = input_shape

    w = random.normal(key, shape=self.filter_shape + (C, C))
    self.w = jax.vmap(jax.vmap(util.whiten))(w)

  def __call__(self,
               x: Array,
               y: Optional[Array] = None,
               inverse: bool = False,
               **kwargs) -> Array:
    """
    See http://developer.download.nvidia.com/compute/cuda/2_2/sdk/website/projects/convolutionFFT2D/doc/convolutionFFT2D.pdf
    **Arguments**:

    - `x`: The input to the transformation
    - `y`: The conditioning information
    - `inverse`: Whether to inverse the transformation

    **Returns**:
    `(z, log_det)`
    """
    assert x.shape == self.input_shape, 'Only works on unbatched data'
    H, W, C = x.shape

    Kx, Ky, _, _ = self.w.shape

    # See how much we need to roll the filter
    W_x = (Kx - 1) // 2
    W_y = (Ky - 1) // 2

    # Pad the filter to match the fft size and roll it so that its center is at (0,0)
    W_padded = jnp.pad(self.w[::-1,::-1,:,:], ((0, H - Kx), (0, W - Ky), (0, 0), (0, 0)))
    W_padded = jnp.roll(W_padded, (-W_x, -W_y), axis=(0, 1))

    # Apply the FFT to get the convolution
    if inverse == False:
      image_fft = fft_channel_vmap(x)
    else:
      image_fft = fft_channel_vmap(x)
    W_fft = fft_double_channel_vmap(W_padded)

    if inverse == True:
      z_fft = jnp.einsum("abij,abj->abi", W_fft, image_fft)
      z = ifft_channel_vmap(z_fft).real
    else:
      # For deconv, we need to invert the W over the channel dims
      W_fft_inv = inv_height_width_vmap(W_fft)

      x_fft = jnp.einsum("abij,abj->abi", W_fft_inv, image_fft)
      z = ifft_channel_vmap(x_fft).real

    # The log determinant is the log det of the frequencies over the channel dims
    log_det = -slogdet_height_width_vmap(W_fft).sum()

    if inverse:
      log_det = -log_det

    return z, log_det

################################################################################################################

class CaleyOrthogonalConv(BijectiveTransform):
  """Caley parametrization of an orthogonal convolution.
        https://arxiv.org/pdf/2104.07167.pdf
  """
  filter_shape: Tuple[int] = eqx.field(static=True)
  v: Array
  g: Array

  def __init__(self,
               input_shape: Tuple[int],
               filter_shape: Tuple[int] = (3, 3),
               *,
               key: PRNGKeyArray,
               **kwargs):
    """**Arguments**:

    - `input_shape`: The input shape.  Output size is the same as shape.
    - `key`: A `jax.random.PRNGKey` for initialization
    - `filter_shape`: Height and width for the convolutional filter, (Kx, Ky).  The full
                      kernel will have shape (Kx, Ky, C, C)
    """
    assert len(filter_shape) == 2
    self.filter_shape = filter_shape
    super().__init__(input_shape=input_shape,
                     **kwargs)

    H, W, C = input_shape

    k1, k2 = random.split(key, 2)
    self.v = random.normal(k1, shape=self.filter_shape + (C, C))
    self.g = random.normal(k2, shape=())

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

    w = self.g*self.v/jnp.linalg.norm(self.v)

    Kx, Ky, _, _ = w.shape

    # See how much we need to roll the filter
    W_x = (Kx - 1) // 2
    W_y = (Ky - 1) // 2

    # Pad the filter to match the fft size and roll it so that its center is at (0,0)
    W_padded = jnp.pad(w[::-1,::-1,:,:], ((0, H - Kx), (0, W - Ky), (0, 0), (0, 0)))
    W_padded = jnp.roll(W_padded, (-W_x, -W_y), axis=(0, 1))

    # Apply the FFT to get the convolution
    if inverse == False:
      image_fft = fft_channel_vmap(x)
    else:
      image_fft = fft_channel_vmap(x)
    W_fft = fft_double_channel_vmap(W_padded)

    A_fft = W_fft - W_fft.conj().transpose((0, 1, 3, 2))
    I = jnp.eye(W_fft.shape[-1])

    if inverse == True:
      IpA_inv = inv_height_width_vmap(I[None,None] + A_fft)
      y_fft = jnp.einsum("abij,abj->abi", IpA_inv, image_fft)
      z_fft = y_fft - jnp.einsum("abij,abj->abi", A_fft, y_fft)
      z = ifft_channel_vmap(z_fft).real
    else:
      ImA_inv = inv_height_width_vmap(I[None,None] - A_fft)
      y_fft = jnp.einsum("abij,abj->abi", ImA_inv, image_fft)
      z_fft = y_fft + jnp.einsum("abij,abj->abi", A_fft, y_fft)
      z = ifft_channel_vmap(z_fft).real

    log_det = jnp.zeros(x.shape[:1])
    return z, log_det

################################################################################################################

class OneByOneConv(BijectiveTransform):
  """ 1x1 convolution.  Uses a dense parametrization because the channel dimension will probably
      never be that big.  Costs O(C^3).  Used in GLOW https://arxiv.org/pdf/1807.03039.pdf
  """
  w: Array

  def __init__(self,
               input_shape: Tuple[int],
               *,
               key: PRNGKeyArray,
               **kwargs):
    """**Arguments**:

    - `input_shape`: The input shape.  Output size is the same as shape.
    - `key`: A `jax.random.PRNGKey` for initialization
    """
    super().__init__(input_shape=input_shape,
                     **kwargs)

    H, W, C = input_shape
    w = random.normal(key, shape=(C, C))
    self.w = util.whiten(w)

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

    # Using lax.conv instead of matrix multiplication over the channel dimension
    # is faster and also more numerically stable for some reason.

    # Run the flow
    if inverse == False:
      z = util.conv(self.w[None,None,:,:], x)
    else:
      w_inv = jnp.linalg.inv(self.w)
      z = util.conv(w_inv[None,None,:,:], x)

    log_det = jnp.linalg.slogdet(self.w)[1]*H*W
    log_det *= jnp.ones(x.shape[:1])

    if inverse:
      log_det = -log_det

    return z, log_det

################################################################################################################

if __name__ == '__main__':
  from debug import *
  import matplotlib.pyplot as plt
  from generax.flows.base import Sequential
  # switch to x64
  jax.config.update("jax_enable_x64", True)

  key = random.PRNGKey(0)
  x, y = random.normal(key, shape=(2, 10, 5, 5, 2))

  layer = OneByOneConv(input_shape=x.shape[1:],
                             key=key)

  z, log_det = layer(x[0])
  x_reconstr, log_det2 = layer(z, inverse=True)

  G = jax.jacobian(lambda x: layer(x)[0])(x[0])
  G = einops.rearrange(G, 'h1 w1 c1 h2 w2 c2 -> (h1 w1 c1) (h2 w2 c2)')
  log_det_true = jnp.linalg.slogdet(G)[1]

  assert jnp.allclose(log_det, log_det_true)
  assert jnp.allclose(x[0], x_reconstr)


  import pdb; pdb.set_trace()



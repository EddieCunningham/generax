from collections.abc import Callable
from typing import Literal, Optional, Union, Tuple
import jax
import jax.random as random
from jaxtyping import Array, PRNGKeyArray
import equinox as eqx
import jax.numpy as jnp
from generax.nn.layers import *
from generax.nn.resnet_blocks import *
from generax.nn.time_condition import *
import generax.util as util

def freeu_filter(x, threshold, scale):
  """https://arxiv.org/pdf/2309.11497.pdf"""

  H, W, C = x.shape
  # FFT
  x_freq = jnp.fft.fftn(x, dim=(0, 1))
  x_freq = jnp.fft.fftshift(x_freq, dim=(0, 1))

  mask = jnp.ones_like(x_freq)

  crow, ccol = H // 2, W //2
  t = threshold
  mask = mask.at[crow-t:crow+t,ccol-t:ccol+t,:].set(scale)
  x_freq = x_freq * mask

  # IFFT
  x_freq = jnp.fft.ifftshift(x_freq, dim=(0, 1))
  x_filtered = jnp.fft.ifftn(x_freq, dim=(0, 1)).real

  return x_filtered

class UNet(eqx.Module):
  """Unet architecture.
  """

  input_shape: Tuple[int] = eqx.field(static=True)
  dim: int = eqx.field(static=True)
  dim_mults: Tuple[int] = eqx.field(static=True)
  in_out: Tuple[Tuple[int, int]] = eqx.field(static=True)
  conv_in: WeightNormConv
  time_features: TimeFeatures

  down_blocks: Tuple[Union[ImageResBlock, AttentionBlock, Downsample]]
  middle_blocks: Tuple[Union[ImageResBlock, AttentionBlock]]
  up_blocks: Tuple[Union[ImageResBlock, AttentionBlock, Upsample]]
  final_block: ImageResBlock
  proj_out: WeightNormConv

  freeu: bool = eqx.field(static=True)
  time_dependent: bool = eqx.field(static=True)
  cond_shape: Optional[Tuple[int]] = eqx.field(static=True)

  def __init__(self,
               input_shape: Tuple[int],
               dim: int = 16,
               dim_mults: Tuple[int] = (1, 2, 4, 8),
               resnet_block_groups: int = 8,
               attn_heads: int = 4,
               attn_dim_head: int = 32,
               cond_shape: Optional[Tuple[int]] = None,
               *,
               key: PRNGKeyArray,
               freeu: bool = False,
               time_dependent: bool = True):
    """**Arguments**:

    - `input_shape`: The input shape.  Output size is the same as shape.
    - `dim`: The dimension of the features
    - `dim_mults`: The dimension of the features at each downsampling
    - `resnet_block_groups`: The number of resnet blocks per downsampling
    - `attn_heads`: The number of attention heads per downsampling
    - `attn_dim_head`: The dimension of the attention heads
    - `freeu`: Whether to use freeu filtering
    - `time_dependent`: Whether to use time conditioning
    """

    H, W, C = input_shape
    if H//(2**len(dim_mults)) == 0:
      raise ValueError(f"Image size {(H, W)} is too small for {len(dim_mults)} downsamples.")
    self.input_shape = input_shape
    self.dim = dim
    self.dim_mults = dim_mults
    self.freeu = freeu
    self.time_dependent = time_dependent

    keys = random.split(key, 20)
    key_iter = iter(keys)

    self.conv_in = WeightNormConv(input_shape=input_shape,
                                  out_size=self.dim,
                                  filter_shape=(7, 7),
                                  padding=3,
                                  key=next(key_iter))

    if self.time_dependent:
      self.time_features = TimeFeatures(embedding_size=self.dim,
                                        out_features=4*self.dim,
                                        key=next(key_iter))
      time_shape = (4*self.dim,)
    else:
      self.time_features = None
      time_shape = None

    self.cond_shape = cond_shape
    if cond_shape is not None:
      assert len(cond_shape) == 1
      if time_shape:
        time_shape = (time_shape[0] + cond_shape[0],)
      else:
        time_shape = cond_shape

    def make_resblock(key, input_shape, dim_out):
      return ImageResBlock(input_shape=input_shape,
                           hidden_size=dim_out,
                           out_size=dim_out,
                           groups=resnet_block_groups,
                           cond_shape=time_shape,
                           key=key)

    def make_attention(key, input_shape, linear=True):
      return AttentionBlock(input_shape=input_shape,
                            heads=attn_heads,
                            dim_head=attn_dim_head,
                            key=key,
                            use_linear_attention=linear)

    # Downsampling
    down_blocks = []
    dims = [self.dim*mult for mult in self.dim_mults]
    self.in_out = list(zip(dims[:-1], dims[1:]))
    keys = random.split(next(key_iter), len(self.in_out))
    for i, (key, (dim_in, dim_out)) in enumerate(zip(keys, self.in_out)):
      k1, k2 = random.split(key, 2)
      down_blocks.append(make_resblock(k1, (H, W, dim_in), dim_in))
      down_blocks.append(make_resblock(k2, (H, W, dim_in), dim_in))
      down_blocks.append(make_attention(key, (H, W, dim_in)))

      down = Downsample(input_shape=(H, W, dim_in),
                        out_size=dim_out,
                        key=key)
      down_blocks.append(down)
      assert H%2 == 0
      assert W%2 == 0
      H, W = H//2, W//2
    self.down_blocks = down_blocks

    # Middle
    middle_blocks = []
    middle_blocks.append(make_resblock(next(key_iter), (H, W, dim_out), dim_out))
    middle_blocks.append(make_attention(next(key_iter), (H, W, dim_out), linear=False))
    middle_blocks.append(make_resblock(next(key_iter), (H, W, dim_out), dim_out))
    self.middle_blocks = middle_blocks

    # Upsampling
    keys = random.split(next(key_iter), len(self.in_out))
    up_blocks = []
    for i, (key, (dim_in, dim_out)) in enumerate(zip(keys, self.in_out[::-1])):
      k1, k2 = random.split(key, 2)

      up = Upsample(input_shape=(H, W, dim_out),
                    out_size=dim_in,
                    key=key)
      up_blocks.append(up)
      H, W = H*2, W*2

      # Skip connections contribute a dim_in
      up_blocks.append(make_resblock(k1, (H, W, dim_in + dim_in), dim_in))
      up_blocks.append(make_resblock(k2, (H, W, dim_in + dim_in), dim_in))
      up_blocks.append(make_attention(key, (H, W, dim_in)))

    self.up_blocks = up_blocks

    # Final
    self.final_block = make_resblock(next(key_iter), (H, W, dim_in + dim_in), dim_in)
    self.proj_out = WeightNormConv(input_shape=(H, W, dim_in),
                                    out_size=C,
                                    filter_shape=(1, 1),
                                    key=next(key_iter))

  def data_dependent_init(self,
                          x: Array,
                          y: Optional[Array] = None,
                          key: PRNGKeyArray = None) -> eqx.Module:
    """Initialize the parameters of the layer based on the data.

    **Arguments**:

    - `x`: The data to initialize the parameters with.
    - `y`: The conditioning information
    - `key`: A `jax.random.PRNGKey` for initialization

    **Returns**:
    A new layer with the parameters initialized.
    """
    return self

  def __call__(self, *args) -> Array:
    if self.time_dependent:
      if len(args) == 3:
        t, x, y = args
      else:
        t, x = args
        y = None
      assert t.shape == ()
    else:
      if len(args) == 2:
        x, y = args
      else:
        x = args[0]
        y = None

    assert x.shape == self.input_shape

    # Time embedding
    if self.time_dependent:
      conditional_embedding = self.time_features(t)
      if y is not None:
        conditional_embedding = jnp.concatenate([conditional_embedding, y], axis=-1)
    else:
      conditional_embedding = y

    hs = []

    # Initial convolution
    h = self.conv_in(x)
    hs.append(h)

    # Downsampling
    block_iter = iter(self.down_blocks)
    for i, (dim_in, dim_out) in enumerate(self.in_out):
      # Resnet block
      h = next(block_iter)(h, conditional_embedding)
      hs.append(h)

      # Resnet block + attention block
      h = next(block_iter)(h, conditional_embedding)
      h = next(block_iter)(h)
      hs.append(h)

      # Downsample
      h = next(block_iter)(h)

    # Middle
    res_block1, attn_block, res_block2 = self.middle_blocks
    h = res_block1(h)
    h = attn_block(h)
    h = res_block2(h)

    # Upsampling
    block_iter = iter(self.up_blocks)
    for i, (dim_in, dim_out) in enumerate(self.in_out[::-1]):

      # Upsample
      h = next(block_iter)(h)

      hs_ = hs.pop()
      if self.freeu:
        assert 0, 'Not tested yet'
        if i == 0:
          h_mean = h.mean(axis=-1)[:,:,None]
          h_max = h_mean.max()
          h_min = h_mean.max()
          h_mean = (h_mean - h_min[None, None])/(h_max - h_min)[None, None]

          b1 = 1.5
          s1 = 0.9
          h = h.at[:,:640].mul((b1 - 1 ) * h_mean + 1)
          hs_ = freeu_filter(hs_, threshold=1.0, scale=s1)

      # Resnet block
      h = jnp.concatenate([h, hs_], axis=-1)
      h = next(block_iter)(h, conditional_embedding)

      # Resnet block
      h = jnp.concatenate([h, hs.pop()], axis=-1)
      h = next(block_iter)(h, conditional_embedding)

      # Attention block
      h = next(block_iter)(h)

    # Final
    h_in = hs.pop()
    h = jnp.concatenate([h, h_in], axis=-1)
    h = self.final_block(h, conditional_embedding)

    h = self.proj_out(h)

    return h

################################################################################################################

class Encoder(eqx.Module):
  """Half of the Unet architecture to use as an encoder.  Input is an image and output is a vector
  """

  input_shape: Tuple[int] = eqx.field(static=True)
  dim: int = eqx.field(static=True)
  dim_mults: Tuple[int] = eqx.field(static=True)
  in_out: Tuple[Tuple[int, int]] = eqx.field(static=True)
  conv_in: WeightNormConv

  down_blocks: Tuple[Union[ImageResBlock, AttentionBlock, Downsample]]
  middle_blocks: Tuple[Union[ImageResBlock, AttentionBlock]]
  proj_out: WeightNormConv

  def __init__(self,
               input_shape: Tuple[int],
               dim: int = 16,
               dim_mults: Tuple[int] = (1, 2, 4, 8),
               resnet_block_groups: int = 8,
               attn_heads: int = 4,
               attn_dim_head: int = 32,
               out_size: int = 16,
               cond_shape: Optional[Tuple[int]] = None,
               *,
               key: PRNGKeyArray):
    """**Arguments**:

    - `input_shape`: The input shape.  Output size is the same as shape.
    - `dim`: The dimension of the features
    - `dim_mults`: The dimension of the features at each downsampling
    - `resnet_block_groups`: The number of resnet blocks per downsampling
    - `attn_heads`: The number of attention heads per downsampling
    - `attn_dim_head`: The dimension of the attention heads
    - `out_size`: The dimension of the output
    """

    H, W, C = input_shape
    if H//(2**len(dim_mults)) == 0:
      raise ValueError(
          f"Image size {(H, W)} is too small for {len(dim_mults)} downsamples.")
    self.input_shape = input_shape
    self.dim = dim
    self.dim_mults = dim_mults

    keys = random.split(key, 20)
    key_iter = iter(keys)

    self.conv_in = WeightNormConv(input_shape=input_shape,
                                  out_size=self.dim,
                                  filter_shape=(7, 7),
                                  padding=3,
                                  key=next(key_iter))

    def make_resblock(key, input_shape, dim_out):
      return ImageResBlock(input_shape=input_shape,
                           hidden_size=dim_out,
                           out_size=dim_out,
                           groups=resnet_block_groups,
                           key=key)

    def make_attention(key, input_shape, linear=True):
      return AttentionBlock(input_shape=input_shape,
                            heads=attn_heads,
                            dim_head=attn_dim_head,
                            key=key,
                            use_linear_attention=linear)

    # Downsampling
    down_blocks = []
    dims = [self.dim*mult for mult in self.dim_mults]
    self.in_out = list(zip(dims[:-1], dims[1:]))
    keys = random.split(next(key_iter), len(self.in_out))
    for i, (key, (dim_in, dim_out)) in enumerate(zip(keys, self.in_out)):
      k1, k2 = random.split(key, 2)
      down_blocks.append(make_resblock(k1, (H, W, dim_in), dim_in))
      down_blocks.append(make_resblock(k2, (H, W, dim_in), dim_in))
      down_blocks.append(make_attention(key, (H, W, dim_in)))

      down = Downsample(input_shape=(H, W, dim_in),
                        out_size=dim_out,
                        key=key)
      down_blocks.append(down)
      assert H % 2 == 0
      assert W % 2 == 0
      H, W = H//2, W//2
    self.down_blocks = down_blocks

    # Middle
    middle_blocks = []
    middle_blocks.append(make_resblock(next(key_iter), (H, W, dim_out), dim_out))
    middle_blocks.append(make_attention(next(key_iter), (H, W, dim_out), linear=False))
    middle_blocks.append(make_resblock(next(key_iter), (H, W, dim_out), dim_out))
    self.middle_blocks = middle_blocks

    self.proj_out = WeightNormConv(input_shape=(H, W, dim_out),
                                   out_size=out_size,
                                   filter_shape=(H, W),
                                   padding=0,
                                   key=next(key_iter))

  def data_dependent_init(self,
                          x: Array,
                          y: Optional[Array] = None,
                          key: PRNGKeyArray = None) -> eqx.Module:
    """Initialize the parameters of the layer based on the data.

    **Arguments**:

    - `x`: The data to initialize the parameters with.
    - `y`: The conditioning information
    - `key`: A `jax.random.PRNGKey` for initialization

    **Returns**:
    A new layer with the parameters initialized.
    """
    return self

  def __call__(self, x, y=None) -> Array:
    assert x.shape == self.input_shape

    conditional_embedding = y


    # Initial convolution
    h = self.conv_in(x)

    # Downsampling
    block_iter = iter(self.down_blocks)
    for i, (dim_in, dim_out) in enumerate(self.in_out):
      # Resnet block
      h = next(block_iter)(h, conditional_embedding)

      # Resnet block + attention block
      h = next(block_iter)(h, conditional_embedding)
      h = next(block_iter)(h)

      # Downsample
      h = next(block_iter)(h)

    # Middle
    res_block1, attn_block, res_block2 = self.middle_blocks
    h = res_block1(h)
    h = attn_block(h)
    h = res_block2(h)

    # Final
    h = self.proj_out(h)
    return h.ravel()

################################################################################################################

if __name__ == '__main__':
  from debug import *
  import matplotlib.pyplot as plt
  from generax.flows.base import Sequential

  key = random.PRNGKey(0)
  dtype = jnp.bfloat16
  x = random.normal(key, shape=(10, 16, 16, 3), dtype=dtype)
  y = random.normal(key, (x.shape[0], 10), dtype=dtype)
  cond_shape = y.shape[1:]
  y, cond_shape = None, None

  # layer = UNet(input_shape=x.shape[1:],
  #                           dim_mults=(1, 2, 4),
  #                           key=key,
  #                           time_dependent=True,
  #                           cond_shape=cond_shape)
  layer = Encoder(input_shape=x.shape[1:],
               dim_mults=(1, 2, 4),
               key=key,
               out_size=16)

  params, static = eqx.partition(layer, eqx.is_inexact_array)
  params = jax.tree_util.tree_map(lambda x: x.astype(dtype), params)
  layer = eqx.combine(params, static)


  t = random.uniform(key, shape=x.shape[:1], dtype=dtype)
  # layer(util.unbatch(x))
  layer(util.unbatch(t), util.unbatch(x))
  # layer(util.unbatch(t), util.unbatch(x), util.unbatch(y))

  # out = eqx.filter_vmap(layer)(x)
  out = eqx.filter_vmap(layer)(t, x)
  # out = eqx.filter_vmap(layer)(t, x, y)

  import pdb; pdb.set_trace()


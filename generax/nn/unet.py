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

class TimeDependentUNet(eqx.Module):

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

  def __init__(self,
               input_shape: Tuple[int],
               dim: int = 16,
               dim_mults: Tuple[int] = (1, 2, 4, 8),
               resnet_block_groups: int = 8,
               attn_heads: int = 4,
               attn_dim_head: int = 32,
               *,
               key: PRNGKeyArray,
               freeu: bool = False):

    H, W, C = input_shape
    if H//(2**len(dim_mults)) == 0:
      raise ValueError(f"Image size {(H, W)} is too small for {len(dim_mults)} downsamples.")
    self.input_shape = input_shape
    self.dim = dim
    self.dim_mults = dim_mults
    self.freeu = freeu

    keys = random.split(key, 20)
    key_iter = iter(keys)

    self.conv_in = WeightNormConv(input_shape=input_shape,
                                  out_size=self.dim,
                                  filter_shape=(7, 7),
                                  padding=3,
                                  key=next(key_iter))

    self.time_features = TimeFeatures(embedding_size=self.dim,
                                      out_features=4*self.dim,
                                      key=next(key_iter))
    time_shape = (4*self.dim,)

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

  def __call__(self,
               t: Array,
               x: Array,
               y: Array = None) -> Array:

    assert t.shape == ()
    assert x.shape == self.input_shape

    # Time embedding
    time_emb = self.time_features(t)

    hs = []

    # Initial convolution
    h = self.conv_in(x)
    hs.append(h)

    # Downsampling
    block_iter = iter(self.down_blocks)
    for i, (dim_in, dim_out) in enumerate(self.in_out):
      # Resnet block
      h = next(block_iter)(h, time_emb)
      hs.append(h)

      # Resnet block + attention block
      h = next(block_iter)(h, time_emb)
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
      h = next(block_iter)(h, time_emb)

      # Resnet block
      h = jnp.concatenate([h, hs.pop()], axis=-1)
      h = next(block_iter)(h, time_emb)

      # Attention block
      h = next(block_iter)(h)

    # Final
    h_in = hs.pop()
    h = jnp.concatenate([h, h_in], axis=-1)
    h = self.final_block(h, time_emb)
    h = self.proj_out(h)

    assert len(hs) == 0

    return h


################################################################################################################

if __name__ == '__main__':
  from debug import *
  import matplotlib.pyplot as plt
  from generax.flows.base import Sequential

  key = random.PRNGKey(0)
  dtype = jnp.bfloat16
  x, y = random.normal(key, shape=(2, 10, 16, 16, 3), dtype=dtype)
  cond_shape = y.shape[1:]
  y, cond_shape = None, None

  layer = TimeDependentUNet(input_shape=x.shape[1:],
                            dim_mults=(1, 2, 4),
                            key=key)

  params, static = eqx.partition(layer, eqx.is_inexact_array)
  params = jax.tree_util.tree_map(lambda x: x.astype(dtype), params)
  layer = eqx.combine(params, static)


  t = random.uniform(key, shape=x.shape[:1], dtype=dtype)
  layer(util.unbatch(t), util.unbatch(x))

  out = eqx.filter_vmap(layer)(t, x)

  import pdb; pdb.set_trace()


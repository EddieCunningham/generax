from collections.abc import Callable
from typing import Literal, Optional, Union, Tuple
import jax
import jax.random as random
from jaxtyping import Array, PRNGKeyArray
import equinox as eqx
import jax.numpy as jnp
import generax.util as util
import einops

__all__ = ['WeightNormDense',
           'WeightNormConv',
           'WeightStandardizedConv',
           'ChannelConvention',
           'ConvAndGroupNorm',
           'Upsample',
           'Downsample',
           'GatedGlobalContext',
           'Attention',
           'LinearAttention',
           'AttentionBlock']

class WeightNormDense(eqx.Module):
  """Weight normalization parametrized linear layer
  https://arxiv.org/pdf/1602.07868.pdf
  """

  in_size: int = eqx.field(static=True)
  out_size: int = eqx.field(static=True)
  W: Array
  b: Array
  g: Array

  def __init__(self,
               in_size: int,
               out_size: int,
               key: PRNGKeyArray,
               **kwargs):
    super().__init__(**kwargs)

    self.in_size = in_size
    self.out_size = out_size


    w_init = jax.nn.initializers.he_uniform(in_axis=-2, out_axis=-1)
    self.W = w_init(key, shape=(out_size, in_size))
    self.g = jnp.array(1.0)
    self.b = jnp.zeros(out_size)

  def data_dependent_init(self,
                          x: Array,
                          key: PRNGKeyArray = None) -> eqx.Module:
    """Initialize the parameters of the layer based on the data.

    **Arguments**:

    - `x`: The data to initialize the parameters with.
    - `key`: A `jax.random.PRNGKey` for initialization

    **Returns**:
    A new layer with the parameters initialized.
    """
    assert x.shape[-1] == self.in_size, 'Only works on batched data'

    # Initialize g and b.
    W = self.W*jax.lax.rsqrt((self.W**2).sum(axis=1))[:, None]
    x = jnp.einsum('ij,bj->bi', W, x)

    std = jnp.std(x.reshape((-1, x.shape[-1])), axis=0) + 1e-5
    g = 1/std

    x *= g

    mean = jnp.mean(x.reshape((-1, x.shape[-1])), axis=0)
    b = -mean

    # Turn the new parameters into a new module
    get_g = lambda tree: tree.g
    get_b = lambda tree: tree.b
    updated_layer = eqx.tree_at(get_g, self, g)
    updated_layer = eqx.tree_at(get_b, updated_layer, b)

    return updated_layer

  def __call__(self, x: Array, y: Array = None) -> Array:
    W = self.W*jax.lax.rsqrt((self.W**2).sum(axis=1))[:, None]
    x = self.g*(W@x) + self.b
    return x


class WeightNormConv(eqx.Module):
  """Weight normalization parametrized convolutional layer
  https://arxiv.org/pdf/1602.07868.pdf
  """

  input_shape: int = eqx.field(static=True)
  out_size: int = eqx.field(static=True)
  filter_shape: Tuple[int] = eqx.field(static=True)
  padding: Union[int,str] = eqx.field(static=True)
  stride: int = eqx.field(static=True)
  W: Array
  b: Array
  g: Array

  def __init__(self,
               input_shape: Tuple[int], # in_channels
               filter_shape: Tuple[int],
               out_size: int,
               *,
               key: PRNGKeyArray,
               padding: Union[int,str] = 'SAME',
               stride: int = 1,
               **kwargs):
    super().__init__(**kwargs)
    H, W, C = input_shape

    self.input_shape = input_shape
    self.filter_shape = filter_shape
    self.out_size = out_size
    self.padding = padding
    self.stride = stride
    w_init = jax.nn.initializers.he_uniform(in_axis=-2, out_axis=-1)
    self.W = w_init(key, shape=self.filter_shape + (C, out_size))
    self.g = jnp.array(1.0)
    self.b = jnp.zeros(out_size)

  def data_dependent_init(self,
                          x: Array,
                          y: Optional[Array] = None,
                          key: PRNGKeyArray = None) -> eqx.Module:
    """Initialize the parameters of the layer based on the data.

    **Arguments**:

    - `x`: The data to initialize the parameters with.
    - `key`: A `jax.random.PRNGKey` for initialization

    **Returns**:
    A new layer with the parameters initialized.
    """
    assert x.shape[1:] == self.input_shape, 'Only works on batched data'

    # Initialize g and b.
    W = self.W*jax.lax.rsqrt((self.W**2).sum(axis=(0, 1, 2)))[None,None,None,:]
    x = util.conv(W, x, stride=self.stride, padding=self.padding)

    std = jnp.std(x.reshape((-1, x.shape[-1])), axis=0) + 1e-5
    g = 1/std

    x *= g

    mean = jnp.mean(x.reshape((-1, x.shape[-1])), axis=0)
    b = -mean

    # Turn the new parameters into a new module
    get_g = lambda tree: tree.g
    get_b = lambda tree: tree.b
    updated_layer = eqx.tree_at(get_g, self, g)
    updated_layer = eqx.tree_at(get_b, updated_layer, b)

    return updated_layer

  def __call__(self, x: Array, y: Array = None) -> Array:
    assert x.shape == self.input_shape, 'Only works on unbatched data'
    W = self.W*jax.lax.rsqrt((self.W**2).sum(axis=(0, 1, 2)))[None,None,None,:]
    x = self.g*util.conv(W, x, stride=self.stride, padding=self.padding) + self.b
    return x


class WeightStandardizedConv(eqx.Module):
  """Weight standardized parametrized convolutional layer
  https://arxiv.org/pdf/1903.10520.pdf
  """

  input_shape: int = eqx.field(static=True)
  out_size: int = eqx.field(static=True)
  filter_shape: Tuple[int] = eqx.field(static=True)
  padding: Union[int,str] = eqx.field(static=True)
  stride: int = eqx.field(static=True)
  W: Array
  b: Array

  def __init__(self,
               input_shape: Tuple[int], # in_channels
               filter_shape: Tuple[int],
               out_size: int,
               *,
               key: PRNGKeyArray,
               padding: Union[int,str] = 'SAME',
               stride: int = 1,
               **kwargs):
    super().__init__(**kwargs)
    H, W, C = input_shape

    self.input_shape = input_shape
    self.filter_shape = filter_shape
    self.out_size = out_size
    self.padding = padding
    self.stride = stride

    w_init = jax.nn.initializers.he_uniform(in_axis=-2, out_axis=-1)
    self.W = w_init(key, shape=self.filter_shape + (C, out_size))
    self.b = jnp.zeros(out_size)

  def data_dependent_init(self,
                          x: Array,
                          y: Optional[Array] = None,
                          key: PRNGKeyArray = None) -> eqx.Module:
    """Initialize the parameters of the layer based on the data.

    **Arguments**:

    - `x`: The data to initialize the parameters with.
    - `key`: A `jax.random.PRNGKey` for initialization

    **Returns**:
    A new layer with the parameters initialized.
    """

    axes = (0, 1, 2)
    mean = jnp.mean(self.W, axis=axes, keepdims=True)
    var = jnp.var(self.W, axis=axes, keepdims=True)

    W_hat = (self.W - mean)/jnp.sqrt(var + 1e-5)
    x = util.conv(W_hat, x, stride=self.stride, padding=self.padding)

    # Initialize b.
    mean = jnp.mean(x.reshape((-1, x.shape[-1])), axis=0)
    b = -mean

    # Turn the new parameters into a new module
    get_b = lambda tree: tree.b
    updated_layer = eqx.tree_at(get_b, self, b)

    return updated_layer

  def __call__(self, x: Array, y: Array = None) -> Array:
    assert x.shape == self.input_shape, 'Only works on unbatched data'

    axes = (0, 1, 2)
    mean = jnp.mean(self.W, axis=axes, keepdims=True)
    var = jnp.var(self.W, axis=axes, keepdims=True)

    H, W, C_in, C_out = self.W.shape
    fan_in = H*W*C_in
    W_hat = (self.W - mean)*jax.lax.rsqrt(fan_in*var + 1e-5)
    x = util.conv(W_hat, x, stride=self.stride, padding=self.padding) + self.b
    return x

class ChannelConvention(eqx.Module):
  module: eqx.Module
  def __init__(self, module: eqx.Module):
    super().__init__()
    self.module = module

  def __call__(self, x):
    x = einops.rearrange(x, 'H W C -> C H W')
    x = self.module(x)
    x = einops.rearrange(x, 'C H W -> H W C')
    return x

class ConvAndGroupNorm(eqx.Module):
  """Weight standardized conv + group norm
  """
  input_shape: int = eqx.field(static=True)
  conv: WeightStandardizedConv
  norm: eqx.nn.GroupNorm

  def __init__(self,
               input_shape: Tuple[int], # in_channels
               filter_shape: Tuple[int],
               out_size: int,
               groups: int,
               *,
               key: PRNGKeyArray,
               padding: Union[int,str] = 'SAME',
               stride: int = 1,
               **kwargs):
    super().__init__(**kwargs)

    if out_size%groups != 0:
      raise ValueError("The number of groups must divide the number of channels.")

    self.conv = WeightStandardizedConv(input_shape=input_shape,
                                        filter_shape=filter_shape,
                                        out_size=out_size,
                                        key=key,
                                        padding=padding,
                                        stride=stride)
    self.norm = ChannelConvention(eqx.nn.GroupNorm(groups=groups, channels=out_size))
    self.input_shape = self.conv.input_shape

  def data_dependent_init(self,
                          x: Array,
                          y: Optional[Array] = None,
                          shift_scale: Optional[Array] = None,
                          key: PRNGKeyArray = None) -> eqx.Module:
    """Initialize the parameters of the layer based on the data.

    **Arguments**:

    - `x`: The data to initialize the parameters with.
    - `key`: A `jax.random.PRNGKey` for initialization

    **Returns**:
    A new layer with the parameters initialized.
    """
    new_conv = self.conv.data_dependent_init(x, y, key=key)
    get_conv = lambda tree: tree.conv
    updated_layer = eqx.tree_at(get_conv, self, new_conv)
    return updated_layer

  def __call__(self,
               x: Array,
               y: Array = None) -> Array:
    assert x.shape == self.input_shape, 'Only works on unbatched data'
    x = self.conv(x)
    x = self.norm(x)
    return x

################################################################################################################

class Upsample(eqx.Module):
  """https://arxiv.org/ftp/arxiv/papers/1707/1707.02937.pdf
  """

  input_shape: int = eqx.field(static=True)
  out_size: int = eqx.field(static=True)
  conv: WeightStandardizedConv

  def __init__(self,
               input_shape: Tuple[int],
               out_size: Optional[int] = None,
               *,
               key: PRNGKeyArray,
               **kwargs):
    super().__init__(**kwargs)
    H, W, C = input_shape
    self.input_shape = input_shape
    self.out_size = out_size if out_size is not None else C
    self.conv = WeightStandardizedConv(input_shape=(H, W, C),
                                       filter_shape=(3, 3),
                                       out_size=4*self.out_size,
                                       key=key)

  def data_dependent_init(self, *args, **kwargs) -> eqx.Module:
    return self

  def __call__(self, x: Array, y: Array = None) -> Array:
    assert x.shape == self.input_shape, 'Only works on unbatched data'
    H, W, C = x.shape
    x = self.conv(x)
    x = jax.nn.silu(x)
    x = einops.rearrange(x, 'h w (c k1 k2) -> (h k1) (w k2) c', k1=2, k2=2)
    assert x.shape == (H*2, W*2, self.out_size)
    return x

class Downsample(eqx.Module):

  input_shape: int = eqx.field(static=True)
  out_size: int = eqx.field(static=True)
  conv: WeightStandardizedConv

  def __init__(self,
               input_shape: Tuple[int],
               out_size: Optional[int] = None,
               *,
               key: PRNGKeyArray,
               **kwargs):
    super().__init__(**kwargs)
    H, W, C = input_shape
    self.input_shape = input_shape
    self.out_size = out_size if out_size is not None else C
    self.conv = WeightStandardizedConv(input_shape=(H//2, W//2, C*4),
                                       filter_shape=(3, 3),
                                       out_size=self.out_size,
                                       key=key)

  def data_dependent_init(self, *args, **kwargs) -> eqx.Module:
    return self

  def __call__(self, x: Array, y: Array = None) -> Array:
    assert x.shape == self.input_shape, 'Only works on unbatched data'
    H, W, C = x.shape
    x = einops.rearrange(x, '(h k1) (w k2) c -> h w (c k1 k2)', k1=2, k2=2)
    x = self.conv(x)
    assert x.shape == (H//2, W//2, self.out_size)
    return x

################################################################################################################

class GatedGlobalContext(eqx.Module):
  """Modified version of https://arxiv.org/pdf/1904.11492.pdf used in imagen https://github.com/lucidrains/imagen-pytorch/"""

  input_shape: int = eqx.field(static=True)
  linear1: WeightNormConv
  linear2: WeightNormConv
  context_conv: WeightNormConv

  def __init__(self,
               input_shape: Tuple[int],
               *,
               key: PRNGKeyArray,
               **kwargs):
    super().__init__(**kwargs)
    H, W, C = input_shape
    self.input_shape = input_shape
    out_size = C

    hidden_dim = max(3, out_size//2)
    k1, k2, k3 = random.split(key, 3)
    self.linear1 = WeightNormDense(in_size=C,
                                   out_size=hidden_dim,
                                   key=k1)

    self.linear2 = WeightNormDense(in_size=hidden_dim,
                                out_size=out_size,
                                key=k2)

    self.context_conv = WeightNormConv(input_shape=input_shape,
                                       filter_shape=(1, 1),
                                       out_size=1,
                                       key=k3)

  def data_dependent_init(self, *args, **kwargs) -> eqx.Module:
    return self

  def __call__(self, x: Array, y: Array = None) -> Array:
    assert x.shape == self.input_shape, 'Only works on unbatched data'
    x_in = x
    H, W, C = x.shape

    # Reduce channels to (H, W, 1)
    context = self.context_conv(x)

    # Flatten
    c_flat = einops.rearrange(context, 'h w c -> (h w) c')
    x_flat = einops.rearrange(x, 'h w c -> (h w) c')

    # Context over the pixels
    c_sm = jax.nn.softmax(c_flat, axis=0)

    # Reweight the channels
    out = jnp.einsum('tu,tv->uv', c_sm, x_flat)
    assert out.shape == (1, C)
    out = out[0]

    out = self.linear1(out)
    out = jax.nn.silu(out)
    out = self.linear2(out)
    out = jax.nn.sigmoid(out)
    return x_in*out[None,None,:]

################################################################################################################

class Attention(eqx.Module):

  input_shape: int = eqx.field(static=True)
  heads: int = eqx.field(static=True)
  dim_head: int = eqx.field(static=True)
  scale: float = eqx.field(static=True)

  conv_in: eqx.nn.Conv3d
  conv_out: eqx.nn.Conv3d

  def __init__(self,
               input_shape: Tuple[int],
               heads: int = 4,
               dim_head: int = 32,
               scale: float = 10,
               *,
               key: PRNGKeyArray,
               **kwargs):
    super().__init__(**kwargs)
    H, W, C = input_shape
    self.input_shape = input_shape
    self.heads = heads
    self.dim_head = dim_head
    self.scale = scale

    k1, k2 = random.split(key, 2)
    dim = self.dim_head*self.heads
    self.conv_in = ChannelConvention(eqx.nn.Conv2d(in_channels=C,
                                                   out_channels=3*dim,
                                                   kernel_size=1,
                                                   use_bias=False,
                                                   key=k1))
    self.conv_out = ChannelConvention(eqx.nn.Conv2d(in_channels=dim,
                                                    out_channels=C,
                                                    kernel_size=1,
                                                    use_bias=True,
                                                    key=k2))

  def data_dependent_init(self, *args, **kwargs) -> eqx.Module:
    return self

  def __call__(self, x: Array, y: Array = None) -> Array:
    assert x.shape == self.input_shape, 'Only works on unbatched data'
    H, W, C = x.shape
    qkv = self.conv_in(x) # (H, W, heads*dim_head*3)
    qkv = einops.rearrange(qkv, 'H W (u h d) -> (H W) h d u', h=self.heads, d=self.dim_head, u=3)
    q, k, v = jnp.split(qkv, 3, axis=-1)
    q, k, v = q[...,0], k[...,0], v[...,0]
    assert q.shape == k.shape == v.shape == (H*W, self.heads, self.dim_head)

    def normalize(x):
      return x/jnp.clip(jnp.linalg.norm(x, axis=0, keepdims=True), 1e-8)
    q, k = normalize(q), normalize(k)

    sim = jnp.einsum('ihd,jhd->hij', q, k)*self.scale
    attn = jax.nn.softmax(sim, axis=-1)
    assert attn.shape == (self.heads, H*W, H*W)

    out = jnp.einsum('hij,jhd->hid', attn, v)
    out = einops.rearrange(out, 'h (H W) d -> H W (h d)', H=H, W=W, h=self.heads, d=self.dim_head)
    assert out.shape == (H, W, self.dim_head*self.heads)

    out = self.conv_out(out)
    return out

class LinearAttention(eqx.Module):

  input_shape: int = eqx.field(static=True)
  heads: int = eqx.field(static=True)
  dim_head: int = eqx.field(static=True)

  conv_in: eqx.nn.Conv3d
  conv_out: eqx.nn.Conv3d
  norm: eqx.nn.LayerNorm

  def __init__(self,
               input_shape: Tuple[int],
               heads: int = 4,
               dim_head: int = 32,
               *,
               key: PRNGKeyArray,
               **kwargs):
    super().__init__(**kwargs)
    H, W, C = input_shape
    self.input_shape = input_shape
    self.heads = heads
    self.dim_head = dim_head

    k1, k2 = random.split(key, 2)
    dim = self.dim_head*self.heads
    self.conv_in = ChannelConvention(eqx.nn.Conv2d(in_channels=C,
                                                   out_channels=3*dim,
                                                   kernel_size=1,
                                                   use_bias=False,
                                                   key=k1))
    self.conv_out = ChannelConvention(eqx.nn.Conv2d(in_channels=dim,
                                                    out_channels=C,
                                                    kernel_size=1,
                                                    use_bias=True,
                                                    key=k2))
    self.norm = eqx.nn.LayerNorm(shape=(C,), use_bias=False)

  def data_dependent_init(self, *args, **kwargs) -> eqx.Module:
    return self

  def __call__(self, x: Array, y: Array = None) -> Array:
    assert x.shape == self.input_shape, 'Only works on unbatched data'
    H, W, C = x.shape
    qkv = self.conv_in(x) # (H, W, heads*dim_head*3)
    qkv = einops.rearrange(qkv, 'H W (u h d) -> (H W) h d u', h=self.heads, d=self.dim_head, u=3)
    q, k, v = jnp.split(qkv, 3, axis=-1)
    q, k, v = q[...,0], k[...,0], v[...,0]
    assert q.shape == k.shape == v.shape == (H*W, self.heads, self.dim_head)

    q = jax.nn.softmax(q, axis=-1)
    k = jax.nn.softmax(k, axis=-3)

    q = q/jnp.sqrt(self.dim_head)
    v = v/(H*W)

    context = jnp.einsum("n h d, n h e -> h d e", k, v)
    out = jnp.einsum("h d e, n h d -> h e n", context, q)
    out = einops.rearrange(out, "h e (x y) -> x y (h e)", x=H)
    assert out.shape == (H, W, self.dim_head*self.heads)

    out = self.conv_out(out)
    out = eqx.filter_vmap(eqx.filter_vmap(self.norm))(out)
    return out

class AttentionBlock(eqx.Module):

  input_shape: int = eqx.field(static=True)
  attn: Union[Attention, LinearAttention]
  norm: eqx.nn.LayerNorm

  def __init__(self,
               input_shape: Tuple[int],
               heads: int = 4,
               dim_head: int = 32,
               *,
               key: PRNGKeyArray,
               use_linear_attention: bool = True,
               **kwargs):
    super().__init__(**kwargs)

    if use_linear_attention:
      self.attn = LinearAttention(input_shape=input_shape,
                                  heads=heads,
                                  dim_head=dim_head,
                                  key=key)
    else:
      self.attn = Attention(input_shape=input_shape,
                            heads=heads,
                            dim_head=dim_head,
                            key=key)
    self.input_shape = self.attn.input_shape
    H, W, C = input_shape
    self.norm = eqx.nn.LayerNorm(shape=(C,), use_bias=False)

  def data_dependent_init(self, *args, **kwargs) -> eqx.Module:
    return self

  def __call__(self, x: Array, y: Array = None) -> Array:
    assert x.shape == self.input_shape, 'Only works on unbatched data'
    normed_x = eqx.filter_vmap(eqx.filter_vmap(self.norm))(x)
    out = self.attn(normed_x)
    return out + x

################################################################################################################

if __name__ == '__main__':
  from debug import *
  import matplotlib.pyplot as plt
  from generax.flows.base import Sequential

  key = random.PRNGKey(0)
  x, y = random.normal(key, shape=(2, 10, 4, 4, 3))
  y, cond_size = None, None

  layer = ConvAndGroupNorm(input_shape=x.shape[1:],
                      filter_shape=(3, 3),
                      groups=4,
                         out_size=16,
                         key=key)

  layer = LinearAttention(input_shape=x.shape[1:],
                        key=key)

  layer(util.unbatch(x), util.unbatch(y))

  out = eqx.filter_vmap(layer)(x, y)

  layer = layer.data_dependent_init(x, y, key=key)
  out = eqx.filter_vmap(layer)(x, y)
  import pdb; pdb.set_trace()





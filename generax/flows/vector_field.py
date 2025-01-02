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
import numpy as np
from generax.flows.base import BijectiveTransform, TimeDependentBijectiveTransform
from generax.nn.resnet import TimeDependentResNet, ResNet

class FlowVF(eqx.Module):
  flow: BijectiveTransform
  w: Array
  input_shape: Tuple[int]

  def __init__(self,
               flow: BijectiveTransform):
    self.flow = flow
    self.w = jnp.ones(self.flow.input_shape)
    self.input_shape = self.flow.input_shape

  def __call__(self, x: Array) -> Array:
    return self.vf_and_div(x)[0]

  def divergence(self, x: Array) -> Array:
    return self.vf_and_div(x)[1]

  def vf_and_div(self, x: Array) -> Array:
    assert x.shape == self.flow.input_shape
    z, _ = self.flow(x)
    def fwd(z):
      x, log_det = self.flow(z, inverse=True)
      return x, log_det
    x_reconstr, (v, div) = jax.jvp(fwd, (z,), (self.w,))
    return v, div


class TimeDependentFlowVF(eqx.Module):
  flow: TimeDependentBijectiveTransform
  net: eqx.Module
  input_shape: Tuple[int]

  def __init__(self,
               flow: TimeDependentBijectiveTransform):
    self.flow = flow
    self.input_shape = self.flow.input_shape
    key = random.PRNGKey(0)
    self.net = TimeDependentResNet(input_shape=(1,),
                              working_size=3,
                              hidden_size=64,
                              out_size=1,
                              n_blocks=4,
                              embedding_size=32,
                              out_features=8,
                              key=key)

  def __call__(self, t: Array, x: Array, y=None) -> Array:
    return self.vf_and_div(t, x)[0]

  def divergence(self, t: Array, x: Array) -> Array:
    return self.vf_and_div(t, x)[1]

  def vf_and_div(self, t: Array, x: Array) -> Array:
    assert x.shape == self.flow.input_shape
    z, _ = self.flow(t, x)
    def fwd(z):
      x, log_det = self.flow(t, z, inverse=True)
      return x, log_det

    @eqx.filter_vmap
    def _net(x):
      return self.net(t, x[None])[0]

    z_flat = z.ravel()
    w_flat, dw = eqx.filter_jvp(_net, (z_flat,), (jnp.ones_like(z_flat),))
    w = w_flat.reshape(z.shape)
    elt_div = dw.sum()

    (x_reconstr, log_det), (v, div) = jax.jvp(fwd, (z,), (w,))
    return v, elt_div + div

################################################################################################################

if __name__ == '__main__':
  from debug import *
  import matplotlib.pyplot as plt
  from generax.flows.base import Sequential
  import generax as gx

  # switch to x64
  jax.config.update("jax_enable_x64", True)

  key = random.PRNGKey(0)
  x = random.normal(key, shape=(10, 3))
  x_shape = x.shape[1:]

  A = gx.DenseAffine(input_shape=x_shape, key=key)

  flow_vf = FlowVF(flow=A)
  v, div = flow_vf.vf_and_div(x[0])
  import pdb; pdb.set_trace()

import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
import einops
import equinox as eqx
from abc import ABC, abstractmethod
from jaxtyping import Array, PRNGKeyArray
import generax.nn.util as util
from generax.flows.base import BijectiveTransform
from generax.nn.neural_ode import NeuralODE

class FFJORDTransform(BijectiveTransform):
  """Flow parametrized by a neural ODE https://arxiv.org/pdf/1810.01367.pdf

  **Attributes**:

  - `input_shape`: The input shape.  Output shape will have the same dimensionality
                  as the input.
  - `cond_shape`: The shape of the conditioning information.  If there is no
                  conditioning information, this is None.
  - `neural_ode`: The neural ODE
  - `trace_estimate_likelihood`: Whether to use a trace estimate for the likelihood.
  - `adjoint`: The adjoint method to use.  Can be one of the following:
  - `key`: The random key to use for initialization
  """

  neural_ode: NeuralODE
  trace_estimate_likelihood: bool

  def __init__(self,
               input_shape: Tuple[int],
               vf: eqx.Module,
               *,
               controller_rtol: Optional[float] = 1e-3,
               controller_atol: Optional[float] = 1e-5,
               trace_estimate_likelihood: Optional[bool] = False,
               adjoint='recursive_checkpoint',
               key: PRNGKeyArray,
               **kwargs):
    """**Arguments**:

    - `input_shape`: The shape of the input to the transformation
    - `vf`: A function that computes the vector field.  It must output
            a vector of the same shape as its input.
    - `controller_rtol`: The relative tolerance of the stepsize controller.
    - `controller_atol`: The absolute tolerance of the stepsize controller.
    - `trace_estimate_likelihood`: Whether to use a trace estimate for the likelihood.
    - `adjoint`: The adjoint method to use.  Can be one of the following:
        - `"recursive_checkpoint"`: Use the recursive checkpoint method.  Doesn't support jvp.
        - `"direct"`: Use the direct method.  Supports jvps.
        - `"seminorm"`: Use the seminorm method.  Does fast backprop through the solver.
    - `key`: The random key to use for initialization
    """
    self.neural_ode = NeuralODE(vf=vf,
                                adjoint=adjoint,
                                controller_rtol=controller_rtol,
                                controller_atol=controller_atol)
    self.trace_estimate_likelihood = trace_estimate_likelihood

    super().__init__(input_shape=input_shape,
                     **kwargs)

  def __call__(self,
               x: Array,
               y: Optional[Array] = None,
               inverse: bool=False) -> Array:
    """**Arguments**:

    - `x`: The input to the transformation
    - `y`: The conditioning information
    - `inverse`: Whether to inverse the transformation

    **Returns**:
    `(z, log_det)`
    """
    assert x.shape == self.input_shape, 'Only works on unbatched data'

    z, log_det = self.neural_ode(x,
                                 y=y,
                                 inverse=inverse,
                                 log_likelihood=True,
                                 trace_estimate_likelihood=self.trace_estimate_likelihood,
                                 save_at=None)
    return z, log_det

################################################################################################################

if __name__ == '__main__':
  from debug import *
  import matplotlib.pyplot as plt
  from generax.flows.base import Sequential
  from generax.nn.resnet_1d import ResNet1d
  from generax.nn.resnet_1d import TimeDependentResNet1d
  # enable x64
  from jax.config import config
  config.update("jax_enable_x64", True)


  key = random.PRNGKey(0)
  x = random.normal(key, shape=(10, 2))

  net = TimeDependentResNet1d(in_size=x.shape[-1],
                              working_size=8,
                              hidden_size=16,
                              out_size=x.shape[-1],
                              n_blocks=4,
                              cond_size=None,
                              embedding_size=16,
                              out_features=8,
                              key=key)

  layer = FFJORDTransform(input_shape=x.shape[1:],
                 vf=net,
                 key=key,
                 controller_rtol=1e-8,
                 controller_atol=1e-8)



  x = random.normal(key, shape=(2, 2))
  layer = layer.data_dependent_init(x, key=key)

  z, log_det = eqx.filter_vmap(layer)(x)

  z, log_det = layer(x[0])
  x_reconstr, log_det2 = layer(z, inverse=True)

  G = jax.jacobian(lambda x: layer(x)[0])(x[0])
  log_det_true = jnp.linalg.slogdet(G)[1]

  assert jnp.allclose(log_det, log_det_true)
  assert jnp.allclose(x[0], x_reconstr)


  import pdb; pdb.set_trace()

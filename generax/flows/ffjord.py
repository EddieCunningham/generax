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
from generax.nn.neural_ode import NeuralODE
from generax.nn.resnet import TimeDependentResNet

__all__ = ['FFJORDTransform']

class FFJORDTransform(BijectiveTransform):
  """Flow parametrized by a neural ODE https://arxiv.org/pdf/1810.01367.pdf

  **Attributes**:

  - `input_shape`: The input shape.  Output shape will have the same dimensionality
                  as the input.
  - `cond_shape`: The shape of the conditioning information.  If there is no
                  conditioning information, this is None.
  - `neural_ode`: The neural ODE
  - `adjoint`: The adjoint method to use.  Can be one of the following:
  - `key`: The random key to use for initialization
  """

  neural_ode: NeuralODE

  def __init__(self,
               input_shape: Tuple[int],
               net: eqx.Module = None,
               working_size: int = 16,
               hidden_size: int = 32,
               n_blocks: int = 4,
               time_embedding_size = 16,
               n_time_features = 8,
               cond_shape: Optional[Tuple[int]] = None,
               *,
               controller_rtol: Optional[float] = 1e-8,
               controller_atol: Optional[float] = 1e-8,
               adjoint='recursive_checkpoint',
               key: PRNGKeyArray,
               **kwargs):
    """**Arguments**:

    - `input_shape`: The shape of the input to the transformation
    - `net`: The neural network to use for the vector field.  If None, a default
              network will be used.  `net` should accept `net(t, x, y=y)`
    - `controller_rtol`: The relative tolerance of the stepsize controller.
    - `controller_atol`: The absolute tolerance of the stepsize controller.
    - `trace_estimate_likelihood`: Whether to use a trace estimate for the likelihood.
    - `adjoint`: The adjoint method to use.  Can be one of the following:
        - `"recursive_checkpoint"`: Use the recursive checkpoint method.  Doesn't support jvp.
        - `"direct"`: Use the direct method.  Supports jvps.
        - `"seminorm"`: Use the seminorm method.  Does fast backprop through the solver.
    - `key`: The random key to use for initialization
    """

    if net is None:
      net = TimeDependentResNet(input_shape=input_shape,
                            working_size=working_size,
                            hidden_size=hidden_size,
                            out_size=input_shape[-1],
                            n_blocks=n_blocks,
                            cond_shape=cond_shape,
                            embedding_size=time_embedding_size,
                            out_features=n_time_features,
                            key=key)

    self.neural_ode = NeuralODE(vf=net,
                                adjoint=adjoint,
                                controller_rtol=controller_rtol,
                                controller_atol=controller_atol)

    super().__init__(input_shape=input_shape,
                     **kwargs)

  @property
  def vector_field(self):
    return self.neural_ode.vector_field

  def __call__(self,
               x: Array,
               y: Optional[Array] = None,
               inverse: bool = False,
               log_likelihood: bool = True,
               trace_estimate_likelihood: Optional[bool] = False,
               save_at: Optional[Array] = None,
               key: Optional[PRNGKeyArray] = None,
               **kwargs) -> Array:
    """**Arguments**:

    - `x`: The input to the transformation
    - `y`: The conditioning information
    - `inverse`: Whether to inverse the transformation
    - `log_likelihood`: Whether to compute the log likelihood of the transformation
    - `trace_estimate_likelihood`: Whether to compute a trace estimate of the likelihood of the neural ODE.
    - `save_at`: The times to save the neural ODE at.
    - `key`: The random key to use for initialization

    **Returns**:
    `(z, log_det)`
    """
    assert x.shape == self.input_shape, 'Only works on unbatched data'
    if log_likelihood and (trace_estimate_likelihood and (key is None)):
      raise TypeError(f'When using trace estimation, must pass random key')

    if log_likelihood == False:
      trace_estimate_likelihood = False

    solution = self.neural_ode(x,
                                 y=y,
                                 inverse=inverse,
                                 log_likelihood=log_likelihood,
                                 trace_estimate_likelihood=trace_estimate_likelihood,
                                 save_at=save_at,
                                 key=key,
                                 **kwargs)
    return solution.ys, solution.log_det

################################################################################################################

if __name__ == '__main__':
  from debug import *
  import matplotlib.pyplot as plt
  from generax.flows.base import Sequential
  from generax.nn.resnet import TimeDependentResNet
  # enable x64
  from jax.config import config
  config.update("jax_enable_x64", True)


  key = random.PRNGKey(0)
  x = random.normal(key, shape=(10, 2))

  layer = FFJORDTransform(input_shape=x.shape[1:],
                 working_size=16,
                 hidden_size=32,
                 n_blocks=2,
                 time_embedding_size=8,
                 n_time_features=4,
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

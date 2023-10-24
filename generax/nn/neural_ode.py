import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
import einops
import equinox as eqx
from abc import ABC, abstractmethod
import diffrax
from jaxtyping import Array, PRNGKeyArray

__all__ = ['NeuralODE']

class NeuralODE(eqx.Module):
  """Neural ODE"""

  vector_field: eqx.Module
  adjoint: diffrax.AbstractAdjoint
  stepsize_controller: diffrax.AbstractAdaptiveStepSizeController

  def __init__(self,
               vf: eqx.Module,
               adjoint: Optional[str] = "recursive_checkpoint",
               controller_rtol: Optional[float] = 1e-3,
               controller_atol: Optional[float] = 1e-5,
  ):
    """**Arguments**:

    - `vf`: A function that computes the vector field.  It must output
            a vector of the same shape as its input.
    - `adjoint`: The adjoint method to use.  Can be one of the following:
       - `"recursive_checkpoint"`: Use the recursive checkpoint method.  Doesn't support jvp.
       - `"direct"`: Use the direct method.  Supports jvps.
       - `"seminorm"`: Use the seminorm method.  Does fast backprop through the solver.
    - `controller_rtol`: The relative tolerance of the stepsize controller.
    - `controller_atol`: The absolute tolerance of the stepsize controller.
    """
    self.vector_field = vf

    if adjoint == "recursive_checkpoint":
      self.adjoint = diffrax.RecursiveCheckpointAdjoint()
    elif adjoint == "direct":
      self.adjoint = diffrax.DirectAdjoint()
    elif adjoint == "seminorm":
      adjoint_controller = diffrax.PIDController(
          rtol=1e-3, atol=1e-6, norm=diffrax.adjoint_rms_seminorm)
      self.adjoint = diffrax.BacksolveAdjoint(stepsize_controller=adjoint_controller)

    self.stepsize_controller = diffrax.PIDController(rtol=controller_rtol, atol=controller_atol)

  def __call__(self,
               x: Array,
               key: Optional[PRNGKeyArray] = None,
               *,
               inverse: Optional[bool] = False,
               log_likelihood: Optional[bool] = False,
               trace_estimate_likelihood: Optional[bool] = False,
               save_at: Optional[Array] = None) -> Array:
    """**Arguemnts**:

    - `x`: The input to the neural ODE.  Must be a rank 1 array.
    - `key`: The random number generator key.
    - `inverse`: Whether to compute the inverse of the neural ODE.  `inverse=True`
               corresponds to going from the base space to the data space.
    - `log_likelihood`: Whether to compute the log likelihood of the neural ODE.
    - `trace_estimate_likelihood`: Whether to compute a trace estimate of the likelihood of the neural ODE.
    - `save_at`: The times to save the neural ODE at.

    **Returns**:
    - `z`: The output of the neural ODE.
    - `log_likelihood`: The log likelihood of the neural ODE if `log_likelihood=True`.
    """
    assert x.ndim == 1, "x must be unbatched"

    if trace_estimate_likelihood:
      # Get a random vector for hutchinsons trace estimator
      k1, _ = random.split(key, 2)
      v = random.normal(k1, x.shape)

    # Split the model into its static and dynamic parts so that backprop
    # through the ode solver can be faster.
    params, static = eqx.partition(self.vector_field, eqx.is_array)

    def f(t, x_and_logpx, params):
      x, log_px = x_and_logpx

      if inverse == False:
        # If we're inverting the flow, we need to adjust the time
        t = 1.0 - t

      # Recombine the model
      model = eqx.combine(params, static)

      # Fill the model with the current time
      def apply_vf(x):
        return model(t, x)

      if log_likelihood:
        if trace_estimate_likelihood:
          # Hutchinsons trace estimator.  See FFJORD https://arxiv.org/pdf/1810.01367.pdf
          dxdt, dudxv = jax.jvp(apply_vf, (x,), (v,))
          dlogpxdt = -jnp.sum(dudxv*v)
        else:
          # Brute force dlogpx/dt.  See NeuralODE https://arxiv.org/pdf/1806.07366.pdf
          eye = jnp.eye(x.shape[-1])
          def jvp(dx):
            dx = jnp.broadcast_to(dx, x.shape)
            dxdt, d2dx_dtdx = jax.jvp(apply_vf, (x,), (dx,))
            return dxdt, d2dx_dtdx

          dxdt, d2dx_dtdx = jax.vmap(jvp)(eye)
          dxdt = dxdt[0]
          dlogpxdt = -jnp.trace(d2dx_dtdx)
      else:
        # Don't worry about the log likelihood
        dxdt = apply_vf(x)
        dlogpxdt = jnp.zeros_like(log_px)

      if inverse == False:
        # If we're inverting the flow, we need to flip the sign of dxdt
        dxdt = -dxdt
      return dxdt, dlogpxdt

    term = diffrax.ODETerm(f)
    solver = diffrax.Dopri5()

    # All of our flows will go from 0 to 1.
    t0, t1 = 0.0, 1.0

    # Determine which times we want to save the neural ODE at.
    if save_at is None:
      saveat = diffrax.SaveAt(ts=[t1])
    else:
      saveat = diffrax.SaveAt(ts=save_at)

    # Run the ODE solver
    solution = diffrax.diffeqsolve(term,
                                   solver,
                                   saveat=saveat,
                                   t0=t0,
                                   t1=t1,
                                   dt0=0.0001,
                                   y0=(x, jnp.zeros(x.shape[:1])),
                                   args=params,
                                   adjoint=self.adjoint,
                                   stepsize_controller=self.stepsize_controller,
                                   throw=True)
    outs = solution.ys

    if save_at is None:
      # Only take the first time
      outs = jax.tree_util.tree_map(lambda x: x[0], outs)

    z, log_px = outs
    if log_likelihood:
      return z, log_px
    return z

################################################################################################################
################################################################################################################
# TESTS
################################################################################################################
################################################################################################################

def test_basic_run(neural_ode, x):
  neural_ode(x[0])
  z = eqx.filter_vmap(neural_ode)(x)

def test_inverse(neural_ode, x):
  ts = jnp.linspace(0.0, 1.0, 20)
  z0 = neural_ode(x[0], save_at=ts)
  x0 = neural_ode(z0[-1], inverse=True, save_at=ts)
  assert jnp.allclose(z0[-1], x0[-1])

  z = eqx.filter_vmap(neural_ode)(x)
  x2 = eqx.filter_vmap(partial(neural_ode, inverse=True))(z)
  assert jnp.allclose(x, x2)

  import pdb; pdb.set_trace()

def test_log_likelihood(neural_ode):
  pass

def test_trace_estimator(neural_ode):
  pass

def test_multivariate(neural_ode):
  pass


if __name__ == "__main__":
  from debug import *
  jax.config.update("jax_enable_x64", True)
  import equinox as eqx
  from generax.nn.flat_net import TimeDependentResNet

  # Create some data
  key = random.PRNGKey(0)
  x = random.normal(key, (10, 2))

  # Network
  net = TimeDependentResNet(in_size=x.shape[-1],
                            out_size=x.shape[-1],
                            hidden_size=16,
                            n_blocks=6,
                            time_embedding_size=32,
                            key=key)
  neural_ode = NeuralODE(net)

  # Run the tests
  # test_basic_run(neural_ode, x)
  test_inverse(neural_ode, x)
  test_log_likelihood(neural_ode, x)
  test_trace_estimator(neural_ode, x)
  test_multivariate(neural_ode, x)


  import pdb; pdb.set_trace()

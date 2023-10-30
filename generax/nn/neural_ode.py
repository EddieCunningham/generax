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
               adjoint: Optional[str] = 'recursive_checkpoint',
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

    if adjoint == 'recursive_checkpoint':
      self.adjoint = diffrax.RecursiveCheckpointAdjoint()
    elif adjoint == 'direct':
      self.adjoint = diffrax.DirectAdjoint()
    elif adjoint == 'seminorm':
      assert 0, "There is a tracer leak somewhere "
      # TODO: Make a bug report
      adjoint_controller = diffrax.PIDController(
          rtol=1e-3, atol=1e-6, norm=diffrax.adjoint_rms_seminorm)
      self.adjoint = diffrax.BacksolveAdjoint(stepsize_controller=adjoint_controller)

    self.stepsize_controller = diffrax.PIDController(rtol=controller_rtol, atol=controller_atol)

  def __call__(self,
               x: Array,
               y: Optional[Array] = None,
               *,
               inverse: Optional[bool] = False,
               log_likelihood: Optional[bool] = False,
               trace_estimate_likelihood: Optional[bool] = False,
               save_at: Optional[Array] = None,
               key: Optional[PRNGKeyArray] = None) -> Array:
    """**Arguemnts**:

    - `x`: The input to the neural ODE.  Must be a rank 1 array.
    - `key`: The random number generator key.
    - `inverse`: Whether to compute the inverse of the neural ODE.  `inverse=True`
               corresponds to going from the base space to the data space.
    - `log_likelihood`: Whether to compute the log likelihood of the neural ODE.
    - `trace_estimate_likelihood`: Whether to compute a trace estimate of the likelihood of the neural ODE.
    - `save_at`: The times to save the neural ODE at.
    - `key`: The random key to use for initialization

    **Returns**:
    - `z`: The output of the neural ODE.
    - `log_likelihood`: The log likelihood of the neural ODE if `log_likelihood=True`.
    """
    assert x.shape == self.vector_field.input_shape

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
        return model(t, x, y=y)

      if log_likelihood:
        if trace_estimate_likelihood:
          # Hutchinsons trace estimator.  See ContinuousNormalizingFlow https://arxiv.org/pdf/1810.01367.pdf
          dxdt, dudxv = jax.jvp(apply_vf, (x,), (v,))
          dlogpxdt = -jnp.sum(dudxv*v)
        else:
          # Brute force dlogpx/dt.  See NeuralODE https://arxiv.org/pdf/1806.07366.pdf
          x_flat = x.ravel()
          eye = jnp.eye(x_flat.shape[-1])
          x_shape = x.shape

          def jvp_flat(x_flat, dx_flat):
            x = x_flat.reshape(x_shape)
            dx = dx_flat.reshape(x_shape)
            dxdt, d2dx_dtdx = jax.jvp(apply_vf, (x,), (dx,))
            return dxdt, d2dx_dtdx.ravel()

          dxdt, d2dx_dtdx_flat = jax.vmap(jvp_flat, in_axes=(None, 0))(x_flat, eye)
          dxdt = dxdt[0]
          dlogpxdt = -jnp.trace(d2dx_dtdx_flat)

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
                                   y0=(x, jnp.array(0.0)),
                                   args=params,
                                   adjoint=self.adjoint,
                                   stepsize_controller=self.stepsize_controller,
                                   throw=True)
    outs = solution.ys

    if save_at is None:
      # Only take the first time
      outs = jax.tree_util.tree_map(lambda x: x[0], outs)

    z, log_px = outs
    assert log_px.shape == ()

    if inverse:
      log_px = -log_px

    return z, log_px

import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable, Iterator
from jaxtyping import Array, PRNGKeyArray
from generax.trainer import Trainer
import generax.util.misc as misc
import matplotlib.pyplot as plt
import equinox as eqx
import generax as gx
from generax.distributions.flow_models import ContinuousNormalizingFlow

def get_dataset_iter():
  from sklearn.datasets import make_moons, make_swiss_roll
  data, y = make_moons(n_samples=100000, noise=0.01)
  data = data - data.mean(axis=0)
  data = data/data.std(axis=0)
  key = random.PRNGKey(0)

  def get_train_ds(key: PRNGKeyArray,
                  batch_size: int = 64) -> Iterator[Mapping[str, Array]]:
    total_choices = jnp.arange(data.shape[0])
    closed_over_data = data # In case we change the variable "data"
    while True:
      key, _ = random.split(key, 2)
      idx = random.choice(key,
                          total_choices,
                          shape=(batch_size,),
                          replace=True)
      yield dict(x=closed_over_data[idx])

  train_ds = get_train_ds(key)
  return train_ds

if __name__ == '__main__':
  from debug import *
  from generax.distributions.flow_models import *
  from generax.distributions.base import *
  from generax.nn import *

  train_ds = get_dataset_iter()
  data = next(train_ds)
  x = data['x']
  x_shape = x.shape[1:]
  key = random.PRNGKey(0)

  # Construct the flow
  net = gx.TimeDependentResNet(input_shape=x_shape,
                            working_size=16,
                            hidden_size=32,
                            out_size=x_shape[-1],
                            n_blocks=5,
                            embedding_size=16,
                            out_features=32,
                            key=key)
  flow = ContinuousNormalizingFlow(input_shape=x_shape,
                                    net=net,
                                    key=key,
                                    controller_atol=1e-5,
                                    controller_rtol=1e-5,
                                    adjoint='seminorm')

  # Construct the loss function
  def loss(flow, data, key):
    x = data['x']
    keys = random.split(key, x.shape[0])

    def solve_ode(x, key):
      return flow.neural_ode(x,
                             inverse=False,
                             log_likelihood=True,
                             trace_estimate_likelihood=False,
                             key=key)
    solution = jax.vmap(solve_ode)(x, keys)

    z = solution.ys
    log_det = solution.log_det
    log_pz = eqx.filter_vmap(flow.prior.log_prob)(z)
    log_px = log_pz + log_det

    total_vf_norm = solution.total_vf_norm
    total_jac_frob_norm = solution.total_jac_frob_norm

    objective = -log_px + 0.01*total_vf_norm + 0.01*total_jac_frob_norm
    objective = objective.mean()

    aux = dict(log_px=log_px,
               total_vf_norm=total_vf_norm,
               total_jac_frob_norm=total_jac_frob_norm)
    return objective, aux

  # Create the optimizer
  import optax
  schedule = optax.warmup_cosine_decay_schedule(init_value=0.0,
                                    peak_value=1.0,
                                    warmup_steps=1000,
                                    decay_steps=3e5,
                                    end_value=0.1,
                                    exponent=1.0)
  chain = []
  chain.append(optax.clip_by_global_norm(15.0))
  chain.append(optax.adamw(1e-3))
  chain.append(optax.scale_by_schedule(schedule))
  optimizer = optax.chain(*chain)

  # Create the trainer and optimize
  trainer = Trainer(checkpoint_path='tmp/flow/cnf_ml')
  flow = trainer.train(model=flow,
                       objective=loss,
                       evaluate_model=lambda x: x,
                       optimizer=optimizer,
                       num_steps=5000,
                       double_batch=-1,
                       data_iterator=train_ds,
                       checkpoint_every=5000,
                       test_every=-1,
                       retrain=True,
                       just_load=False)

  # Pull samples from the model
  keys = random.split(key, 1000)
  samples = eqx.filter_vmap(flow.sample)(keys)

  fig, ax = plt.subplots(1, 1)
  ax.scatter(*samples.T)
  plt.savefig('examples/cnf_ml_samples.png')
  # plt.show()
  import pdb; pdb.set_trace()
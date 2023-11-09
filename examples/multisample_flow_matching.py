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
  from generax.nn.resnet import TimeDependentResNet
  from generax.flows.affine import *
  from generax.distributions.flow_models import *
  from generax.distributions.base import *
  from generax.nn import *
  from generax.distributions.coupling import OTTCoupling

  train_ds = get_dataset_iter()
  data = next(train_ds)
  x = data['x']
  x_shape = x.shape[1:]
  key = random.PRNGKey(0)

  # Construct the neural network that learn the score
  net = TimeDependentResNet(input_shape=x_shape,
                            working_size=16,
                            hidden_size=32,
                            out_size=x_shape[-1],
                            n_blocks=3,
                            embedding_size=16,
                            out_features=32,
                            key=key)
  flow = ContinuousNormalizingFlow(input_shape=x_shape,
                                   net=net,
                                   key=key,
                                   controller_atol=1e-5,
                                   controller_rtol=1e-5)

  # Build the probability path that we'll use for learning.
  # The target probability path is the expectation of cond_ppath
  # with the expectation taken over the dataset.
  cond_ppath = TimeDependentNormalizingFlow(transform=ConditionalOptionalTransport(input_shape=x_shape, key=key),
                                            prior=Gaussian(input_shape=x_shape))

  # Construct the loss function
  def loss(flow, data, key):
    k1, k2 = random.split(key, 2)

    # Sample
    x1 = data['x']
    keys = random.split(k1, x1.shape[0])
    x0 = eqx.filter_vmap(cond_ppath.prior.sample)(keys)
    t = random.uniform(k2, shape=(x1.shape[0],))

    # Resample from the coupling
    coupling = OTTCoupling(x0, x1)
    x0 = coupling.sample_x0_given_x1(k1)

    # Compute f_t(x_0; x_1)
    def ft(t):
      return eqx.filter_vmap(cond_ppath.to_data_space)(t, x0, x1)
    xt, ut = jax.jvp(ft, (t,), (jnp.ones_like(t),))

    # Compute the parametric vector field
    vt = eqx.filter_vmap(flow.net)(t, xt)

    # Compute the loss
    objective = jnp.sum((ut - vt)**2).mean()

    aux = dict(objective=objective)
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
  trainer = Trainer(checkpoint_path='tmp/flow/multisample_flow_matching')
  flow = trainer.train(model=flow,
                       objective=loss,
                       evaluate_model=lambda x: x,
                       optimizer=optimizer,
                       num_steps=10000,
                       double_batch=1000,
                       data_iterator=train_ds,
                       checkpoint_every=5000,
                       test_every=-1,
                       retrain=True)

  # Pull samples from the model
  keys = random.split(key, 1000)
  samples = eqx.filter_vmap(flow.sample)(keys)

  fig, ax = plt.subplots(1, 1)
  ax.scatter(*samples.T)
  plt.show()
  import pdb; pdb.set_trace()
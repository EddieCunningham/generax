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
import generax.util as util

def get_dataset_iter():
  from sklearn.datasets import make_moons, make_swiss_roll
  data, y = make_moons(n_samples=100000, noise=0.01)
  # data, y = make_moons(n_samples=100000, noise=0.1)
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

  train_ds = get_dataset_iter()
  data = next(train_ds)
  x = data['x']
  x_shape = x.shape[1:]
  key = random.PRNGKey(0)

  # Construct the neural network that learn the score
  energy = TimeDependentResNet(input_shape=x_shape,
                            working_size=16,
                            hidden_size=32,
                            out_size=1,
                            n_blocks=3,
                            embedding_size=16,
                            out_features=32,
                            key=key)
  net = TimeDependentGradWrapper(energy)

  # Build the probability path that we'll use for learning.
  # The target probability path is the expectation of cond_ppath
  # with the expectation taken over the dataset.
  cond_ppath = TimeDependentNormalizingFlow(transform=ConditionalOptionalTransport(input_shape=x_shape, key=key),
                                            prior=Gaussian(input_shape=x_shape))

  # Construct the loss function
  def loss(net, data, key):
    k1, k2 = random.split(key, 2)

    def unbatched_loss(data, key):
      k1, k2 = random.split(key, 2)

      # Sample
      x1 = data['x']
      x0 = cond_ppath.prior.sample(k1)
      t = random.uniform(k2)
      xt = cond_ppath.to_data_space(t, x0, x1)

      # Compute the score
      def log_prob(xt):
        return cond_ppath.log_prob(t, xt, x1)
      grad_logptx = eqx.filter_grad(log_prob)(xt)

      # import pdb; pdb.set_trace()

      # Compute the parametric score
      st = net(t, xt)

      # Compute the loss
      return jnp.sum((grad_logptx - st)**2)

    keys = random.split(key, data['x'].shape[0])
    objective = jax.vmap(unbatched_loss)(data, keys).mean()
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
  chain.append(optax.adamw(1e-4))
  chain.append(optax.scale_by_schedule(schedule))
  optimizer = optax.chain(*chain)

  # Create the trainer and optimize
  trainer = Trainer(checkpoint_path='tmp/flow/score_matching')
  net = trainer.train(model=net,
                      objective=loss,
                      evaluate_model=lambda x: x,
                      optimizer=optimizer,
                      num_steps=50000,
                      double_batch=1000,
                      data_iterator=train_ds,
                      checkpoint_every=5000,
                      test_every=-1,
                      retrain=True)

  # Plot the energy function to check that we've learned the right thing
  N = 50
  x_range, y_range = jnp.linspace(-2, 2, N), jnp.linspace(-2, 2, N)
  X, Y = jnp.meshgrid(x_range, y_range)
  x_grid = jnp.stack([X, Y], axis=-1)
  x_grid = x_grid.reshape(-1, 2)

  t = jnp.array(1.0)
  s = eqx.filter_vmap(net.energy, in_axes=(None, 0))(t, x_grid)
  S = s.reshape(X.shape)
  fig, ax = plt.subplots(1, 1)
  ax.contourf(X, Y, jnp.exp(S))
  plt.show()
  import pdb; pdb.set_trace()
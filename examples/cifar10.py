import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable, Iterator
from jaxtyping import Array, PRNGKeyArray
from generax.trainer import Trainer
import generax.util.misc as misc
import matplotlib.pyplot as plt
import generax.util as util
import equinox as eqx

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

def get_dataset_iter():

  training_data = datasets.CIFAR10(
      root="data",
      train=True,
      download=True,
      transform=ToTensor()
  )

  random_sampler = RandomSampler(training_data, replacement=True)
  train_dataloader = DataLoader(training_data, batch_size=256, sampler=random_sampler, drop_last=True)

  def get_train_ds() -> Iterator[Mapping[str, Array]]:
    train_iter = iter(train_dataloader)
    while True:
      for batch in train_dataloader:
        images, labels = batch
        x = images.numpy().transpose(0, 2, 3, 1)
        yield dict(x=x)

  train_ds = get_train_ds()
  return train_ds

if __name__ == '__main__':
  from debug import *
  from generax.nn.unet import TimeDependentUNet
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
  net = TimeDependentUNet(input_shape=x_shape,
                          dim=64,
                          dim_mults=[1, 2, 2, 4],
                          resnet_block_groups=8,
                          attn_heads=4,
                          attn_dim_head=32,
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
  trainer = Trainer(checkpoint_path='tmp/flow/flow_matching')
  flow = trainer.train(model=flow,
                      objective=loss,
                      evaluate_model=lambda x: x,
                      optimizer=optimizer,
                      num_steps=25000,
                      double_batch=10,
                      data_iterator=train_ds,
                      checkpoint_every=5000,
                      test_every=-1,
                      retrain=False)

  # Pull samples from the model
  keys = random.split(key, 8)
  samples = eqx.filter_vmap(flow.sample)(keys)

  fig, axes = plt.subplots(1, 8)
  for i, ax in enumerate(axes):
    ax.imshow(samples[i])

  # Save the plot
  plt.savefig('tmp/flow/flow_matching/samples.png')
  import pdb; pdb.set_trace()
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

def get_dataset_iter(dtype=jnp.bfloat16):

  training_data = datasets.MNIST(
      root="data",
      train=True,
      download=True,
      transform=ToTensor()
  )

  random_sampler = RandomSampler(training_data, replacement=True)
  train_dataloader = DataLoader(training_data, batch_size=512, sampler=random_sampler, drop_last=True)

  def get_train_ds() -> Iterator[Mapping[str, Array]]:
    train_iter = iter(train_dataloader)
    while True:
      for batch in train_dataloader:
        images, labels = batch
        x = images.numpy().transpose(0, 2, 3, 1).astype(dtype)
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

  dtype = jnp.float32

  train_ds = get_dataset_iter(dtype=dtype)
  data = next(train_ds)
  x = data['x']
  x_shape = x.shape[1:]
  key = random.PRNGKey(0)

  # Construct the neural network that learn the score
  net = TimeDependentUNet(input_shape=x_shape,
                          dim=128,
                          dim_mults=[1, 2, 4],
                          resnet_block_groups=8,
                          attn_heads=4,
                          attn_dim_head=32,
                          key=key)
  flow = ContinuousNormalizingFlow(input_shape=x_shape,
                                   net=net,
                                   key=key,
                                   controller_atol=1e-5,
                                   controller_rtol=1e-5)

  # Change the data type of the parameters
  params, static = eqx.partition(flow, eqx.is_inexact_array)
  params = jax.tree_util.tree_map(lambda x: x.astype(dtype), params)
  flow = eqx.combine(params, static)

  # Count the number of parameters in the flow
  params, _ = eqx.partition(flow, eqx.is_inexact_array)
  num_params = sum(jax.tree_map(lambda x: util.list_prod(x.shape), jax.tree_util.tree_leaves(params)))

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
    t = random.uniform(k2, shape=(x1.shape[0],), dtype=x1.dtype)

    # Resample from the coupling
    x_shape = x1.shape
    x0_flat, x1_flat = map(lambda x: x.reshape((x.shape[0], -1)), (x0, x1))
    coupling = OTTCoupling(x0_flat, x1_flat)
    x0_flat = coupling.sample_x0_given_x1(k1)
    x0 = x0_flat.reshape(x_shape)
    x0 = x0.astype(x1.dtype)

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

  print(f'Number of parameters: {num_params}')

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
  chain.append(optax.adamw(3e-4))
  chain.append(optax.scale_by_schedule(schedule))
  optimizer = optax.chain(*chain)

  # Create the trainer and optimize
  trainer = Trainer(checkpoint_path='tmp/mnist')
  flow = trainer.train(model=flow,
                      objective=loss,
                      evaluate_model=lambda x: x,
                      optimizer=optimizer,
                      num_steps=int(1e6),
                      double_batch=100,
                      data_iterator=train_ds,
                      checkpoint_every=5000,
                      test_every=-1,
                      retrain=False,
                      just_load=True)

  # Pull samples from the model
  keys = random.split(key, 64)
  samples = eqx.filter_vmap(flow.sample)(keys)

  n_rows, n_cols = 8, 8
  size = 4
  fig, axes = plt.subplots(n_rows, n_cols, figsize=(size*n_cols, size*n_rows))
  ax_iter = iter(axes.ravel())
  for i in range(64):
    ax = next(ax_iter)
    ax.imshow(samples[i])
    ax.set_axis_off()

  # Save the plot
  plt.savefig('tmp/mnist/samples.png')
  # import pdb; pdb.set_trace()
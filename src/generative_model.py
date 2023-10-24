import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable, Iterator
import einops
import equinox as eqx
from abc import ABC, abstractmethod
import diffrax
from jaxtyping import Array, PRNGKeyArray
from src.flow.base import BijectiveTransform, Sequential
from src.distributions import ProbabilityDistribution
from src.training.trainer import Trainer
import os
import importlib
import optax

MODELS = dict(flow=['RealNVP',
                    'GLOW',
                    'FFJORD',
                    'NeuralSpline',
                    'ResidualFlow',
                    'ConvexPotentialFlow',
                    'Flow++'],
              ode=['CondOT',
                   'VP',
                   'VE',
                   'Multisample'],
              sde=['VP',
                   'VE'])
TRAINING_ALGORITHMS = dict(flow=['max_likelihood',
                                 'score_matching',
                                 'conjugate_gradient_max_likelihood'],
                           ode=['max_likelihood',
                                'score_matching',
                                'flow_matching'],
                           sde=['score_matching'],)

def GenerativeModel(kind: str,
                    training_algorithm: str,
                    checkpoint_path: str,
                    retrain: bool = False,
                    optimizer: optax.GradientTransformation = optax.adamw(3e-4),
                    *,
                    x: Array,
                    y: Optional[Array] = None,
                    key: PRNGKeyArray,
                    **kwargs) -> ProbabilityDistribution:
  """Construct a generative model.  This will return
  a `ProbabilityDistribution` subclass that can be used
  for sampling and evaluating log probabilities.  Furthermore,
  this object will have a `fit` method that can be used
  to fit the model to data.

  **Arguments**:

  - `kind`: The kind of generative model to construct.  This should
            start with the kind of model that we want and then
            have the specific kind of model like `flow.RealNVP`.
  - `training_algorithm`: The training algorithm to use.
  - `checkpoint_path`: The path to the checkpoint directory
  - `retrain`: Whether to retrain from the latest checkpoint

  """
  model_family, model_name = kind.split('.')

  # Check the model family
  if model_family not in MODELS:
    raise ValueError(f'Unknown model family {model_family}.  Expected one of {MODELS.keys()}')

  # Check the model name
  if model_name not in MODELS[model_family]:
    raise ValueError(f'Unknown model name {model_name}.  Expected one of {MODELS[model_family]}')

  # Check the training algorithm
  if training_algorithm not in TRAINING_ALGORITHMS[model_family]:
    raise ValueError(f'Unknown training algorithm {training_algorithm}.  Expected one of {TRAINING_ALGORITHMS[model_family]}')

  # Retrieve the model and objective function
  model_module = importlib.import_module(f'src.{model_family}.models')
  model_type = getattr(model_module, model_name)

  # Construct the model
  model = model_type(x=x,
                     y=y,
                     key=key,
                     **kwargs)

  # Retrieve the objective function
  import src.training as training
  objective = getattr(training, training_algorithm)

  # Build the trainer and add it to the model
  assert isinstance(model, _TrainerMixin)
  trainer = Trainer(model=model,
                    objective=objective,
                    evaluate_model=lambda x: x,
                    optimizer=optimizer,
                    checkpoint_path=checkpoint_path,
                    retrain=retrain)
  model.trainer = trainer
  return model

################################################################################################################

if __name__ == '__main__':
  from debug import *
  import matplotlib.pyplot as plt
  from src.flow.models import RealNVP, NeuralSpline
  from src.training.max_likelihood.ml import max_likelihood

  jax.config.update('jax_traceback_filtering', 'off')

  # Get the dataset
  from sklearn.datasets import make_moons, make_swiss_roll
  data, y = make_moons(n_samples=100000, noise=0.07)
  data = data - data.mean(axis=0)
  data = data/data.std(axis=0)
  key = random.PRNGKey(0)

  def get_train_ds(key: PRNGKeyArray,
                   batch_size: int = 64) -> Iterator[Mapping[str, Array]]:
    total_choices = jnp.arange(data.shape[0])
    closed_over_data = data  # In case we change the variable "data"
    while True:
      key, _ = random.split(key, 2)
      idx = random.choice(key,
                          total_choices,
                          shape=(batch_size,),
                          replace=True)
      yield dict(x=closed_over_data[idx])

  train_ds = get_train_ds(key)

  x = data[:10]

  # P = RealNVP(x=x,
  #             y=None,
  #             key=key,
  #             n_layers=10,
  #             n_res_blocks=8,
  #             hidden_size=32,
  #             working_size=16)

  P = NeuralSpline(x=x,
                   y=None,
                   key=key,
                   n_layers=3,
                   n_res_blocks=4,
                   hidden_size=32,
                   working_size=16)

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

  trainer = Trainer(checkpoint_path='tmp/RealNVP')

  model = trainer.train(model=P,
                        objective=max_likelihood,
                        evaluate_model=lambda x: x,
                        optimizer=optimizer,
                        num_steps=10000,
                        # num_steps=1000,
                        data_iterator=train_ds,
                        # double_batch=-1,
                        double_batch=1000,
                        checkpoint_every=1000,
                        test_every=1000,
                        retrain=True)

  # Pull some samples and plot
  samples = model.sample(key, 1000)
  plt.scatter(*samples.T)
  plt.show()

  z = eqx.filter_vmap(model.to_base_space)(samples)
  x_reconstr = eqx.filter_vmap(model.to_data_space)(z)

  @eqx.filter_vmap
  def jacobian(x):
    return jax.jacobian(model.to_base_space)(x)

  G = jacobian(samples)
  log_pz = eqx.filter_vmap(model.prior.log_prob)(z)

  import pdb; pdb.set_trace()

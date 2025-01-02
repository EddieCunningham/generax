import tqdm
import jax.numpy as jnp
import jax
import jax.random as random
from functools import partial
from typing import Optional, Mapping, Tuple, List, Sequence, Union, Any, Callable, Dict, Iterator
import optax
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray, PyTree
import tqdm
import jax.tree_util as jtu
import generax.util as util
import os

__all__ = ['TrainingState',
            'Checkpointer',
            'Trainer',
            'default_optimizer']

class TrainingState(eqx.Module):

  i: float
  key: PRNGKeyArray
  model: eqx.Module
  opt_state: optax.OptState

  def __init__(self,
               i: float, # float so that it is not treated as static
               key: PRNGKeyArray,
               model: eqx.Module,
               opt_state: optax.OptState):
    self.i = i
    self.key = key
    self.model = model
    self.opt_state = opt_state

################################################################################################################

class Checkpointer(eqx.Module):

  save_path: str
  model_folder: str

  def __init__(self,
               save_path: str):
    self.save_path = save_path
    self.model_folder = os.path.join(save_path, 'models')
    util.ensure_path_exists(self.model_folder)

  def model_exists(self) -> bool:
    return os.path.exists(self.saved_model_path)

  @property
  def saved_model_path(self):
    return os.path.join(self.model_folder, 'saved_model.pickle')

  def save_eqx_module(self,
                      model: eqx.Module):
    eqx.tree_serialise_leaves(self.saved_model_path, model)

  def load_eqx_module(self,
                      model_example: eqx.Module):
    return eqx.tree_deserialise_leaves(self.saved_model_path, model_example)

################################################################################################################

class AuxiliaryTracker():
  """This class is used to track auxiliary information during training.  It is
  used by the `Trainer` class to keep track of auxiliary information during
  training.  This is useful for things like tracking the training loss, which
  is not part of the model itself.
  """
  history: Dict[str,Array]
  save_path: str

  def __init__(self,
               save_path: str):
    self.history = None
    self.save_path = save_path

  @property
  def aux_folder(self):
    aux_folder = os.path.join(self.save_path, 'aux')
    util.ensure_path_exists(aux_folder)
    return aux_folder

  @property
  def aux_history_path(self):
    return os.path.join(self.aux_folder, 'aux_history.csv')

  @property
  def aux_plots_folder(self):
    plots_folder = os.path.join(self.aux_folder, 'plots')
    util.ensure_path_exists(plots_folder)
    return plots_folder

  def update(self, aux: Dict[str,Array], double_batched: bool = False):

    def to_list(x):
      if double_batched:
        return x.tolist()
      return [float(x)]

    aux_list = jtu.tree_map(to_list, aux)

    # Concatenate the new aux to the history
    if self.history is None:
      self.history = aux_list
    else:
      for key in self.history.keys():
        self.history[key] += aux_list[key]

  def checkpoint(self):
    util.dict_to_csv(self.history, self.aux_history_path)

  def restore(self):
    self.history = util.csv_to_dict(self.aux_history_path)

  def create_plots(self):
    import matplotlib.pyplot as plt

    # Create plots of the aux history
    for key, val_list in self.history.items():
      assert isinstance(val_list, list)
      val = jnp.array(val_list)
      assert val.ndim == 1
      T = val.shape[0]

      # Simple moving average of val
      window_size = 1000
      cumsum = jnp.cumsum(jnp.pad(val, (window_size, 0), constant_values=val[0]))
      sma = (cumsum[window_size:] - cumsum[:-window_size])/window_size

      title = f'{key} vs. training step'
      save_path = os.path.join(self.aux_plots_folder, f'{key}.png')

      # Save off a few different kinds of plots
      n_rows, n_cols = 2, 2
      size = 4
      fig, axes = plt.subplots(n_rows, n_cols, figsize=(size*n_cols, size*n_rows))

      # First plot is the entire plot
      axes[0,0].plot(val, color='blue')
      axes[0,0].plot(sma, color='red')
      axes[0,0].set_title(title)
      axes[0,0].set_xlabel('Training step')
      axes[0,0].set_ylabel(key)

      # Second one is log scale
      axes[0,1].plot(val, color='blue')
      axes[0,1].plot(sma, color='red')
      axes[0,1].set_title(title)
      axes[0,1].set_xlabel('Training step')
      axes[0,1].set_ylabel(key)
      axes[0,1].set_yscale('log')

      # Third one is the N most recent steps
      N = 5000
      t = jnp.arange(max(0,T-N), T)
      axes[1,0].plot(t, val[-N:], color='blue')
      axes[1,0].plot(t, sma[-N:], color='red')
      axes[1,0].set_title(title)
      axes[1,0].set_xlabel('Training step')
      axes[1,0].set_ylabel(key)

      # Fourth one is the N most recent steps in log scale
      axes[1,1].plot(t, val[-N:], color='blue')
      axes[1,1].plot(t, sma[-N:], color='red')
      axes[1,1].set_title(title)
      axes[1,1].set_xlabel('Training step')
      axes[1,1].set_ylabel(key)
      axes[1,1].set_yscale('log')

      fig.savefig(save_path)
      plt.close(fig)

################################################################################################################

class Trainer(eqx.Module):
  """Class that will monitor training and handle checkpointing.

  **Attributes**:

  - `checkpointer`: Object that saves checkpoints of the model
  """

  checkpointer: Checkpointer
  aux_history: AuxiliaryTracker

  def __init__(self,
               checkpoint_path: str):

    self.checkpointer = Checkpointer(checkpoint_path)
    self.aux_history = AuxiliaryTracker(checkpoint_path)

  def train_step(self,
                 objective: Callable,
                 optimizer: optax.GradientTransformation,
                 train_state: TrainingState,
                 data: Dict[str,Array],
                 grad_scan_batch_size: Optional[int] = None) -> Tuple[TrainingState, Mapping[str, Any]]:
    i, model, opt_state = train_state.i, train_state.model, train_state.opt_state
    train_key, next_key = random.split(train_state.key)

    if grad_scan_batch_size:

      # Accumulate the gradients over multiple batches of data
      def get_grads(data):
        return eqx.filter_value_and_grad(objective, has_aux=True)(model, data, train_key)

      out = jax.lax.map(get_grads, data)
      (obj, aux), grads = jtu.tree_map(lambda x: jnp.mean(x, axis=0), out)
    else:

      # Compute the gradients of the objective
      (obj, aux), grads = eqx.filter_value_and_grad(objective, has_aux=True)(model, data, train_key)

    aux['objective'] = obj

    # Get the norm of the gradient
    # grad_sq_norms = jtu.tree_map(jnp.mean, grads)
    grad_sq_norms = jtu.tree_map(lambda x: jnp.mean(x**2), grads)
    grad_sq_norms_list = jtu.tree_flatten(grad_sq_norms)[0]
    aux['grad_norm'] = sum(grad_sq_norms_list)/len(grad_sq_norms_list)

    # We need aux to contain scalars in order to write it correctly.
    aux = jtu.tree_map(jnp.mean, aux)

    # Update the model
    updates, new_opt_state = optimizer.update(grads, opt_state, model)
    new_model = eqx.apply_updates(model, updates)

    # Package the updated training state
    updated_train_state = TrainingState(i=i+1,
                                        key=next_key,
                                        model=new_model,
                                        opt_state=new_opt_state)
    return updated_train_state, aux

  def train(self,
            model: eqx.Module,
            objective: Callable,
            evaluate_model: Callable,
            optimizer: optax.GradientTransformation,
            num_steps: int,
            data_iterator: Iterator,
            double_batch: int = -1,
            checkpoint_every: int = 1000,
            test_every: int = 1000,
            retrain: bool = False,
            just_load: bool = False,
            stop_when: Optional[Callable[[Dict[str,Array]], bool]] = None,
            grad_scan_batch_size: Optional[int] = None):
    """Train the model.  This will load the model if the most
    recent checkpoint exists has completed training.

    **Arguments**:

    - `model`: The model to train
    - `objective`: The objective function to optimize
    - `evaluate_model`: A function that takes in the model and evaluates it
                        on a test set
    - `optimizer`: The optimizer to use
    - `num_steps`: The number of training steps to take
    - `data_iterator`: An iterator that yields batches of data
    - `double_batch`: If `double_batch > 0`, then we will take `double_batch` batches
                      of data at a time and train over them in a fast `jax.lax.scan` loop.
    - `checkpoint_every`: How often to checkpoint the model
    - `test_every`: How often to evaluate the model
    - `retrain`: Whether to force retraining from scratch
    - `just_load`: Whether to just load the most recent checkpoint and return
    """
    key0 = random.PRNGKey(0)

    if grad_scan_batch_size is not None:
      assert double_batch == -1, 'Does not support double_batch with grad_scan_batch_size'

    # Load the most recent checkpoint
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    train_state = TrainingState(jnp.array(0.0), key0, model, opt_state)

    if retrain == False:
      if self.checkpointer.model_exists() == False:
        retrain = True

    # Load the most recent checkpoint
    if retrain == False:
      train_state = self.restore(train_state)

    if just_load:
      return train_state.model

    # Fill in the training step with the objective and optimizer
    train_step = eqx.Partial(self.train_step, objective, optimizer, grad_scan_batch_size=grad_scan_batch_size)

    if double_batch == -1:
      # JIT the training update here
      train_step = eqx.filter_jit(train_step)
    else:
      # We can only pass in parameters dynamically to the scan loop, so we
      # need to extract the static values here (because they won't change)
      # and combine later inside the scan loop
      _, static = eqx.partition(train_state, eqx.is_array)

      # Construct the scan loop that we'll use to process batches of data
      def step(params, data):
        train_state = eqx.combine(params, static)
        new_train_state, aux = train_step(train_state, data)
        new_params, _ = eqx.partition(new_train_state, eqx.is_array)
        return new_params, aux
      scan_step = partial(jax.lax.scan, step)
      scan_step = jax.jit(scan_step)

    # Construct the progress bar
    start = int(train_state.i) if retrain == False else 0
    if double_batch <= 0:
      pbar = tqdm.tqdm(jnp.arange(start, num_steps),
                       initial=start,
                       total=num_steps - start)
    else:
      pbar = tqdm.tqdm(jnp.arange(start, num_steps, double_batch),
                       initial=start,
                       total=num_steps - start)

    # Training loop
    for i in pbar:

      # Take a training step
      if double_batch == -1:
        if grad_scan_batch_size is not None:
          data = util.extract_multiple_batches_from_iterator(data_iterator, grad_scan_batch_size)
        else:
          data = next(data_iterator)
        train_state, aux = train_step(train_state, data)
        pbar.update(1)
        self.aux_history.update(aux)
      else:
        data = util.extract_multiple_batches_from_iterator(data_iterator, double_batch)
        params, static = eqx.partition(train_state, eqx.is_array)
        params, aux = scan_step(params, data)
        train_state = eqx.combine(params, static)
        pbar.update(double_batch)
        self.aux_history.update(aux, double_batched=True)

      # Update the progress bar
      description = ', '.join([f'{k}={float(v.mean()):.4f}' for k, v in aux.items()])
      pbar.set_description(description)

      # Checkpoint the model
      if (i and (i%checkpoint_every == 0)):
        self.checkpoint(train_state)
        print('Checkpointed model')

      # Evaluate the model
      if (i%test_every == 0) or (i == num_steps - 1):
        evaluate_model(train_state.model)

      # Stopping condition
      if stop_when is not None:
        if stop_when(i, aux):
          print('Stopping training because of condition')
          break

    # Final checkpoint
    self.checkpoint(train_state)
    print('Checkpointed model')
    return train_state.model

  def checkpoint(self, train_state: TrainingState):
    # Save off the model
    self.checkpointer.save_eqx_module(train_state)

    # Save off the auxiliary history
    self.aux_history.checkpoint()
    self.aux_history.create_plots()

  def restore(self, train_state: TrainingState) -> TrainingState:
    # Load the model
    train_state = self.checkpointer.load_eqx_module(train_state)
    print(f'Restored train_state {self.checkpointer.saved_model_path}')

    # Load the auxiliary history
    self.aux_history.restore()
    return train_state

################################################################################################################

def default_optimizer(lr=1e-3,
                      clip_norm=15.0,
                      warmup=1000,
                      decay_steps=3e5,
                      end_value=0.1,
                      cosine_exponent=1.0,
                      weight_decay_hyper=10.0,
                      dataset_size=None,
                      batch_size=None) -> optax.GradientTransformation:
  """
  Gradient clipping, AdamW, and cosine decay with warmup.
  """
  schedule = optax.warmup_cosine_decay_schedule(init_value=0.0,
                                                peak_value=1.0,
                                                warmup_steps=warmup,
                                                decay_steps=decay_steps,
                                                end_value=end_value,
                                                exponent=cosine_exponent)
  chain = []
  chain.append(optax.clip_by_global_norm(clip_norm))

  # https://arxiv.org/pdf/2405.13698v1
  if dataset_size is not None:
    weight_decay = weight_decay_hyper/(lr*dataset_size/batch_size)
  else:
    weight_decay = 0.0001

  chain.append(optax.adamw(lr, weight_decay=weight_decay))
  chain.append(optax.scale_by_schedule(schedule))
  optimizer = optax.chain(*chain)
  return optimizer

################################################################################################################

if __name__ == '__main__':
  from debug import *
  import matplotlib.pyplot as plt
  import generax as gx
  import generax.util as util

  key = random.PRNGKey(0)

  # Get the dataset
  from sklearn.datasets import make_moons, make_swiss_roll
  data, y = make_moons(n_samples=100000, noise=0.1)
  data = data - data.mean(axis=0)
  data = data/data.std(axis=0)
  p1 = gx.EmpiricalDistribution(data)
  train_ds = p1.train_iterator(key, batch_size=64)
  x = next(train_ds)['x']
  x_shape = x.shape[1:]

  # Construct the flow
  flow = gx.NeuralSpline(input_shape=x_shape,
                          key=key,
                          n_flow_layers=2,
                          n_blocks=1,
                          hidden_size=4,
                          working_size=2,
                          n_spline_knots=2)

  # Construct the loss function
  def loss(flow, data, key):
    x = data['x']
    log_px = eqx.filter_vmap(flow.log_prob)(x)
    objective = -log_px.mean()

    aux = dict(log_px=log_px)
    return objective, aux

  # Create the trainer and optimize
  trainer = Trainer(checkpoint_path='tmp/trainer_test')
  flow = trainer.train(model=flow,
                       objective=loss,
                       evaluate_model=lambda x: x,
                       optimizer=default_optimizer(lr=1e-3),
                       num_steps=100,
                       double_batch=10,
                       data_iterator=train_ds,
                       checkpoint_every=20,
                       test_every=-1,
                       retrain=True,
                       just_load=False)

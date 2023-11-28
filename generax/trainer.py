import tqdm
import jax.numpy as jnp
import jax
import jax.random as random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable, Dict, Iterator
import optax
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray
import tqdm
import generax.util.misc as misc
import os

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
    misc.ensure_path_exists(self.model_folder)

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

class Trainer(eqx.Module):
  """Class that will monitor training and handle checkpointing.

  **Attributes**:

  - `checkpointer`: Object that saves checkpoints of the model
  """

  checkpointer: Checkpointer
  _aux_history: list

  def __init__(self,
               checkpoint_path: str):

    self.checkpointer = Checkpointer(checkpoint_path)
    self._aux_history = []

  @property
  def aux_history(self):
    return jax.tree_util.tree_map(lambda *xs: jnp.array(xs), *self._aux_history)

  def train_step(self,
                 objective: Callable,
                 optimizer: optax.GradientTransformation,
                 train_state: TrainingState,
                 data: Dict[str,Array]) -> Tuple[TrainingState, Mapping[str, Any]]:
    i, model, opt_state = train_state.i, train_state.model, train_state.opt_state
    train_key, next_key = random.split(train_state.key)

    # Compute the gradients of the objective
    (obj, aux), grads = eqx.filter_value_and_grad(objective, has_aux=True)(model, data, train_key)
    aux['objective'] = obj

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
            just_load: bool = False):
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

    # Load the most recent checkpoint
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    train_state = TrainingState(jnp.array(0.0), key0, model, opt_state)

    # Load the most recent checkpoint
    if retrain == False:
      train_state = self.restore(train_state)

    if just_load:
      return train_state.model

    # Fill in the training step with the objective and optimizer
    train_step = eqx.Partial(self.train_step, objective, optimizer)

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
      pbar = tqdm.tqdm(jnp.arange(start, num_steps), total=num_steps - start)
    else:
      pbar = tqdm.tqdm(jnp.arange(start, num_steps, double_batch), total=num_steps - start)

    # Training loop
    for i in pbar:

      # Take a training step
      if double_batch == -1:
        data = next(data_iterator)
        train_state, aux = train_step(train_state, data)
        pbar.update(1)
      else:
        data = misc.extract_multiple_batches_from_iterator(data_iterator, double_batch)
        params, static = eqx.partition(train_state, eqx.is_array)
        params, aux = scan_step(params, data)
        train_state = eqx.combine(params, static)
        pbar.update(double_batch)
      self._aux_history.append(aux)

      # Update the progress bar
      description = ', '.join([f'{k}={v.mean():.4f}' for k, v in aux.items()])
      pbar.set_description(description)

      # Checkpoint the model
      if (i and (i%checkpoint_every == 0)):
        self.checkpoint(train_state)
        print('Checkpointed model')

      # Evaluate the model
      if (i%test_every == 0) or (i == num_steps - 1):
        evaluate_model(train_state.model)

    # Final checkpoint
    self.checkpoint(train_state)
    print('Checkpointed model')
    return train_state.model

  def checkpoint(self, train_state: TrainingState):
    self.checkpointer.save_eqx_module(train_state)

  def restore(self, train_state: TrainingState) -> TrainingState:
    train_state = self.checkpointer.load_eqx_module(train_state)
    print(f'Restored train_state {self.checkpointer.saved_model_path}')
    return train_state

################################################################################################################

# generax
generax provides implementations of different kinds of generative models.  The library is built on top of [Equinox](https://github.com/patrick-kidger/equinox) which removes the need to worry about keeping track of model parameters.  For example, the following code snippet shows how to create a neural spline flow and sample from it.
```python
key = random.PRNGKey(0) # JAX random key
x = ... # some data

# Create a flow model
model = NeuralSpline(input_shape=x.shape[1:],
                     n_flow_layers=3,
                     n_blocks=4,
                     hidden_size=32,
                     working_size=16,
                     n_spline_knots=8,
                     key=key)

# Data dependent initialization
model = model.data_dependent_init(x, key=key)

# Sample from the model
samples = model.sample(key, n_samples=1000)

# Compute the log probability of data
log_prob = model.log_prob(x)
```

# Installation
generax is available on pip:
```bash
pip install generax
```

# Roadmap
### Implemented
- Normalizing flows
- Continuous normalizing flows
- Diffusion models

And these models can be trained using a variety of methods including:
- Maximum likelihood
- Score matching
- Flow matching
- Variational inference

# Training
Generax provides an easy interface to train these models:
```python
trainer = Trainer(checkpoint_path='tmp/RealNVP')

model = trainer.train(model=model,              # Generax model
                      objective=max_likelihood, # Objective function
                      evaluate_model=tester,    # Testing function
                      optimizer=optimizer,      # Optax optimizer
                      num_steps=10000,          # Number of training steps
                      data_iterator=train_ds,   # Training data iterator
                      double_batch=1000,        # Train these many batches in a scan loop
                      checkpoint_every=1000,    # Checkpoint interval
                      test_every=1000,          # Test interval
                      retrain=True)             # Retrain from checkpoint
```
See the tutorial for an example.
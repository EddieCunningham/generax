import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
from generax.nn.pac import PAC
import generax.util as util

def test_pac_shapes():
  key = random.PRNGKey(0)
  k1, k2, k3 = random.split(key, 3)

  H, W, C_in = 16, 16, 3
  C_out = 8
  feature_dim = 4
  kernel_size = 5

  layer = PAC(in_channels=C_in,
              out_channels=C_out,
              feature_dim=feature_dim,
              kernel_size=kernel_size,
              key=k1)

  x = random.normal(k2, shape=(H, W, C_in))
  f = random.normal(k3, shape=(H, W, feature_dim))

  out = layer(x, f)

  assert out.shape == (H, W, C_out)

def test_pac_constant_guidance():
  """If guidance features are constant, PAC should behave like standard convolution."""
  key = random.PRNGKey(0)
  k1, k2, k3 = random.split(key, 3)

  H, W, C_in = 8, 8, 2
  C_out = 4
  feature_dim = 3
  kernel_size = 3

  layer = PAC(in_channels=C_in,
              out_channels=C_out,
              feature_dim=feature_dim,
              kernel_size=kernel_size,
              key=k1)

  x = random.normal(k2, shape=(H, W, C_in))
  # Constant guidance features
  f = jnp.ones((H, W, feature_dim))

  out = layer(x, f)

  # Standard convolution with weights t * W
  t = jnp.exp(layer.log_t)
  W_effective = layer.W * t[None, None, None, :]

  # util.conv expects (H, W, C_in, C_out) for weights?
  # Wait, let's check util.conv definition.
  # def conv(w, x, stride=1, padding='SAME'):
  # ... jax.lax.conv_general_dilated(x, w, ..., dimension_numbers=("NHWC", "HWIO", "NHWC"))
  # HWIO means (H, W, C_in, C_out). Our layer.W is already (K, K, C_in, C_out).

  expected_out = util.conv(W_effective, x, padding='SAME')

  assert jnp.allclose(out, expected_out, atol=1e-5)

def test_pac_vmap():
  """Test that PAC can be vmapped."""
  key = random.PRNGKey(0)
  k1, k2, k3 = random.split(key, 3)

  batch_size = 4
  H, W, C_in = 10, 10, 3
  C_out = 5
  feature_dim = 2

  layer = PAC(in_channels=C_in,
              out_channels=C_out,
              feature_dim=feature_dim,
              kernel_size=3,
              key=k1)

  x = random.normal(k2, shape=(batch_size, H, W, C_in))
  f = random.normal(k3, shape=(batch_size, H, W, feature_dim))

  # vmap over batch dimension
  vmapped_layer = jax.vmap(layer, in_axes=(0, 0))
  out = vmapped_layer(x, f)

  assert out.shape == (batch_size, H, W, C_out)

if __name__ == "__main__":
  test_pac_shapes()
  test_pac_constant_guidance()
  test_pac_vmap()
  print("All tests passed!")

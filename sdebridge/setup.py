import math
from collections import namedtuple
from functools import partial
from typing import Any, Callable, Dict, Iterable, Sequence, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from ml_collections import ConfigDict
import tensorflow as tf
from einops import rearrange, repeat


tf.config.experimental.set_visible_devices([], "GPU")
# jax.config.update("jax_platform_name", "cpu")

GDRK = jax.random.PRNGKey(0)

from collections import namedtuple
from functools import partial
from typing import Any, Callable, Dict, Iterable, Tuple

import jax
import jax.numpy as jnp
from jax import jit, pmap, random, vmap
from jax.typing import ArrayLike
from ml_collections import ConfigDict

GLOBAL_RNG_KEY = random.PRNGKey(0)

"""
Top–level package for the coil design examples.

Importing this module has no side effects.  See the `__main__` module
for the command–line entry point.
"""

from .booz_driver import run_booz_example  # noqa: F401
from .nearaxis_driver import run_nearaxis_example  # noqa: F401
from .toml_config import load_config  # noqa: F401
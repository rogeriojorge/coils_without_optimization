"""Load example configurations from a TOML file.

The command–line entry point reads a TOML file containing a
configuration for either the Boozer or near–axis example and returns
an instance of the corresponding dataclass.  Unknown keys are
ignored.  The special key ``example`` determines which dataclass is
instantiated.  If no example is provided, ``"booz"`` is assumed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Type, Union

try:
    import tomllib as toml  # Python ≥3.11
except ImportError:  # pragma: no cover
    import tomli as toml  # type: ignore

from .booz_config import BoozConfig
from .nearaxis_config import NearAxisConfig


def _filter_fields(data: Dict[str, Any], cls: Type) -> Dict[str, Any]:
    """Return a dict containing only keys that appear in the dataclass."""
    return {k: v for k, v in data.items() if k in cls.__dataclass_fields__}


def load_config(path: str | Path) -> Union[BoozConfig, NearAxisConfig]:
    """Load a configuration dataclass from a TOML file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the TOML configuration file.

    Returns
    -------
    BoozConfig or NearAxisConfig
        An instance of the appropriate configuration dataclass.

    Raises
    ------
    ValueError
        If the ``example`` key is missing or unsupported.
    """
    path = Path(path)
    with open(path, "rb") as f:
        data: Dict[str, Any] = toml.load(f)
    # Determine which example to load
    example = data.get("example", "booz").lower()
    # Pop ``example`` so it doesn't become part of the dataclass
    data = {k: v for k, v in data.items() if k != "example"}
    if example in {"booz", "booz_xform", "boozer"}:
        filtered = _filter_fields(data, BoozConfig)
        return BoozConfig(**filtered)
    elif example in {"nearaxis", "near_axis"}:
        filtered = _filter_fields(data, NearAxisConfig)
        return NearAxisConfig(**filtered)
    else:
        raise ValueError(f"Unsupported example type: {example}")
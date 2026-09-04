"""Compatibility shim: import utils now lives in the orts package."""
import warnings as _w
_w.warn("'import utils' is deprecated; use 'from orts import ...' (orts 2.0)", DeprecationWarning, stacklevel=2)
from orts.utils import *  # noqa: F401,F403

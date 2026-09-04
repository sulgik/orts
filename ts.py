"""Compatibility shim: import ts now lives in the orts package."""
import warnings as _w
_w.warn("'import ts' is deprecated; use 'from orts import ...' (orts 2.0)", DeprecationWarning, stacklevel=2)
from orts.ts import *  # noqa: F401,F403

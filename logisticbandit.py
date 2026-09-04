"""Compatibility shim: import logisticbandit now lives in the orts package."""
import warnings as _w
_w.warn("'import logisticbandit' is deprecated; use 'from orts import ...' (orts 2.0)", DeprecationWarning, stacklevel=2)
from orts.logisticbandit import *  # noqa: F401,F403

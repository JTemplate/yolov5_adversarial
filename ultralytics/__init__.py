"""Minimal compatibility surface for this pinned YOLOv5 checkout.

The repository only needs a few utility symbols from ultralytics.  Keeping
these shims local makes the experiment runnable in the existing conda
environment without triggering an automatic network install from utils.general.
"""

__version__ = "8.0.0-local-compat"

"""Training and inference objectives reusable across compatible backbones.

Import each objective explicitly so its optional dependencies stay optional.
"""

from .base import BaseObjective

__all__ = ["BaseObjective"]

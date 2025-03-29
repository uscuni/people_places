import contextlib
from importlib.metadata import PackageNotFoundError, version

from . import gw

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("core")

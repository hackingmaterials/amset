import abc
from pathlib import Path
from typing import Union

from monty.serialization import loadfn
from pkg_resources import resource_filename

from amset.core.data import AmsetData
from amset.util import cast_dict_ndarray

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"

amset_base_style = resource_filename("amset.plot", "amset_base.mplstyle")


class BaseAmsetPlotter(abc.ABC):
    def __init__(self, data: Union[str, Path, AmsetData, dict]):
        if isinstance(data, (str, Path)):
            data = loadfn(data)
        elif isinstance(data, AmsetData):
            data = data.to_dict(include_mesh=True)
        elif isinstance(data, dict):
            data = data
        else:
            raise ValueError("Unrecognised data format")

        self._data = cast_dict_ndarray(data)
        self.spins = list(self.energies.keys())

        self.has_mesh = "energies" in data

    def __getattr__(self, item):
        return self._data[item]

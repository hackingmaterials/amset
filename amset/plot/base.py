import abc
from pathlib import Path
from typing import Union

from monty.serialization import loadfn
from pkg_resources import resource_filename

from amset.core.data import AmsetData
from amset.util import cast_dict_ndarray, load_mesh_data

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"

amset_base_style = resource_filename("amset.plot", "amset_base.mplstyle")


class BaseAmsetPlotter(abc.ABC):
    def __init__(self, data: Union[str, Path, AmsetData, dict]):
        if isinstance(data, (str, Path)):
            mesh_data = Path(str(data).replace("_transport_", "_mesh_"))
            data = loadfn(data)
            if mesh_data.exists():
                # try and load mesh properties also
                mesh_data = load_mesh_data(mesh_data)
                data.update(mesh_data)
        elif isinstance(data, AmsetData):
            data = data.to_dict(include_mesh=True)
        elif isinstance(data, dict):
            data = data
        else:
            raise ValueError("Unrecognised data format")

        if "mesh" in data:
            mesh_data = data.pop("mesh")
            data.update(mesh_data)

        self._data = cast_dict_ndarray(data)
        self.spins = list(self.energies.keys())

        self.has_mesh = "energies" in data

    def __getattr__(self, item):
        if item not in self._data:
            raise RuntimeError(
                "No mesh data available, AMSET must be run with write_mesh=True"
            )
        return self._data[item]

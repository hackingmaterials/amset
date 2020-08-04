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


class BaseMeshPlotter(abc.ABC):
    def __init__(self, mesh_data: Union[str, Path, AmsetData, dict]):
        if isinstance(mesh_data, (str, Path)):
            if "h5" not in mesh_data:
                raise ValueError("mesh.h5 file needed for plot. "
                                 "Run AMSET with write_mesh=True")
            print("loading")
            mesh_data = load_mesh_data(mesh_data)
            print("done")
        elif isinstance(mesh_data, AmsetData):
            mesh_data = mesh_data.to_dict(include_mesh=True)["mesh"]
        elif not isinstance(mesh_data, dict):
            raise ValueError("Unrecognised data format")

        self._data = cast_dict_ndarray(mesh_data)
        self.spins = list(self.energies.keys())

    def __getattr__(self, item):
        return self._data[item]


class BaseTransportPlotter(abc.ABC):
    def __init__(self, data: Union[str, Path, AmsetData, dict]):
        if isinstance(data, (str, Path)):
            # guess the mesh filename
            data = loadfn(data)
        elif isinstance(data, AmsetData):
            data = data.to_dict(include_mesh=True)
        elif isinstance(data, dict):
            data = data
        else:
            raise ValueError("Unrecognised data format")

        self._data = cast_dict_ndarray(data)
        self.spins = list(self.energies.keys())

    def __getattr__(self, item):
        return self._data[item]

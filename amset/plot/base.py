import abc
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from monty.serialization import loadfn
from pkg_resources import resource_filename

from amset import __version__
from amset.core.data import AmsetData
from amset.io import load_mesh
from amset.util import cast_dict_ndarray

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"

amset_base_style = resource_filename("amset.plot", "amset_base.mplstyle")
revtex_style = resource_filename("amset.plot", "revtex.mplstyle")
base_total_color = "#193A4A"


class BaseMeshPlotter(abc.ABC):
    def __init__(self, mesh_data: Union[str, Path, AmsetData, dict]):
        if isinstance(mesh_data, (str, Path)):
            if "h5" not in str(mesh_data):
                raise ValueError(
                    "mesh.h5 file needed for plot. Run AMSET with write_mesh=True"
                )
            mesh_data = load_mesh(mesh_data)
        elif isinstance(mesh_data, AmsetData):
            mesh_data = mesh_data.to_dict(include_mesh=True)["mesh"]
        elif "mesh" in mesh_data:
            mesh_data = mesh_data["mesh"]
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

    def __getattr__(self, item):
        return self._data[item]

    @property
    def has_mobility(self):
        return "mobility" in self._data and self._data["mobility"] is not None

    @property
    def mobility(self):
        return self.get_mobility()

    def get_mobility(self, mechanism="overall"):
        if not self.has_mobility:
            raise IndexError("Mobility does not exist in transport data")
        return self._data["mobility"][mechanism]

    @property
    def mechanisms(self):
        if not self.has_mobility:
            raise ValueError("Mobility must be in transport data to extract mechanisms")

        return tuple(self._data["mobility"].keys())


class BaseMultiTransportPlotter(abc.ABC):
    def __init__(self, data: List[Union[str, Path, AmsetData, dict, list]]):
        if not isinstance(data, (tuple, list)):
            data = [data]

        if isinstance(data, tuple):
            data = list(data)

        if len(data) < 2:
            raise ValueError("More than 2 transport data needed for plotting")

        new_data = []
        for d in data:
            if isinstance(d, (str, Path)):
                # guess the mesh filename
                d = loadfn(d)
            elif isinstance(d, AmsetData):
                d = d.to_dict(include_mesh=True)
            elif isinstance(d, dict):
                d = d
            else:
                raise ValueError("Unrecognised data format")

            new_data.append(cast_dict_ndarray(d))

        temperatures = set(new_data[0]["temperatures"])
        doping = set(new_data[0]["doping"])
        for d in new_data[1:]:
            if set(d["temperatures"]) != temperatures or set(d["doping"]) != doping:
                raise ValueError(
                    "Transport data contain inconsistent doping or temperatures"
                )

        self.temperatures = new_data[0]["temperatures"]
        self.doping = new_data[0]["doping"]
        self.n = len(new_data)
        self._data = new_data

    def __getattr__(self, item):
        return np.array([d[item] for d in self._data])

    @property
    def has_mobility(self):
        return all(["mobility" in d for d in self._data])

    @property
    def mobility(self):
        return self.get_mobility()

    def get_mobility(self, mechanism="overall"):
        if not self.has_mobility:
            raise IndexError("Mobility does not exist in all transport data")
        return np.array([d["mobility"][mechanism] for d in self._data])


@dataclass
class PlotData:
    """A class to represent basic plot data.

    Args:
        x_property: The x-axis property.
        y_property: The y-axis property.
        x_label: The x-axis label containing the unit.
        y_label: The y-axis label containing the unit.
        labels: The labels for each line on the plot. A list of strings of length m.
        x: The x-axis values as a numpy array with shape (n).
        y: The y-axis values with the shape (m, n).
    """

    x_property: str
    y_property: str
    x_label: str
    y_label: str
    labels: List[str]
    x: np.ndarray
    y: np.ndarray

    @property
    def comment(self) -> str:
        """Comment containing the labels, properties, and units."""
        comment = f"File generated using amset {__version__}"
        comment += f"\nx-axis: {self.x_label}"
        comment += f"\ny-axis: {self.y_label}"
        comment += f"\n{self.x_property.replace(' ', '_')}"

        for label in self.labels:
            comment += f"  {label.replace(' ', '')}"

        return comment

    @property
    def data(self) -> np.ndarray:
        """Data (containing x and y values) as a numpy array."""
        return np.vstack([self.x, self.y])

    def to_file(self, filename: [str, Path], fmt: str = "%.8e"):
        """Write plot data to file.

        Args:
            filename: The file name.
            fmt: Number format specifier.
        """
        np.savetxt(filename, self.data.T, fmt=fmt, header=self.comment)


def write_plot_data(
    plot_data: List[PlotData],
    prefix: Optional[Union[str, Path]] = None,
    directory: Optional[Union[str, Path]] = None,
) -> List[Path]:
    """Write plot data to files.

    Args:
        plot_data: A list of PlotData objects. The y_property attribute is used as the
            filename.
        prefix: A prefix for the data files.
        directory: The directory in which to store the data files.

    Returns:
        A list of path objects for each file.
    """
    if directory is None:
        directory = ""
    directory = Path(directory)

    if prefix is None:
        prefix = ""
    prefix = str(prefix)
    if len(prefix) > 0 and prefix[-1] != "_":
        prefix += "_"

    filenames = []
    for data in plot_data:
        propname = data.y_property.replace(" ", "_")
        filename = directory / f"{prefix}{propname}.dat"
        data.to_file(filename)
        filenames.append(filename)

    return filenames

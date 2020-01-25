import abc
from pathlib import Path
from typing import Union

from monty.serialization import loadfn
from pkg_resources import resource_filename

from amset.core.data import AmsetData
from amset.util import cast_dict_ndarray

seaborn_colors = [
    (0.2980392156862745, 0.4470588235294118, 0.6901960784313725),
    (0.8666666666666667, 0.5176470588235295, 0.3215686274509804),
    (0.3333333333333333, 0.6588235294117647, 0.40784313725490196),
    (0.7686274509803922, 0.3058823529411765, 0.3215686274509804),
    (0.5058823529411764, 0.4470588235294118, 0.7019607843137254),
    (0.5764705882352941, 0.47058823529411764, 0.3764705882352941),
    (0.8549019607843137, 0.5450980392156862, 0.7647058823529411),
    (0.5490196078431373, 0.5490196078431373, 0.5490196078431373),
    (0.8, 0.7254901960784313, 0.4549019607843137),
    (0.39215686274509803, 0.7098039215686275, 0.803921568627451),
]

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

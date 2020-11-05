"""
Module defining io functions.
"""

from typing import Any, Dict

import numpy as np
from monty.serialization import dumpfn, loadfn

from amset.util import cast_dict_list, logger, validate_settings

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"


def write_settings(settings: Dict[str, Any], filename: str):
    """Write amset configuration settings to a formatted yaml file.

    Args:
        settings: The configuration settings.
        filename: A filename.
    """
    settings = cast_dict_list(settings)
    dumpfn(settings, filename, indent=4, default_flow_style=False)


def load_settings(filename: str) -> Dict[str, Any]:
    """Load amset configuration settings from a yaml file.

    If the settings file does not contain a required parameter, the default
    value will be added to the configuration.

    An example file is given in *amset/examples/example_settings.yaml*.

    Args:
        filename: Path to settings file.

    Returns:
        The settings, with any missing values set according to the amset defaults.
    """
    logger.info("Loading settings from: {}".format(filename))
    settings = loadfn(filename)

    return validate_settings(settings)


def write_mesh(mesh_data, filename="mesh.h5"):
    import h5py
    from pymatgen import Structure

    with h5py.File(filename, "w") as f:

        def add_data(name, data):
            if isinstance(data, np.ndarray):
                f.create_dataset(name, data=data, compression="gzip")
            elif isinstance(data, Structure):
                f["structure"] = np.string_(data.to_json())
            elif isinstance(data, (tuple, list)):
                data = np.array(data)
                if isinstance(data[0], str):
                    data = data.astype("S")
                f.create_dataset(name, data=data)
            elif data is None:
                f.create_dataset(name, data=False)
            else:
                f.create_dataset(name, data=data)

        for key, value in mesh_data.items():
            if isinstance(value, dict):
                # dict entries are given for different spins
                for spin, spin_value in value.items():
                    key = "{}_{}".format(key, spin.name)
                    add_data(key, spin_value)
            else:
                add_data(key, value)


def load_mesh(filename):
    import h5py
    from pymatgen import Structure

    from amset.constants import str_to_spin

    def read_data(name, data):
        if name == "structure":
            data_str = np.string_(data[()]).decode()
            return Structure.from_str(data_str, fmt="json")
        if name == "scattering_labels":
            return data[()].astype("U13")  # decode string
        if name == "vb_idx":
            d = data[()]
            return d if d is not False else None
        return data[()]

    mesh_data = {}
    with h5py.File(filename, "r") as f:
        for key, value in f.items():
            if "_up" in key or "_down" in key:
                spin = str_to_spin[key.split("_")[-1]]
                key = key.replace("_{}".format(spin.name), "")
                if key not in mesh_data:
                    mesh_data[key] = {}
                mesh_data[key][spin] = read_data(key, value)
            else:
                mesh_data[key] = read_data(key, value)

    return mesh_data

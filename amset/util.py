import collections
import copy
import logging
import sys
from typing import Any, Dict

import numpy as np
import scipy
from monty.serialization import dumpfn, loadfn
from tqdm import tqdm

from amset.constants import amset_defaults, output_width
from pymatgen import Spin

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"

logger = logging.getLogger(__name__)

_bar_format = "{desc} {percentage:3.0f}%|{bar}| {elapsed}<{remaining}{postfix}"


def validate_settings(user_settings):
    settings = copy.deepcopy(amset_defaults)
    settings.update(user_settings)

    # validate the type of some settings
    if isinstance(settings["doping"], (int, float)):
        settings["doping"] = [settings["doping"]]
    elif isinstance(settings["doping"], str):
        settings["doping"] = parse_doping(settings["doping"])

    if isinstance(settings["temperatures"], (int, float)):
        settings["temperatures"] = [settings["temperatures"]]
    elif isinstance(settings["temperatures"], str):
        settings["temperatures"] = parse_temperatures(settings["temperatures"])

    if isinstance(settings["deformation_potential"], str):
        settings["deformation_potential"] = parse_deformation_potential(
            settings["deformation_potential"]
        )

    settings["doping"] = np.asarray(settings["doping"])
    settings["temperatures"] = np.asarray(settings["temperatures"])

    return settings


def tensor_average(tensor):
    return np.average(scipy.linalg.eigvalsh(tensor))


def groupby(a, b):
    # Get argsort indices, to be used to sort a and b in the next steps
    sidx = b.argsort(kind="mergesort")
    a_sorted = a[sidx]
    b_sorted = b[sidx]

    # Get the group limit indices (start, stop of groups)
    cut_idx = np.flatnonzero(np.r_[True, b_sorted[1:] != b_sorted[:-1], True])

    # Split input array with those start, stop ones
    out = np.array([a_sorted[i:j] for i, j in zip(cut_idx[:-1], cut_idx[1:])])
    return out


def write_settings_to_file(settings: Dict[str, Any], filename: str):
    """Write amset configuration settings to a formatted yaml file.

    Args:
        settings: The configuration settings.
        filename: A filename.
    """
    settings = cast_dict(settings)
    dumpfn(settings, filename, indent=4, default_flow_style=False)


def load_settings_from_file(filename: str) -> Dict[str, Any]:
    """Load amset configuration settings from a yaml file.

    If the settings file does not contain a required parameter, the default
    value will be added to the configuration.

    An example file is given in *amset/examples/example_settings.yaml*.

    Args:
        filename: Path to settings file.

    Returns:
        The settings, with any missing values set according to the amset
        defaults.
    """
    logger.info("Loading settings from: {}".format(filename))
    settings = loadfn(filename)

    return validate_settings(settings)


def cast_dict(d):
    if d is None:
        return d

    new_d = {}
    for k, v in d.items():
        # cast keys
        if isinstance(k, Spin):
            k = k.value

        if isinstance(v, collections.Mapping):
            new_d[k] = cast_dict(v)
        else:
            # cast values
            if isinstance(v, np.ndarray):
                v = v.tolist()
            elif isinstance(v, tuple):
                v = list(v)

            new_d[k] = v
    return new_d


def parse_doping(doping_str: str):
    doping_str = doping_str.strip().replace(" ", "")

    try:
        if ":" in doping_str:
            parts = list(map(float, doping_str.split(":")))

            if len(parts) != 3:
                raise ValueError

            return np.geomspace(parts[0], parts[1], int(parts[2]))

        else:
            return np.array(list(map(float, doping_str.split(","))))

    except ValueError:
        raise ValueError("ERROR: Unrecognised doping format: {}".format(doping_str))


def parse_temperatures(temperatures_str: str):
    temperatures_str = temperatures_str.strip().replace(" ", "")

    try:
        if ":" in temperatures_str:
            parts = list(map(float, temperatures_str.split(":")))

            if len(parts) != 3:
                raise ValueError

            return np.linspace(parts[0], parts[1], int(parts[2]))

        else:
            return np.array(list(map(float, temperatures_str.split(","))))

    except ValueError:
        raise ValueError(
            "ERROR: Unrecognised temperature format: {}".format(temperatures_str)
        )


def parse_deformation_potential(deformation_pot_str: str):
    deformation_pot_str = deformation_pot_str.strip().replace(" ", "")

    try:
        parts = list(map(float, deformation_pot_str.split(",")))

        if len(parts) == 1:
            return parts[0]
        elif len(parts) == 2:
            return tuple(parts)
        else:
            raise ValueError

    except ValueError:
        raise ValueError(
            "ERROR: Unrecognised deformation potential format: "
            "{}".format(deformation_pot_str)
        )


def get_progress_bar(
    iterable=None, total=None, desc="", min_desc_width=18, prepend_pipe=True
):
    if prepend_pipe:
        desc = "    ├── " + desc

    desc += ":"

    if len(desc) < min_desc_width:
        desc += " " * (min_desc_width - len(desc))

    if iterable is not None:
        return tqdm(
            iterable=iterable,
            total=total,
            ncols=output_width,
            desc=desc,
            bar_format=_bar_format,
            file=sys.stdout,
        )
    elif total is not None:
        return tqdm(
            total=total,
            ncols=output_width,
            desc=desc,
            bar_format=_bar_format,
            file=sys.stdout,
        )
    else:
        raise ValueError("Error creating progress bar, need total or iterable")

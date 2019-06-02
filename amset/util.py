import collections
import copy
import logging
from multiprocessing.sharedctypes import RawArray
from typing import Dict, Any

import numpy as np
import scipy
from monty.serialization import dumpfn, loadfn

from amset import amset_defaults
from pymatgen import Spin
from pymatgen.util.string import latexify_spacegroup

spin_name = {Spin.up: "spin-up", Spin.down: "spin-down"}

logger = logging.getLogger(__name__)


def create_shared_array(data, return_buffer=False):
    data = np.asarray(data)
    shared_data = RawArray("d", int(np.prod(data.shape)))
    buffered_data = np.frombuffer(shared_data).reshape(data.shape)
    buffered_data[:] = data[:]

    if return_buffer:
        return shared_data, buffered_data
    else:
        return shared_data


def validate_settings(user_settings):
    settings = copy.deepcopy(amset_defaults)

    def recursive_update(d, u):
        """ Recursive dict update."""
        for k, v in u.items():
            if isinstance(v, collections.Mapping):
                d[k] = recursive_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    recursive_update(settings, user_settings)

    # validate the type of some settings
    if isinstance(settings["general"]["doping"], (int, float)):
        settings["general"]["doping"] = [settings["general"]["doping"]]

    if isinstance(settings["general"]["temperatures"], (int, float)):
        settings["general"]["temperatures"] = [
            settings["general"]["temperatures"]]

    settings["general"]["doping"] = np.asarray(settings["general"]["doping"])
    settings["general"]["temperatures"] = np.asarray(
        settings["general"]["temperatures"])

    return settings


def tensor_average(tensor):
    return np.average(scipy.linalg.eigvalsh(tensor))


def groupby(a, b):
    # Get argsort indices, to be used to sort a and b in the next steps
    sidx = b.argsort(kind='mergesort')
    a_sorted = a[sidx]
    b_sorted = b[sidx]

    # Get the group limit indices (start, stop of groups)
    cut_idx = np.flatnonzero(
        np.r_[True, b_sorted[1:] != b_sorted[:-1], True])

    # Split input array with those start, stop ones
    out = np.array(
        [a_sorted[i:j] for i, j in zip(cut_idx[:-1], cut_idx[1:])])
    return out


def unicodeify_spacegroup(spacegroup_symbol: str):
    subscript_unicode_map = {
        0: "₀",
        1: "₁",
        2: "₂",
        3: "₃",
        4: "₄",
        5: "₅",
        6: "₆",
        7: "₇",
        8: "₈",
        9: "₉",
    }

    symbol = latexify_spacegroup(spacegroup_symbol)

    for number, unicode_number in subscript_unicode_map.items():
        symbol = symbol.replace("$_{" + str(number) + "}$", unicode_number)

    overline = "\u0305"  # u"\u0304" (macron) is also an option

    symbol = symbol.replace("$\\overline{", overline)
    symbol = symbol.replace("$", "")
    symbol = symbol.replace("{", "")
    symbol = symbol.replace("}", "")

    return symbol


def write_settings_to_file(settings: Dict[str, Any], filename: str):
    """Write amset configuration settings to a formatted yaml file.

    Args:
        settings: The configuration settings.
        filename: A filename.
    """
    cast_dict(settings)
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


def gen_even_slices(n, n_packs):
    """Generator to create n_packs slices going up to n.

    Parameters
    ----------
    n : int
    n_packs : int
        Number of slices to generate.

    Yields
    ------
    slice

    Examples
    --------
    """
    start = 0
    if n_packs < 1:
        raise ValueError("gen_even_slices got n_packs=%s, must be >=1"
                         % n_packs)
    for pack_num in range(n_packs):
        this_n = n // n_packs
        if pack_num < n % n_packs:
            this_n += 1
        if this_n > 0:
            end = start + this_n
            yield slice(start, end, None)
            start = end

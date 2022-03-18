"""
Module defining utility functions.
"""
import collections
import copy
import logging
import sys
from multiprocessing.sharedctypes import RawArray
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from tqdm.auto import tqdm

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"


logger = logging.getLogger(__name__)

_bar_format = "{desc} {percentage:3.0f}%|{bar}| {elapsed}<{remaining}{postfix}"


def validate_settings(user_settings: Dict[str, Any]) -> Dict[str, Any]:
    """Parse, validate and fill amset settings.

    Missing settings will be inferred from the amset defaults.

    Args:
        user_settings: A dictionary of settings.

    Returns:
        The validated settings.
    """
    from amset.constants import defaults

    settings = copy.deepcopy(defaults)
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
    elif isinstance(settings["deformation_potential"], list):
        settings["deformation_potential"] = tuple(settings["deformation_potential"])

    if settings["static_dielectric"] is not None:
        settings["static_dielectric"] = cast_tensor(settings["static_dielectric"])

    if settings["high_frequency_dielectric"] is not None:
        settings["high_frequency_dielectric"] = cast_tensor(
            settings["high_frequency_dielectric"]
        )

    if settings["elastic_constant"] is not None:
        settings["elastic_constant"] = cast_elastic_tensor(settings["elastic_constant"])

    if settings["piezoelectric_constant"] is not None:
        settings["piezoelectric_constant"] = cast_piezoelectric_tensor(
            settings["piezoelectric_constant"]
        )

    settings["doping"] = np.asarray(settings["doping"], dtype=np.float)
    settings["temperatures"] = np.asarray(settings["temperatures"])

    for charge_setting in ("donor_charge", "acceptor_charge"):
        if charge_setting in settings:
            settings["defect_charge"] = settings.pop(charge_setting)

    for setting in settings:
        if setting not in defaults:
            raise ValueError(f"Unrecognised setting: {setting}")

    return settings


def cast_tensor(
    tensor: Union[float, List[float], List[List[float]], np.ndarray]
) -> np.ndarray:
    """Cast a number/list into a 3x3 tensor.

    Args:
        tensor: A number, 3x1 list of numbers, or 3x3 list of numbers.

    Returns:
        A 3x3 tensor.
    """
    from amset.constants import numeric_types

    if isinstance(tensor, numeric_types):
        return np.eye(3) * tensor

    tensor = np.asarray(tensor)
    if len(tensor.shape) == 1:
        return np.diag(tensor)

    if tensor.shape != (3, 3):
        raise ValueError("Unsupported tensor shape.")

    return tensor


def cast_elastic_tensor(
    elastic_tensor: Union[int, float, List[List[float]], np.ndarray]
) -> np.ndarray:
    """Cast elastic tensor from single value or Voigt to full 3x3x3x3 tensor.

    Args:
        elastic_tensor: A single number, 6x6 Voigt tensor, or 3x3x3x3 tensor.

    Returns:
        The elastic constant as a 3x3x3x3 tensor.
    """
    from pymatgen.core.tensors import Tensor

    from amset.constants import numeric_types

    if isinstance(elastic_tensor, numeric_types):
        elastic_tensor = np.eye(6) * elastic_tensor
        elastic_tensor[([3, 4, 5], [3, 4, 5])] /= 2

    elastic_tensor = np.array(elastic_tensor)
    if elastic_tensor.shape == (6, 6):
        elastic_tensor = Tensor.from_voigt(elastic_tensor)

    if elastic_tensor.shape != (3, 3, 3, 3):
        raise ValueError(
            "Unsupported elastic tensor shape. Should be (6, 6) or (3, 3, 3, 3)."
        )

    return np.array(elastic_tensor)


def cast_piezoelectric_tensor(
    piezoelectric_tensor: Union[np.ndarray, List[List[float]], np.ndarray]
) -> np.ndarray:
    """Cast piezoelectric tensor from Voigt form to full 3x3x3 tensor.

    Args:
        piezoelectric_tensor: A 3x6 Voigt tensor, or 3x3x3 tensor.

    Returns:
        The piezoelectric constant as a 3x3x3 tensor.
    """
    from pymatgen.core.tensors import Tensor

    piezoelectric_tensor = np.array(piezoelectric_tensor)
    if piezoelectric_tensor.shape == (3, 6):
        piezoelectric_tensor = Tensor.from_voigt(piezoelectric_tensor)

    if piezoelectric_tensor.shape != (3, 3, 3):
        raise ValueError(
            "Unsupported piezoelectric tensor shape. Should be (3, 6) or (3, 3, 3)."
        )

    return np.array(piezoelectric_tensor)


def tensor_average(tensor: Union[List, np.ndarray]) -> Union[float, np.ndarray]:
    """Calculate the average of the tensor eigenvalues.

    Args:
        tensor: A tensor

    Returns:
        The average of the eigenvalues.
    """
    return np.average(np.linalg.eigvalsh(tensor), axis=-1)


def groupby(
    elements: Union[List[Any], np.ndarray], groups: Union[List[int], np.ndarray]
) -> np.ndarray:
    """Groups elements based on the group indices.

    I.e., if elements is `["a", "b", "1", "2", "c", "d"]` and groups is
    `[2, 0, 1, 2, 0, 0]` the output will be `[["b", "c", "d"], ["1"], ["a", "2"]]`.

    Args:
        elements: A list of objects.
        groups: The groups that the objects belong to.

    Returns:
        A nested list of elements that have been grouped.
    """
    groups = np.array(groups)
    elements = np.array(elements)

    # Get argsort indices, to be used to sort a and b in the next steps
    sidx = groups.argsort(kind="mergesort")
    a_sorted = elements[sidx]
    b_sorted = groups[sidx]

    # Get the group limit indices (start, stop of groups)
    cut_idx = np.flatnonzero(np.r_[True, b_sorted[1:] != b_sorted[:-1], True])

    # Split input array with those start, stop ones
    out = np.array(
        [a_sorted[i:j] for i, j in zip(cut_idx[:-1], cut_idx[1:])], dtype=object
    )
    return out


def cast_dict_list(d):
    """Recursively cast numpy arrays in a dictionary to lists.

    Also casts pymatgen Spin objects to str.

    Args:
        d: A dictionary.

    Returns:
        The casted dictionary.
    """
    from pymatgen.electronic_structure.core import Spin

    if d is None:
        return d

    new_d = {}
    for k, v in d.items():
        # cast keys
        if isinstance(k, Spin):
            k = k.name

        if isinstance(v, collections.abc.Mapping):
            new_d[k] = cast_dict_list(v)
        else:
            # cast values
            if isinstance(v, np.ndarray):
                v = v.tolist()
            elif isinstance(v, tuple):
                v = list(v)

            new_d[k] = v
    return new_d


def cast_dict_ndarray(d):
    """Recursively cast lists in dictionaries to numpy arrays.

    Also casts the keys "up" and "down" to pymatgen Spin objects.

    Args:
        d: A dictionary.

    Returns:
        The casted dictionary.
    """
    from pymatgen.electronic_structure.core import Spin

    if d is None:
        return d

    new_d = {}
    for k, v in d.items():
        # cast keys back to spin
        if isinstance(k, str) and k in ["up", "down"]:
            k = Spin.up if "k" == "up" else Spin.up

        if isinstance(v, collections.abc.Mapping):
            new_d[k] = cast_dict_ndarray(v)
        else:
            # cast values
            if isinstance(v, list):
                v = np.array(v)

            new_d[k] = v
    return new_d


def parse_doping(doping_str: str) -> np.ndarray:
    """Parse doping string.

    Args:
        doping_str: String specifying doping. Can be a list of comma separated numbers,
            or a range specification in log space with start:stop:step
            (i.e., "1e16:1e19:4" would give `[1e16, 1e17, 1e18, 1e19]`).

    Returns:
        The doping as a numpy array.
    """
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
        raise ValueError(f"ERROR: Unrecognised doping format: {doping_str}")


def parse_temperatures(temperatures_str: str) -> np.ndarray:
    """Parse temperature string.

    Args:
        temperatures_str: String specifying temperatures. Can be a list of comma
            separated numbers, or a range specification in with start:stop:step
            (i.e., "100:400:4" would give `[100, 200, 300, 400]`).

    Returns:
        The temperatures as a numpy array.
    """
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
        raise ValueError(f"ERROR: Unrecognised temperature format: {temperatures_str}")


def parse_deformation_potential(
    deformation_pot_str: str,
) -> Union[str, float, Tuple[float, ...]]:
    """Parse deformation potential string.

    Args:
        deformation_pot_str: The deformation potential string. Can be a path to a
            deformation.h5 file, a single deformation potential to use for all bands
            or two deformation potentials separated by a comma for valence and
            conduction bands.

    Returns:
        The deformation potential(s) or path to deformation potential file.
    """

    if "h5" in deformation_pot_str:
        return deformation_pot_str

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
    iterable: Optional[Iterable] = None,
    total: Optional[int] = None,
    desc: str = "",
    min_desc_width: int = 18,
    prepend_pipe: bool = True,
) -> tqdm:
    """Get a formatted progress bar.

    One of `iterable` or `total` must be specific.

    Args:
        iterable: An iterable (list, tuple, etc).
        total: The total number of items in the progress bar.
        desc: The descriptive label.
        min_desc_width: Minimum description width.
        prepend_pipe: Add indent and fancy pipe symbol before description.

    Returns:
        The progress bar.
    """
    from amset.constants import output_width

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


def parse_ibands(ibands: Union[str, Tuple[List[int], List[int]]]) -> Dict:
    """
    Parse ibands string or list to dictionary with Spin.up and Spin.down keys.

    Args:
        ibands: The ibands specification. Can be either a string or tuple. Allowed
            str formats include (i) a series of comma separated numbers, i.e., "1, 2, 3"
            that results in an ibands of `{Spin.up: [0, 1, 2]}` (bands in pymatgen are
            zero indexed; (ii) a range can be specified, i.e., "1:3" results
            in the same ibands as above (iii) Spin up and spin down ibands (only useful
            for spin-polarized systems, can be specified by separating the spin channels
            with a period. I.e., "1,2,3.4:6", results in an ibands of `{Spin.up:
            [0, 1, 2], Spin.down: [3, 4, 5]}`. Alternatively, a tuple of lists can be
            given. I.e., `([1, 2, 3], [4, 5, 6])` gives the same ibands as the last
            example.

    Returns:
        The ibands specification.
    """
    from pymatgen.electronic_structure.core import Spin

    new_ibands = {}
    if isinstance(ibands, str):
        try:
            for spin, spin_ibands in zip((Spin.up, Spin.down), ibands.split(".")):
                if ":" in spin_ibands:
                    parts = list(map(int, spin_ibands.split(":")))
                    if len(parts) != 2:
                        raise ValueError
                    new_ibands[spin] = list(range(parts[0], parts[1] + 1))
                else:
                    new_ibands[spin] = list(map(int, spin_ibands.split(",")))

        except ValueError:
            raise ValueError(f"ERROR: Unrecognised ibands format: {ibands}")
    elif isinstance(ibands, (list, tuple)):
        if not isinstance(ibands[0], (list, tuple)):
            new_ibands[Spin.up] = ibands
        else:
            new_ibands[Spin.up] = ibands[0]
            new_ibands[Spin.down] = ibands[1]
    return {s: np.array(i, dtype=int) - 1 for s, i in new_ibands.items()}


def create_shared_array(data: np.ndarray, return_shared_data=False):
    data = np.asarray(data)
    if data.dtype == np.complex:
        data_type = "complex"
        data_buffer = RawArray("d", int(np.prod(data.shape)) * 2)
    else:
        data_type = np.ctypeslib.as_ctypes_type(data.dtype)
        data_buffer = RawArray(data_type, int(np.prod(data.shape)))

    buffer = (data_buffer, data.shape, data_type)
    data_shared = array_from_buffer(buffer)
    data_shared[:] = data[:]

    if return_shared_data:
        return buffer, data_shared
    else:
        return buffer


def create_shared_dict_array(data: Dict[Any, np.ndarray], return_shared_data=False):
    # turns a dict of key: np.ndarray to a dict of key: buffer
    data_buffer = {}
    data_shared = {}
    for key, value in data.items():
        data_buffer[key], data_shared[key] = create_shared_array(
            value, return_shared_data=True
        )

    if return_shared_data:
        return data_buffer, data_shared
    else:
        return data_buffer


def array_from_buffer(buffer):
    data_buffer, data_shape, data_type = buffer
    if data_type == "complex":
        return np.frombuffer(data_buffer).view(np.complex).reshape(data_shape)
    else:
        return np.frombuffer(data_buffer, dtype=data_type).reshape(data_shape)


def dict_array_from_buffer(buffer):
    data = {}
    for key, value_buffer in buffer.items():
        data[key] = array_from_buffer(value_buffer)
    return data

import copy
import collections
import logging
from typing import Any, Dict

from monty.serialization import loadfn, dumpfn

from amset import amset_defaults

logger = logging.getLogger(__name__)


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

    def recursive_update(d, u):
        """ Recursive dict update."""
        for k, v in u.items():
            if isinstance(v, collections.Mapping):
                d[k] = recursive_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    settings = copy.deepcopy(amset_defaults)
    user_settings = loadfn(filename)

    recursive_update(settings, user_settings)

    return settings


def write_settings_to_file(settings: Dict[str, Any], filename: str):
    """Write amset configuration settings to a formatted yaml file.

    Args:
        settings: The configuration settings.
        filename: A filename.
    """
    logger.info("Writing settings to: {}".format(filename))
    dumpfn(settings, filename, indent=4, default_flow_style=False)


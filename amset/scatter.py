"""
This module implements methods to calculate electron scattering based on an
ElectronStructure object.
"""
from typing import Optional, Union, List

from monty.json import MSONable


class Scatterer(MSONable):

    def __init__(self,
                 materials_properties,
                 scattering: Optional[Union[str, List[str], float]] = "auto",
                 energy_tol: float = 0.01,
                 g_tol: float = 0.01
                 ):
        pass



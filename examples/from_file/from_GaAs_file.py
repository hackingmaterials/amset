"""
This example shows how to read a calculations from files for post-processing.
For example for plotting various scattering rates at difference concentrations
and temperatures.

Note that this should be run after GaAs example is run with the .to_file()
method called after the calculation.
"""
import os
from amset.core import Amset

if __name__ == "__main__":
    amset = Amset.from_file(path=os.path.join("..", "GaAs", "run_data"),
                            filename="amsetrun.json")

    amset.plot(k_plots=['energy', 'velocity'],
               E_plots=['velocity', 'ACD', 'S_i', 'S_o']+amset.elastic_scats,
               show_interactive=True,
               carrier_types=amset.all_types)
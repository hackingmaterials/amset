# GaAs Example

This folder contains the files needed to calculated the transport properties of
GaAs using isotropic materials parameters. The vasprun coefficients were calculated on a 
Γ-centered 17x17x17 k-point mesh. The `wavefunction.h5` file was extracted from the VASP
WAVECAR file using the `amset wave` command.

## Running AMSET

AMSET can be run using either the command line or python API.

### Via the command-line

Steps:
1. Run `amset run` in this directory to calculate transport properties. 
2. Run `amset plot rates mesh_99x99x99.h5` to plot the scattering rates. You can select
   the doping and temperature to plot using the `--temperature-idx` and `--doping-idx` 
   options. The doping indexes are zero indexed. E.g., 
   `amset plot rates mesh_99x99x99.h5 --temperature-idx 5` would plot the rates for 
   789 K, the 6th temperature in the `temperatures` array.
   
### Via the python API
   
Steps:
1. An example python script has been included as `GaAs.py`. This will run AMSET and plot
   the scattering rates. You can run the script by executing `python GaAs.py` in the 
   current directory.

## Output files

The transport results will be written to `transport_99x99x99.json`. The `99x99x99` is the
final interpolated mesh upon which scattering rates and the transport properties are
computed.

The `write_mesh` option is set to True, meaning scattering rates for all doping levels
and temperatures will be written to the `mesh_99x99x99.h5` file.


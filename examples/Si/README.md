# Si Example

This folder contains the files needed to calculated the transport properties of
Si using fully anisotropic materials parameters. The vasprun and wavefunction 
coefficients were calculated on a Γ-centered 18x18x18 k-point mesh. The 
`wavefunction.h5` file was extracted from the VASP WAVECAR file using the `amset wave` 
command. Band and k-point dependent deformation potentials were computed using 
`amset deform read` based on deformations created using `amset deform create` — 
further information is [available in the documentation](https://hackingmaterials.lbl.gov/amset/).


In this folder we specify to only calculate acoustic deformation potential and
ionized impurity scattering. Si is non-polar, so polar optical phonon scattering does
not occur. We also specify the full dielectric tensor as an example. However, the cubic 
symmetry means the components of the dielectric tensor are all the same.

## Running AMSET

AMSET can be run using either the command line or python API.

### Via the command-line

Steps:
1. Run `amset run` in this directory to calculate transport properties. 
2. Run `amset plot rates mesh_105x105x105.h5` to plot the scattering rates. You can 
   select the doping and temperature to plot using the `--temperature-idx` and 
   `--doping-idx` options. The doping indexes are zero indexed. E.g., 
   `amset plot rates mesh_105x105x105.h5 --doping-idx 5` would plot the rates for 
   n<sub>e</sub> = 4.4<sup>18</sup> cm<sup>–3</sup>, the 6th doping level in the 
   `doping` array.
   
### Via the python API
   
Steps:
1. An example python script has been included as `Si.py`. This will run AMSET and plot
   the scattering rates. You can run the script by executing `python Si.py` in the 
   current directory.

## Output files

The transport results will be written to `transport_105x105x105.json`. The `105x105x105`
is the final interpolated mesh upon which scattering rates and the transport properties
are computed.

The `write_mesh` option is set to True, meaning scattering rates for all doping levels
and temperatures will be written to the `mesh_105x105x105.h5` file.

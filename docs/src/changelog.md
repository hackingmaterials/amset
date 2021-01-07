# Change log

## [Unreleased]

## v0.4.5

Enhancements:

- `--gnuplot` option added to transport plotter to allow writing the plot data to simple 
  text files.


## v0.4.4

Enhancements:

- Amset can now be run from a band_structure_data.json file. This should contain the
  keys "band_structure" and "nelect".

Bug fixes:

- Improved support for spin-polarized calculations.
- Fixed projection overlaps.

## v0.4.3

Enhancements:

- `--n-type` and `--p-type` options added to transport, mobility, and convergence plotters.
- Power factor added to transport and convergence plotters.

Bug fixes:

- Fix for mean free path scattering (@kbspooner).
- Fix for piezoelectric scattering.
- Fix for `cache_wavefunction = False` with non-SOC wavefunctions.
- Specify numba version for interoperability with interpolation package.

## v0.4.2

New features:

- Added tool to plot transport properties (`amset plot transport`).
- Added tool to plot mobility in more detail (`amset plot mobility`).
- Added tool to plot convergence (`amset plot convergence`).
- Added option to highlight scattering rates by the derivative of the Fermi–Dirac
  thought `amset plot rates <filename> --dfde`.

Enhancements:

- Reduce memory requirements when `cache_wavefunction = False`.
- Don't write output files if `file_format = None`. 

Bug fixes:

- Re-enabled CRTA and MFP scattering.
- Don't use multiprocessing with basic scatterers.
- Fix direction dependent effective masses.

## v0.4.1

Enhancements:

- Faster wave function overlap calculation using numba jit.
- Better management of memory and error reporting in subprocesses
- Automatically handle memory errors when caching wave function coefficients.

## v0.4.0

New features:

- Multiprocessing now used in the calculation of scattering rates. Number of processes
  controlled using the `nworkers` option.

Bug fixes:

- More robust extraction of deformation potentials.
- Only use ascii characters in output log files.

## v0.3.3

New features:

- Enable amset to handle systems containing a single k-point in a certain direction
  (useful for 2D materials).

## v0.3.2

Bug fixes:

- Fixed a bug in extracting deformation potentials introduced in version 0.3.1.

## v0.3.1

New features:

- `free_carrier_screening` option to allow free carriers to screen polar optical and 
  piezoelectric scattering rates (see docs for me details).
 
Bug fixes:

- Fixed a bug in extracting deformation potentials introduced in version 0.3.0.

## v0.3.0

New features:

- `cache_wavefunction` option to control memory demand (see docs for more details).
- Revtex plot style support. Enabled by adding `--style revtex` to the end of 
  plotting commands.
- Support for spin–orbit coupling.
- Support for non-Gamma centered k-point meshes.
- Ability to extract deformation potential for specific bands.
- `zero_weighted_kpoints` option to control processing of zero-weighted k-points 
  (see the docs for more details).
  
Enhancements:

- Revamped lineshape plotter.
- Massive (~100x) speedup for calculating polar optical phonon frequency.
- Better handling of Fermi levels from VASP band structures.
- Speed up effective mass calculation, and cases where only basic scatterers are used.

## v0.2.2

Fix PyPi installation.

## v0.2.1

Fix GitHub releases.

## v0.2.0

Major update with many new features:

- Elastic, dielectric, and piezoelectric tensors are now supported.
- Wave function coefficients are now desymmetrised on the fly, meaning 
  `wavefunction.h5` files are much smaller.
- New tool to extract wave function coefficients that removes the `pawpyseed` 
  requirement and is much faster. This is a python only implementation and doesn't 
  require compiling any additional codes.
- Mesh properties (scattering rates etc, energies, velocities) stored in a separate 
  mesh.h5 file which is much smaller and faster to read.
- Revamped unit tests.

Lots of bug fixes, including fixing compatibility with quadpy.

## v0.1.3

Bug fix for latest quadpy version.

## v0.1.2

Fix pypi description.

## v0.1.1

Add release and packaging support.

## v0.1.0

Initial release containing:

- `amset` command line tool
- Ionized impurity, acoustic deformation potential, piezeoelectric, and polar
  optical phonon scattering.
- Quantum mechanical wave function overlaps.
- Modified tetrahedron integration.

## v0.0.0

Project created.

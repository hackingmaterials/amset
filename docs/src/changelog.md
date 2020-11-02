# Change log

## [Unreleased]

Added new features:

- Revamped lineshape plotter.
- Added `cache_wavefunction` option to control memory demand (see docs for more details).
- Added revtex plot style support. Enabled by adding `--style revtex` to the end of 
  plotting commands.
- Massive (~100x) speedup for calculating polar optical phonon frequency.
- Added support for spin–orbit coupling.
- Better handling of Fermi levels from VASP band structures.

## v0.2.2

Fix PyPi installation.

## v0.2.1

Fix GitHub releases.

## v0.2.0

Major update with many new features:

- Elastic, dielectric, and piezoelectric tensors are now supported.
- Wave function coefficients are now desymmetrised on the fly, meaning 
  `wavefunction.h5` files are much smaller.
- New tool to extract wave function coefficients that removes the `pawpyseed`  and is 
  much faster. This is a python only implementation and doesn't require compiling any 
  additional codes.
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

# Change log

## [Unreleased]

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

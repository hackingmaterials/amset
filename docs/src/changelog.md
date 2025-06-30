# Change log

## v0.5.1

* Correction to phonon eigenvector reshaping in `phonon_frequency.py` by @Shiva-sslg in [PR #938](https://github.com/hackingmaterials/amset/pull/958)
* Bugfix in variable name by @Grimenes in [PR #960](https://github.com/hackingmaterials/amset/pull/960)

## v0.5.0

This version introduces a major bugfix in the calculation of wavefunction overlaps. This can alter transport properties for systems where the band edge is located at a reciprocal zone-boundary and the dominant scattering type is from polar optical phonons. Thanks to Øven Grimenes for idenitfying this bug and contributing a fix ([PR #938](https://github.com/hackingmaterials/amset/pull/938), @Grimenes).

A new feature has been provided to include non analytical corrections (NAC) in the calculation of the polar optical phonon frequency. More information is provided in the [online documentation](https://hackingmaterials.lbl.gov/amset/inputs/#dielectric-constants-piezoelectric-constants-and-polar-phonon-frequency). Thanks to Sara Shivalingam Goud for providing this feature ([PR #931](https://github.com/hackingmaterials/amset/pull/931), @Shiva-sslg).

## v0.4.22

Fix deformation potentials with spin-polarised materials.

## v0.4.21

Fix compatbility with spglib.

## v0.4.20

Fix intermittent casting bug.

## v0.4.19

Fix numpy deprecations.

## v0.4.18

- Fix issue with interpolation package [#657](https://github.com/hackingmaterials/amset/pull/657)
  by [@lizhenzhupearl])(https://github.com/lizhenzhupearl).
- `eff-mass` can now be run on vaspruns without orbital projections.

## v0.4.17

Remove unused desymmetrisation routines.

## v0.4.16

AMSET dependencies are now not pinned to specific versions. This should make installing
and upgrading a bit easier. Additionally, the quadpy package has been removed as a
dependency.

## v0.4.15

Bug fixes:

- Fixed extraction of polar phonon frequency for VASP 6.

## v0.4.14

Bug fixes:

- Fixed extraction of deformation potentials when a deformation calculation has more
  bands than the undeformed calculation.

## v0.4.13

Bug fixes:

- Fixed interpolation of scattering rates at low doping concentrations.

## v0.4.12

Bug fixes:

- Fixed issues in the interpolation of IMP scattering rates.
- Lineshape plotter now works again.

## v0.4.11

Bug fixes:

- Fixed calculation of k-point meshes from k-point differences in `amset deform read`.

## v0.4.10

Bug fixes:

- Fix printing of VBM and CBM band indices in `amset deform read`.
- Better handling of symprec option in `amset deform read`.
- Make amset compatible with pymatgen v2022

## v0.4.9

Enhancements:

- Better warnings in plotting module.
- Support for `band_structure_data.json` with `amset eff-mass`.

Bug fixes:

- Fixed desymmetrization of spin–orbit coupling (spinor) wave functions.
- Use `eigh` rather than `eig` for transport tensors.

## v0.4.8

Bug fixes:

- Fixed a number of issues in extracting deformation potentials. amset now attempts
  to handle cases where the reciprocal and k-space lattices belong to difference
  classes.

## v0.4.7

Changes:

- Default of `zero_weighted_kpoints` option has been changed from `keep` to `prefer`.
- `acceptor_charge` and `donor_charge` options have been merged into a single option,
  `defect_charge`.

Enhancements:

- Added `--bands` option to `amset wave` to allow selecting specific band ranges.

Bug fixes:

- Fixed the calculation of ionized impurity concentration in bipolar materials and for
  charge states != 1.
- Fixed the calculation of spin-orbit wave function overlaps.
- Fixed warning messages in extraction of wave function coefficients.
- Clarified `phonon_frequency` output.

## v0.4.6

Enhancements:

- `--stats` option added to band plotter that prints the effective masses and band
  structure information.

Bug fixes:

- Fixed extracting wavefunction coefficients in systems with zero weighted k-points.

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

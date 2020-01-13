# AMSET: *ab initio* scattering and transport

## Introduction

Accurately calculating electronic transport properties from first-principles
is often highly computationally expensive and difficult to perform. AMSET is a
fast and easy-to-use package to model carrier transport in solid-state materials.
A primary aim of AMSET is to be **amenable to high-throughput screenings**.
Features of AMSET include:

- All inputs easily obtainable from first-principles calculations. The
  primary input for AMSET is an *ab initio* uniform band structure.
- Scattering rates approximated based on common materials properties
  such as phonon frequencies and dielectric constants.
- Transport properties calculated by solving the iterative Boltzmann transport
  equation.
- Heavily optimised code that can run on a personal laptop. High-performance
  computing clusters not necessary.

AMSET is built on top of state-of-the-art open-source libraries:
[BoltzTraP2](http://boltztrap.org/) for band structure interpolation,
[numpy](https://www.numpy.org/) and
[scipy](https://scipy.org) to enable high-performance matrix operations, and
[pymatgen](http://pymatgen.org) for handling DFT calculation data.

!!! info "Supported ab initio codes"
    Currently, AMSET is best integrated with VASP, however,
    support for additional periodic DFT codes will be added in the future.

### Scattering Mechanisms

The scattering mechanisms currently implemented in AMSET are:

- Acoustic deformation potential scattering
- Ionized impurity scattering
- Polar optical phonon scattering
- Piezoelectric scattering

More information on the formalism for each scattering mechanism is available
in the [scattering section](scattering) of the documentation.

## What's new?

Track changes to AMSET through the [changelog](changelog).

## Contributing / Contact / Support

Want to see something added or changed? Some ways to get involved are:

- Help us improve the documentation – tell us where you got stuck and improve
  the install process for everyone.
- Let us know if you'd like to see certain features.
- Point us to areas of the code that are difficult to understand or use.
- Contribute code. You can do this by forking
  [AMSET on Github](https://github.com/hackingmaterials/amset) and submitting
  a pull request.

The list of contributors to AMSET can be found [here](contributors).
Read more about contributing code to AMSET [here](contributing).

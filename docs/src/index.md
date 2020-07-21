# AMSET: *ab initio* scattering and transport

## Introduction

<img alt="amset properties" src="properties.jpg">

AMSET is an efficient package for calculating electron lifetimes and
transport properties in solid-state materials from first principles.
A primary aim of AMSET is to be amenable to high-throughput computational
screening.
Features of AMSET include:

- Inputs obtainable from first-principles calculations. The
  primary input for AMSET is an uniform band structure calculation.
- Scattering rates calculated in the Born approximation using common materials
  properties such as phonon frequencies and dielectric constants.
- Transport properties calculated through the Boltzmann transport equation.
- Efficient implementation that can run on a personal laptop.

!!! info "Supported ab initio codes"
    Currently, AMSET only supports VASP calculations, however,
    additional periodic DFT codes will be added in the future.

## Scattering Mechanisms

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

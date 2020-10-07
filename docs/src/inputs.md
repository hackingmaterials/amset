# Calculation Inputs

## Structural relaxation

In order to obtain accurate results, the crystal structure should first be relaxed
using "tight" calculation settings including high force and energy convergence
criteria. Note, that this can often be expensive for very large structures.

!!! summary "VASP settings for tight convergence "
    ```python
    ADDGRID = True
    EDIFF = 1E-8
    EDIFFG = -5E-4
    PREC = Accurate
    NSW = 100
    ISIF = 3
    NELMIN = 5
    ```
    
## Dense uniform band structure and wave function coefficients

AMSET should be run on a `vasprun.xml` file from a "dense" uniform band structure 
calculation. Typically a k-point mesh density at least twice that needed to converge 
the total energy will be necessary to converge transport properties. Note this refers
to the initial DFT mesh before Fourier interpolation. In order to obtain accurate band 
gaps often a hybrid DFT functional such as HSE06 is required.

Wave function coefficients are required to calculate wave function overlaps.
This requires the `WAVECAR` file to be written by VASP (achieved by setting 
`LWAVE = True`). Wave function coefficients can then be extracted using the 
`amset wave` command. Coefficients are stored in the `wavefunction.h5` file.

!!! summary "VASP settings for uniform calculations "
    ```python
    ADDGRID = True
    EDIFF = 1E-8
    PREC = Accurate
    NSW = 1
    LWAVE = True
    ```
    
## Elastic constants

Elastic constants can be calculated using finite differences in VASP.
It is very important to first relax the structure using tight convergence 
settings, as described in the [structural relaxation section](#structural-relaxation).
Details on the finite difference approach in VASP can be found on the 
[IBRION](https://www.vasp.at/wiki/index.php/IBRION) documentation page.

!!! summary "VASP settings for dielectric constants and phonon frequency "
    ```python
    ADDGRID = True
    EDIFF = 1E-8
    PREC = Accurate
    NSW = 1
    IBRION = 6
    ```
    
## Deformation potentials

The absolute deformation potential describes the change in energy of the bands with change 
in  volume and is calculated as $`\mathbf{D}_{n\mathbf{k}} = \delta \varepsilon_{n\mathbf{k}} / \delta S_{\alpha\beta}`$ where $`\mathbf{S}`$ is the uniform stress tensor.
The deformation potential should be averaged over contraction (–0.5 %) and 
expansion (+0.5 %) of the lattice and calculated separately for
each component of the strain tensor. To account for shifts in the average electrostatic
potential between deformed cells, the eigenvalues are aligned to the average energy 
level of the core states.

AMSET includes a tool to assist with the calculation of the deformation potentials.
The initial input is a "tight" optimised structure as described in the 
[structural relaxation section](#structural-relaxation). Deformed structures are 
generated using the `amset deform create` command, which will generate a list of 
POSCARs each corresponding to a component of the strain tensor. Note that symmetry is
automatically used to reduce the number of calculations needed. A single point 
calculation (no relaxation, i.e., `NSW = 0`) should be performed for each deformed
POSCAR as well as the undeformed structure.

!!! summary "VASP settings for deformation calculations "
    ```python
    ADDGRID = True
    EDIFF = 1E-8
    PREC = Accurate
    NSW = 1
    ICORELEVEL = 1  # needed to write the core levels to OUTCAR
    ```

The deformation potentials can be calculated using the `amset deform read` command.
This requires the paths to the undeformed and deformation calculations as inputs.
The undeformed folder should be specified first, followed by the deformation folders.
For example,

```bash
amset deform read undeformed def-1 def-2 def-3
```

This will write the deformations potentials to a `deformation.h5` file in the current
directory. You can specify to use this file when calculating scattering rates by
setting the `deformation_potential` option to `"deformation.h5"`. 
[See the settings](settings.md#deformation_potential) page for more details.


## Dielectric constants, piezoelectric constants and polar-phonon frequency

Static and high-frequency dielectric constants, piezoelectric constants, and the 
"effective polar phonon frequency" can be obtained using density functional perturbation 
theory (DFPT). It is very important to first relax the structure using tight convergence 
settings, as described in the [structural relaxation section](#structural-relaxation).
Details on DFPT in VASP can be found on the [IBRION](https://www.vasp.at/wiki/index.php/IBRION)
and [LEPSILON](https://www.vasp.at/wiki/index.php/LEPSILON) documentation pages.

!!! summary "VASP settings for dielectric constants and phonon frequency "
    ```python
    ADDGRID = True
    EDIFF = 1E-8
    PREC = Accurate
    NSW = 1
    IBRION = 8
    LEPSILON = True
    ```

Note, DFPT cannot be used with hybrid exchange-correlation functionals. In these
cases the [LCALCEPS](https://www.vasp.at/wiki/index.php/LCALCEPS) flag should be
used in combination with `IBRION = 6`.

The dielectric constants and polar phonon frequency can be extracted from the
VASP outputs using the command:
```bash
amset phonon-frequency
```
The command should be run in a folder containing the `vasprun.xml` file output
from the DFPT calculation.

The effective phonon frequency is determined from the phonon frequencies 
$`\omega_{\mathbf{q}\nu}`$ (where $`\nu`$ is a phonon branch and $`\mathbf{q}`$
is a phonon wave vector) and eigenvectors $`\mathbf{e}_{\kappa\nu}(\mathbf{q})`$
(where $`\kappa`$ is an atom in the unit cell). In order to capture scattering 
from the full phonon band structure in a single phonon frequency, each phonon 
mode is weighted by the dipole moment it produces according to
```math
w_{\nu} = \sum_\kappa \left [ \frac{1}{M_\kappa \omega_{\mathbf{q}\nu}} \right]^{1/2}
\times \left[ \mathbf{q} \cdot \mathbf{Z}_\kappa^* \cdot \mathbf{e}_{\kappa\nu}(\mathbf{q}) \right ]
```
where $`\mathbf{Z}_\kappa^*`$ is the Born effective charge.
This naturally suppresses the contributions from transverse-optical and acoustic
modes in the same manner as the [more general formalism](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.115.176401) 
for computing Frölich based electron-phonon coupling.

The weight is calculated only for $`\Gamma`$-point phonon frequencies and 
averaged over the full unit sphere to capture both the polar divergence 
at $`\mathbf{q} \rightarrow 0`$ and any anisotropy in the dipole moments.
The effective phonon frequency is calculated as the weighted sum over all 
$`\Gamma`$-point phonon modes according to
```math
\omega_\mathrm{po} = \frac{\omega_{\Gamma\nu} w_{\nu}}{\sum_{\nu} w_\nu}.
```


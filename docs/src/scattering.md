# Calculating scattering rates

AMSET calculates mode dependent scattering rates within the Born approximation 
using common materials parameters. The differential scattering rate from state 
$`\mathinner{|n\mathbf{k}\rangle}`$ to state 
$`\mathinner{|m\mathbf{k} + \mathbf{q}\rangle}`$ is calculated using 
Fermi's golden rule as

```math
    \tilde{\tau}_{n\mathbf{k}\rightarrow m\mathbf{k}+\mathbf{q}}^{-1} = 
        \frac{2\pi}{\hbar} \lvert g_{nm}(\mathbf{k}, \mathbf{q}) \rvert^2
        \delta(\varepsilon_{n\mathbf{k}} - 
        \varepsilon_{m\mathbf{k}+\mathbf{q}}),
```

where $`\varepsilon_{n\mathbf{k}}`$ is the energy of state
$`\mathinner{|n\mathbf{k}\rangle}`$, and $`g_{nm}(\mathbf{k}, \mathbf{q})`$
is the matrix element for scattering from state 
$`\mathinner{|n\mathbf{k}\rangle}`$ into 
state $`\mathinner{|m\mathbf{k} + \mathbf{q}\rangle}`$. 

!!! info 
    Note, this is the expression for elastic scattering. Inelastic scattering
    contains addition terms, as detailed in the 
    [elastic vs inelastic scattering section](#elastic-vs-inelastic-scattering).

The overall mode-dependent scattering rate is obtained by 
[integrating the scattering rates](#brillouin-zone-integration) 
over the full Brillouin zone.  In this section, we report the matrix elements
for each scattering mechanism implemented in AMSET. Information on calculating 
transport properties is given in the 
[transport properties section](transport-properties.md).

## Summary of scattering rates

Mechanism                                                                               | Code  | Requires                                                                       | Type
---                                                                                     | ---   | ---                                                                            | ---
[Acoustic deformation potential scattering](#acoustic-deformation-potential-scattering) | ADP   | *n*- and *p*-type deformation potential,  elastic constant                     | Elastic
[Piezoelectric scattering](#piezoelectric-scattering)                                   | PIE   | high-frequency dielectric constant, elastic constant, piezoelectric coefficient ($`\mathbf{e}`$) | Elastic
[Polar optical phonon scattering](#polar-optical-phonon-scattering)                     | POP   | polar optical phonon frequency, static and high-frequency dielectric constants | Inelastic
[Ionized impurity scattering](#ionized-impurity-scattering)                             | IMP   | static dielectric constant                                                     | Elastic

### Acoustic deformation potential scattering

The acoustic deformation potential matrix element is given by
```math
g_{nm}^\mathrm{ad}(\mathbf{k}, \mathbf{q}) = 
   \sqrt{k_\mathrm{B} T}  \sum_{\mathbf{G} \neq -\mathbf{q}} \left[ 
        \frac{\mathbf{\tilde{D}}_{n\mathbf{k}} \mathbin{:} \mathbf{\hat{S}}_l}{c_l\sqrt{\rho}} + 
        \frac{\mathbf{\tilde{D}}_{n\mathbf{k}} \mathbin{:} \mathbf{\hat{S}}_{t_1}}{c_{t_1}\sqrt{\rho}} + 
        \frac{\mathbf{\tilde{D}}_{n\mathbf{k}} \mathbin{:} \mathbf{\hat{S}}_{t_2}}{c_{t_2}\sqrt{\rho}}
    \right]
    \mathinner{\langle m\mathbf{k}+\mathbf{q} \left | e^{i(\mathbf{q} + \mathbf{G})\cdot\mathbf{r}} \right | n\mathbf{k} \rangle},
```
where $`\mathbf{\tilde{D}}_{n\mathbf{k}} = \mathbf{D}_{n\mathbf{k}} + \mathbf{v}_{n\mathbf{k}} \otimes \mathbf{v}_{n\mathbf{k}}`$ 
in which $`\mathbf{D}_{n\mathbf{k}}`$ is the rank 2 deformation potential tensor,  $`\mathbf{\hat{S}} = \mathbf{\hat{q}}\otimes\mathbf{\hat{u}}`$ is the unit strain associated 
with an acoustic mode, $`\mathbf{\hat{u}}`$ is the unit vector of phonon polarization,
and the subscripts $`l`$, $`t_1`$, and $`t_2`$ indicate properties belonging to the 
longitudinal and transverse modes.

!!! quote ""
    - *Abbreviation:* APD
    - *Type:* Elastic
    - *References:* [^Bardeen], [^Shockley], [^Rode]
    - *Requires:* `deformation_potential`, `elastic_constant`
    
### Piezoelectric scattering

The piezoelectric differential scattering rate is given by

```math
g_{nm}^\mathrm{pi}(\mathbf{k}, \mathbf{q}) =
   \sqrt{k_\mathrm{B} T} \sum_{\mathbf{G} \neq -\mathbf{q}}  \left[ 
        \frac{\mathbf{\hat{n}} \mathbf{h} \mathbin{:} \mathbf{\hat{S}}_l}{c_l\sqrt{\rho}} + 
        \frac{\mathbf{\hat{n}} \mathbf{h} \mathbin{:} \mathbf{\hat{S}}_{t_1}}{c_{t_1}\sqrt{\rho}} + 
        \frac{\mathbf{\hat{n}} \mathbf{h} \mathbin{:} \mathbf{\hat{S}}_{t_2}}{c_{t_2}\sqrt{\rho}}
    \right ]
    \frac{\mathinner{\langle m\mathbf{k}+\mathbf{q} \left | e^{i(\mathbf{q} + \mathbf{G})\cdot\mathbf{r}} \right | n\mathbf{k} \rangle}}
         {\left | \mathbf{q} + \mathbf{G} \right |},
```
where $`\mathbf{h}`$ is the full piezoelectric stress tensor and $`\mathbf{\hat{n}} = (\mathbf{q} + \mathbf{G}) / \left | \mathbf{q} + \mathbf{G} \right |`$ is a unit vector in the direction of scattering. 

!!! quote ""
    - *Abbreviation:* PIE
    - *Type:* Elastic
    - *References:* [^Rode]
    - *Requires:* `piezoelectric_coefficient`, `static_dielectric`

### Polar optical phonon scattering

The polar optical phonon differential scattering rate is given by

```math
g_{nm}^\mathrm{po}(\mathbf{k}, \mathbf{q}) =
    \left [ \frac{\hbar \omega_\mathrm{po}}{2} \right ] ^ {1/2} 
    \sum_{\mathbf{G} \neq -\mathbf{q}}
        \left (\frac{1}{\mathbf{\hat{n}}\cdot\boldsymbol{\epsilon}_\infty\cdot\mathbf{\hat{n}}} - \frac{1}{\mathbf{\hat{n}}\cdot\boldsymbol{\epsilon}_\mathrm{s}\cdot\mathbf{\hat{n}}}\right)
         ^\frac{1}{2}
    \frac{\mathinner{\langle m\mathbf{k}+\mathbf{q} \left | e^{i(\mathbf{q} + \mathbf{G})\cdot\mathbf{r}} \right | n\mathbf{k} \rangle}}
         {\left | \mathbf{q} + \mathbf{G} \right |},
```

where  $`\boldsymbol{\epsilon}_\mathrm{s}`$ and $`\boldsymbol{\epsilon}_\infty`$ are the 
static and high-frequency dielectric tensors and $`\omega_\mathrm{po}`$ is the polar 
optical phonon frequency. To capture scattering from the full phonon band structure in 
a single phonon frequency, each phonon mode is weighted by the dipole moment it 
produces.

!!! quote ""
    - *Abbreviation:* POP
    - *Type:* Inelastic
    - *References:* [^Frohlich], [^Conwell], [^Rode]
    - *Requires:* `pop_frequency`, `static_dielectric`, `high_frequency_dielectric`

### Ionized impurity scattering

The ionized impurity matrix element is given by
```math
g_{nm}^\mathrm{ii}(\mathbf{k}, \mathbf{q}) =
    \sum_{\mathbf{G} \neq -\mathbf{q}} 
     \frac{n_\mathrm{ii}^{1/2} Z e }{\mathbf{\hat{n}} \cdot \boldsymbol{\epsilon}_\mathrm{s} \cdot \mathbf{\hat{n}}}
    \frac{\mathinner{\langle m\mathbf{k}+\mathbf{q} \left | e^{i(\mathbf{q} + \mathbf{G})\cdot\mathbf{r}} \right | n\mathbf{k} \rangle}}
         {\left | \mathbf{q} + \mathbf{G} \right | ^2 + \beta^2},
```
where $`Z`$ is the charge state of the impurity center, 
$`n_\mathrm{ii}`$ is the concentration of ionized impurities
(i.e., $`n_\mathrm{holes} + n_\mathrm{electrons}`$),
and $`\beta`$ is the inverse screening length, defined as

```math
    \beta^2 = \frac{e^2}{\epsilon_\mathrm{s}  k_\mathrm{B} T}
        \int \frac{\mathrm{d}\varepsilon}{V}\,D(\varepsilon) f(1-f),
```

where $`V`$ is the unit cell volume, $`D`$ is the density of states, and 
$`f`$ is the Fermi–Dirac distribution given in the
[transport properties section](transport-properties.md).

!!! quote ""
    - *Abbreviation:* IMP
    - *Type:* Elastic
    - *References:* [^Dingle], [^Rode]
    - *Requires:* `static_dielectric`


## Elastic vs inelastic scattering

AMSET treats elastic and inelastic scattering mechanisms separately. 

### Inelastic

The  differential scattering rate for inelastic processes is calculated as

```math
\begin{aligned}
    \tau_{n\mathbf{k}\rightarrow m\mathbf{k}+\mathbf{q}}^{-1} = 
        \frac{2\pi}{\hbar} \lvert g_{nm}(\mathbf{k}, \mathbf{q}) \rvert^2
        \times [ &{} (n_\mathrm{po} + 1 - f_{m\mathbf{k} + \mathbf{q}})
        \delta(\varepsilon_{n\mathbf{k}} -  \varepsilon_{m\mathbf{k}+\mathbf{q}} - \hbar\omega_\mathrm{po}) \\
        &{} (n_\mathrm{po} + f_{m\mathbf{k} + \mathbf{q}})
        \delta(\varepsilon_{n\mathbf{k}} -  \varepsilon_{m\mathbf{k}+\mathbf{q}} + \hbar\omega_\mathrm{po})],
\end{aligned}
```

where $`\omega_\mathrm{po}`$ is an effective phonon frequency, 
$`n_\mathrm{po} = 1 / [\exp (\hbar \omega_\mathrm{po} / k_\mathrm{B} T) - 1]`$
denotes the Bose–Einstein distribution of phonons, and the
$`-\hbar \omega_\mathrm{po}`$ and 
$`+\hbar \omega_\mathrm{po}`$ terms correspond to scattering by phonon 
absorption and emission, respectively.

The overall inelastic scattering rate for state 
$`\mathinner{|n\mathbf{k}\rangle}`$ is calculated as

```math
\tau^{-1}_{n\mathbf{k}} = \sum_m \int \frac{\mathrm{d}^3q}{\Omega} 
\tau_{n\mathbf{k}\rightarrow m\mathbf{k}+\mathbf{q}}^{-1}
```

where $`\Omega`$ is the volume of the Brillouin zone.

### Elastic

Elastic rates are calculated using the *momentum relaxation time approximation*
(MRTA), given by

```math
\tilde{\tau}^{-1}_{n\mathbf{k}} = \sum_m \int \frac{\mathrm{d}^3q}{\Omega} 
    \left [ 1 - \frac{\mathbf{v}_{n\mathbf{k}} \cdot \mathbf{v}_{m\mathbf{k} + 
        \mathbf{q}}}{\lvert \mathbf{v}_{n\mathbf{k}} \rvert^2} \right ]

    \tilde{\tau}_{n\mathbf{k}\rightarrow m\mathbf{k}+\mathbf{q}}^{-1}
```
where $`\tilde{\tau}_{n\mathbf{k}\rightarrow m\mathbf{k}+\mathbf{q}}^{-1}`$ is
the elastic differential scattering rate defined at the top of this page and 
$`\mathbf{v}_{n\mathbf{k}}`$ is the group velocity of state 
$`\mathinner{|n\mathbf{k}\rangle}`$.

## Overlap integral

In the Born approximation, the scattering rate equations depend on the 
wavefunction overlap 
$`\mathinner{\langle m\mathbf{k}+\mathbf{q} \left | e^{i(\mathbf{q} + \mathbf{G})\cdot\mathbf{r}} \right | n\mathbf{k} \rangle}`$.
AMSET uses [pawpyseed](https://pypi.org/project/pawpyseed/) to obtain
wavefunction coefficients including PAW core regions from the pseudo wavefunction
coefficients written by VASP.
The wavefunctions coefficient are linearly interpolated onto the mesh used to
calculate scattering rates.

## Brillouin zone integration

All scattering rates depend on the Dirac delta function $`\delta`$,
which imposes conservation of energy. Due to finite k-point sampling and 
numerical noise, it is unlikely that this condition will ever be satisfied 
exactly. Furthermore, many scattering rates have a 
$`1 / \lvert\mathbf{q}\rvert ^2`$ dependence which
requires an extremely dense k-point mesh to achieve convergence.

To account for this, AMSET employs a modified tetrahedron integration
scheme. AMSET first identifies a constant energy surface by computing
tetrahedral cross sections using the tetrahedron method. Next, the constant 
energy surface is resampled using an ultra-fine mesh of k-points generated using
the [quadpy](https://github.com/nschloe/quadpy) numerical integration package.
The wavefunction coefficients and group velocities are reinterpolated into the 
ultra-fine mesh using linear interpolation and the matrix elements are 
calculated directly. This methodology allows for significantly faster 
convergence than the regular tetrahedron method.

The methodology for combining rates from multiple scattering mechanisms is given
in the [transport properties section](transport-properties.md).

[^Rode]: Rode, D. L. *Low-field electron transport. Semiconductors and semimetals* **10**, (Elsevier, 1975).

[^Shockley]: Shockley, W. & others. *Electrons and holes in semiconductors: with applications to transistor electronics.* (van Nostrand New York, 1950).

[^Bardeen]: Bardeen, J. & Shockley, W. *Deformation potentials and mobilities in non-polar crystals.* Phys. Rev. **80**, 72-80 (1950).

[^Dingle]: Dingle, R. B. *XCIV. Scattering of electrons and holes by charged donors and acceptors in semiconductors.* London, Edinburgh, Dublin Philos. Mag. J. Sci. **46**, 831-840 (1955).

[^Frohlich]: Fröhlich, H. *Electrons in lattice fields.* Adv. Phys. **3**, 325–361 (1954).

[^Conwell]: Conwell, E. M. *High Field Transport in Semiconductors.* Academic Press, New York (1967).

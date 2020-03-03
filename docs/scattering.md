# Calculating scattering rates

AMSET calculates mode dependent scattering rates within the Born approximation 
using common materials parameters. The differential scattering rate, 
$`s_{nm}(\mathbf{k}, \mathbf{k}^\prime)`$,
gives the rate from band $`n`$, k-point $`\mathbf{k}`$ to band $`m`$, k-point 
$`\mathbf{k}^\prime`$ .

The final scattering rate at each band and k-point is obtained by 
[integrating the differential scattering rates](#brillouin-zone-integration) 
over the full Brillouin zone.  In this section report the differential 
scattering rate equations and references for each scattering mechanism. More 
information on calculating transport properties is given in the 
[transport properties section](transport-properties.md).

## Summary of scattering rates

Mechanism                                                                               | Code  | Requires                                                                       | Type
---                                                                                     | ---   | ---                                                                            | ---
[Acoustic deformation potential scattering](#acoustic-deformation-potential-scattering) | ADP   | *n*- and *p*-type deformation potential,  elastic constant                     | Elastic
[Ionized impurity scattering](#ionized-impurity-scattering)                             | IMP   | static dielectric constant                                                     | Elastic
[Piezoelectric scattering](#piezoelectric-scattering)                                   | PIE   | static dielectric constant, piezoelectric coefficient                          | Elastic
[Polar optical phonon scattering](#polar-optical-phonon-scattering)                     | POP   | polar optical phonon frequency, static and high-frequency dielectric constants | Inelastic

### Acoustic deformation potential scattering

The acoustic deformation potential differential scattering rate is given by


```math
s_{nm}(\mathbf{k}, \mathbf{k}^\prime) =
    \frac{e^2 k_\mathrm{B}T E_\mathrm{d}^2}{4 \pi^2 \hbar C_\mathrm{el}}
    \lvert \mathinner{\langle{\psi_{m\mathbf{k}^\prime}|\psi_{n\mathbf{k}}}\rangle}\rvert^2 
    \delta ( E - E^\prime ),
```

```math
g_{nm}(\mathbf{k}, \mathbf{q}) =
    \left [ \frac{k_\mathrm{B} T E_\mathrm{d}^2}{C_\mathrm{el}} \right ] ^\frac{1}{2}
    \mathinner{\langle{\psi_{m\mathbf{k}+\mathbf{q}}|\psi_{n\mathbf{k}}}\rangle}
```


where $`E_\mathrm{d}`$ is the acoustic-phonon deformation-potential,
and $`C_\mathrm{el}`$ is the elastic constant.

!!! info "Acoustic deformation potential scattering information"
    - *Abbreviation:* APD
    - *Type:* Elastic
    - *References:* [^Bardeen], [^Shockley], [^Rode]
    - *Requires:* `deformation_potential`, `elastic_constant`

### Ionized impurity scattering

The ionized impurity differential scattering rate is given by

```math
s_{nm}(\mathbf{k}, \mathbf{k}^\prime) =
    \frac{e^4 N_\mathrm{imp}}{4 \pi^2 \hbar \epsilon_\mathrm{s}^2}
    \frac{\lvert \mathinner{\langle{\psi_{m\mathbf{k}^\prime}|\psi_{n\mathbf{k}}}\rangle}\rvert^2}
         {(\left | \mathbf{k} - \mathbf{k}^\prime \right | ^2 + \beta^2)^2}
    \delta ( E - E^\prime ),
```

```math
g_{nm}(\mathbf{k}, \mathbf{q}) =
    \left [ \frac{e^2 N_\mathrm{imp}}{\epsilon_\mathrm{s}^2} \right ] ^\frac{1}{2}
    \frac{\mathinner{\langle{\psi_{m\mathbf{k}+\mathbf{q}}|\psi_{n\mathbf{k}}}\rangle}}
         {\left | \mathbf{q} \right | ^2 + \beta^2}
```

where $`\epsilon_\mathrm{s}`$ is the static dielectric constant,
$`N_\mathrm{imp}`$ is the concentration of ionized impurities
(i.e., $`N_\mathrm{holes} + N_\mathrm{electrons}`$),
and $`\beta`$ is the inverse screening length, defined as

```math
    \beta^2 = \frac{e^2}{\epsilon_\mathrm{s}  k_\mathrm{B} T}
        \int (\mathbf{k} / \pi)^2 f(1-f) \,\mathrm{d}\mathbf{k}.
```

where $f$ is the Fermi dirac distribution given in the
[transport properties section](transport-properties.md).

!!! info "Ionized impurity scattering information"
    - *Abbreviation:* IMP
    - *Type:* Elastic
    - *References:* [^Dingle], [^Rode]
    - *Requires:* `static_dielectric`

### Piezoelectric scattering

The piezoelectric differential scattering rate is given by

```math
s_{nm}(\mathbf{k}, \mathbf{k}^\prime) =
    \frac{e^2 k_\mathrm{B} T P_\mathrm{pie}^2}{4 \pi \hbar \epsilon_\mathrm{s}}
    \frac{\lvert \mathinner{\langle{\psi_{m\mathbf{k}^\prime}|\psi_{n\mathbf{k}}}\rangle}\rvert^2}
         {\left | \mathbf{k} - \mathbf{k}^\prime \right | ^2 }
    \delta ( E - E^\prime ),
```

```math
g_{nm}(\mathbf{k}, \mathbf{q}) =
    \left [ \frac{k_\mathrm{B} T P_\mathrm{pie}^2}{\epsilon_\mathrm{s}} \right ] ^\frac{1}{2}
    \frac{\mathinner{\langle{\psi_{m\mathbf{k}+\mathbf{q}}|\psi_{n\mathbf{k}}}\rangle}}
         {\left | \mathbf{q} \right |}
```

where $`\epsilon_\mathrm{s}`$ is the static dielectric constant and
$`P_\mathrm{pie}`$ is the dimensionless piezoelectric coefficient.

!!! info "Piezoelectric scattering information"
    - *Abbreviation:* PIE
    - *Type:* Elastic
    - *References:* [^Rode]
    - *Requires:* `piezoelectric_coefficient`, `static_dielectric`

### Polar optical phonon scattering

The polar optical phonon differential scattering rate is given by

```math
\begin{aligned}
s_{nm}(\mathbf{k}, \mathbf{k}^\prime) =
    {}& \frac{e^2 \omega_\mathrm{po}}{8 \pi^2}
    \left (\frac{1}{\epsilon_\infty} - \frac{1}{\epsilon_\mathrm{s}}\right)
    \frac{\lvert \mathinner{\langle{\psi_{m\mathbf{k}^\prime}|\psi_{n\mathbf{k}}}\rangle}\rvert^2}
         {\lvert \mathbf{k} - \mathbf{k}^\prime \rvert ^2 } \\
    {}& \times \begin{cases}
        \delta ( E - E^\prime + \hbar \omega_\mathrm{po})(N_\mathrm{po} + 1), & \text{emission},\\
        \delta ( E - E^\prime - \hbar \omega_\mathrm{po})(N_\mathrm{po}), & \text{absorption},\\
     \end{cases}
\end{aligned}
```

```math
g_{nm}(\mathbf{k}, \mathbf{q}) =
    \left [ 
        \frac{\hbar \omega_\mathrm{po}}{2} 
        \left (\frac{1}{\epsilon_\infty} - \frac{1}{\epsilon_\mathrm{s}}\right)
    \right ] ^\frac{1}{2}
    \frac{\mathinner{\langle{\psi_{m\mathbf{k}+\mathbf{q}}|\psi_{n\mathbf{k}}}\rangle}}
         {\left | \mathbf{q} \right |}
```

where $`\omega_\mathrm{po}`$ is the polar optical phonon frequency,
$`\epsilon_\infty`$ is the high-frequency dielectric constant,
and $`N_\mathrm{po}`$ is the phonon density of states. The
$`-\hbar \omega_\mathrm{po}`$ and $`+\hbar \omega_\mathrm{po}`$ terms
correspond to scattering by phonon absorption and emission, respectively.

The phonon density of states is given by the Bose-Einstein distribution,
according to

```math
N_\mathrm{po} = \frac{1}{\exp (\hbar \omega_\mathrm{po} / k_\mathrm{B} T) - 1}.
```

!!! info "Polar optical phonon scattering information"
    - *Abbreviation:* POP
    - *Type:* Inelastic
    - *References:* [^Frohlich], [^Conwell], [^Rode]
    - *Requires:* `pop_frequency`, `static_dielectric`, `high_frequency_dielectric`

## Overlap integral

In the Born approximation, the scattering rate equations depend on the 
wavefunction overlap 
$`\mathinner{\langle{\psi_{m\mathbf{k}^\prime}|\psi_{n\mathbf{k}}}\rangle}`$.
AMSET uses [pawpyseed](https://pypi.org/project/pawpyseed/) to obtain
wavefunction coefficients including PAW core regions from the pseudo wavefunction
coefficients written by VASP.
The wavefunctions coefficients are linearly interpolated onto the mesh used to
calculate scattering rates.

## Brillouin zone integration

All scattering rates depend on the Dirac delta function, $`\delta(E - E^\prime)`$,
which imposes conservation of energy. Due to finite k-point sampling and 
numerical noise, it is unlikely that two states will ever have exactly the same
energy. Furthermore, many scattering rates have a 
$`1 / {\lvert \mathbf{k} - \mathbf{k}^\prime \rvert ^2 }`$ dependence which
requires extremely dense k-point meshes to achieve convergence.

To account for this, AMSET employs a modified tetrahedron integration
scheme. Similar to the traditional implementation of the tetrahedron method, 
tetrahedra cross sections representing regions of the constant energy surface are 
identified. If the area of these cross section is integrated using the
analytical expressions detailed by Blöchl, this will not satisfactorily account
for the strong k-point dependence at small 
$`{\lvert \mathbf{k} - \mathbf{k}^\prime \rvert ^2 }`$ values. In AMSET, 
we explicitly calculate the scattering rates on a ultra-fine mesh on the 
tetrahedron cross sections and integrate numerically. As the cross sections
represent a constant energy surface, the band structure does not need to be 
interpolated onto the ultra-fine mesh resulting in significant speed-ups.

The methodology for combining scattering rates for multiple scattering
mechanisms is given in the [transport properties section](transport-properties.md).

[^Rode]: Rode, D. L. *Low-field electron transport. Semiconductors and semimetals* **10**, (Elsevier, 1975).

[^Shockley]: Shockley, W. & others. *Electrons and holes in semiconductors: with applications to transistor electronics.* (van Nostrand New York, 1950).

[^Bardeen]: Bardeen, J. & Shockley, W. *Deformation potentials and mobilities in non-polar crystals.* Phys. Rev. **80**, 72-80 (1950).

[^Dingle]: Dingle, R. B. *XCIV. Scattering of electrons and holes by charged donors and acceptors in semiconductors.* London, Edinburgh, Dublin Philos. Mag. J. Sci. **46**, 831-840 (1955).

[^Frohlich]: Fröhlich, H. *Electrons in lattice fields.* Adv. Phys. **3**, 325–361 (1954).

[^Conwell]: Conwell, E. M. *High Field Transport in Semiconductors.* Academic Press, New York (1967).

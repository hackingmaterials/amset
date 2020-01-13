# Scattering rates

AMSET approximates electron scattering rates using common materials properties
in combination with information contained in a DFT band structure calculation.
For every scattering mechanism, the scattering rates are calculated at each
band and k-point in the band structure. This is achieved through the
differential scattering rate, $s_b(\mathbf{k}, \mathbf{k}^\prime)$, which
gives the rate of scattering from k-point, $\mathbf{k}$, to a second
k-point, $\mathbf{k}^\prime$, in band $b$.

The overall scattering rate for each k-point and
band is obtained by integrating the differential scattering rates over the
full Brillouin zone. Umklapp-type scattering is included by considering
periodic boundary conditions.

In this section we report the differential scattering rate equation and
references for each mechanism. More information about how carrier
transport properties are calculated is given in the [theory section](theory.md).

## Overview

Below, we give a summary of each mechanism, its abbreviation, and the material
parameters needed to calculate it.

Mechanism                                   | Code  | Requires                                  | Type
---                                         | ---   | ---                                       | ---
[Acoustic deformation potential scattering](#acoustic-deformation-potential-scattering) | ACD  | *n*- and *p*-type deformation potential,  elastic constant | Elastic
[Ionized impurity scattering](#ionized-impurity-scattering)               | IMP   | static dielectric constant                | Elastic
[Piezoelectric scattering](piezoelectric-scattering)                  | PIE   | static dielectric constant, piezoelectric coefficient | Elastic
[Polar optical phonon scattering](#polar-optical-phonon-scattering)            | POP    | polar optical phonon frequency, static and high-frequency dielectric constants | Inelastic

The differential scattering rate equations include
$G_b(\mathbf{k}, \mathbf{k}^\prime)$, the `overlap integral`,
and $\delta$, the [Dirac delta function](#brillouin-zone-integration).
A description of these factors is given at the bottom of this page.

## Acoustic deformation potential scattering

The acoustic deformation potential differential scattering rate is given by


$$
   s_b(\mathbf{k}, \mathbf{k}^\prime) =
        \frac{e^2 k_\mathrm{B}T E_\mathrm{d}^2}{4 \pi^2 \hbar C_\mathrm{el}}
        G_b(\mathbf{k}, \mathbf{k}^\prime) \delta ( E - E^\prime ),
$$


where $E_\mathrm{d}$ is the acoustic-phonon deformation-potential,
and $C_\mathrm{el}$ is the elastic constant.

!!! info "Acoustic deformation potential scattering information"
    - *Abbreviation:* ACD
    - *Type:* Elastic
    - *References:* [^Bardeen], [^Shockley], [^Rode]
    - *Requires:* `deformation_potential`, `elastic_constant`

## Ionized impurity scattering

The ionized impurity differential scattering rate is given by

$$
   s_b(\mathbf{k}, \mathbf{k}^\prime) =
        \frac{e^4 N_\mathrm{imp}}{4 \pi^2 \hbar \epsilon_\mathrm{s}^2}
        \frac{G_b(\mathbf{k}, \mathbf{k}^\prime)}
             {(\left | \mathbf{k} - \mathbf{k}^\prime \right | ^2 + \beta^2)^2}
        \delta ( E - E^\prime ),
$$

where $\epsilon_\mathrm{s}$ is the static dielectric constant,
$N_\mathrm{imp}$ is the concentration of ionized impurities
(i.e., $N_\mathrm{holes} + N_\mathrm{electrons}$),
and $\beta$ is the inverse screening length, defined as

$$
    \beta^2 = \frac{e^2}{\epsilon_\mathrm{s}  k_\mathrm{B} T}
        \int (\mathbf{k} / \pi)^2 f(1-f) \,\mathrm{d}\mathbf{k}.
$$

where $f$ is the Fermi dirac distribution given in the
[theory section](theory.md).

!!! info "Ionized impurity scattering information"
    - *Abbreviation:* IMP
    - *Type:* Elastic
    - *References:* [^Dingle], [^Rode]
    - *Requires:* `static_dielectric`

## Piezoelectric scattering

The piezoelectric differential scattering rate is given by

$$
   s_b(\mathbf{k}, \mathbf{k}^\prime) =
        \frac{e^2 k_\mathrm{B} T P_\mathrm{pie}^2}{4 \pi \hbar \epsilon_\mathrm{s}}
        \frac{G_b(\mathbf{k}, \mathbf{k}^\prime)}
             {\left | \mathbf{k} - \mathbf{k}^\prime \right | ^2 }
        \delta ( E - E^\prime ),
$$

where $\epsilon_\mathrm{s}$ is the static dielectric constant and
$P_\mathrm{pie}$ is the dimensionless piezoelectric coefficient.

!!! info "Piezoelectric scattering information"
    - *Abbreviation:* PIE
    - *Type:* Elastic
    - *References:* [^Rode]
    - *Requires:* `piezoelectric_coefficient`, `static_dielectric`

## Polar optical phonon scattering

The polar optical phonon differential scattering rate is given by

$$
  \begin{align}
   s_b(\mathbf{k}, \mathbf{k}^\prime) =
        {}& \frac{e^2 \omega_\mathrm{po}}{8 \pi^2}
        \left (\frac{1}{\epsilon_\infty} - \frac{1}{\epsilon_\mathrm{s}}\right)
        \frac{G_b(\mathbf{k}, \mathbf{k}^\prime)}
             {\left | \mathbf{k} - \mathbf{k}^\prime \right | ^2 } \\
        {}& \times \begin{cases}
            \delta ( E - E^\prime + \hbar \omega_\mathrm{po})(N_\mathrm{po} + 1), & \text{emission},\\
            \delta ( E - E^\prime - \hbar \omega_\mathrm{po})(N_\mathrm{po}), & \text{absorption},\\
         \end{cases}
  \end{align}
$$

where $\omega_\mathrm{po}$ is the polar optical phonon frequency,
$\epsilon_\infty$ is the high-frequency dielectric constant,
and $N_\mathrm{po}$ is the phonon density of states. The
$-\hbar \omega_\mathrm{po}$ and $+\hbar \omega_\mathrm{po}$ terms
correspond to scattering by phonon absorption and emission, respectively.

The phonon density of states is given by the Bose-Einstein distribution,
according to

$$
    N_\mathrm{po} = \frac{1}{\exp (\hbar \omega_\mathrm{po} / k_\mathrm{B} T) - 1}.
$$

!!! info "Polar optical phonon scattering information"
    - *Abbreviation:* POP
    - *Type:* Inelastic
    - *References:* [^Frohlich], [^Conwell], [^Rode]
    - *Requires:* `pop_frequency`, `static_dielectric`, `high_frequency_dielectric`

Overlap integral
----------------

Each differential scattering rate equation depends on the integral overlap,
$G_b(\mathbf{k}, \mathbf{k}^\prime)$, which gives the degree of
orbital overlap between a k-point, $\mathbf{k}$ and a second
k-point, $\mathbf{k}^\prime$, in band $b$.

In general, calculating the overlap integral between two k-points requires
access to the wavefunctions of the states of interest. For interpolated band
structures, this poses a problem as the wavefunctions now must also be interpolated.

In AMSET, we use an approximation for the orbital integral based on the projected
orbital contributions. Currently, a simple expression for the overlap is used,
however, in future releases a more sophisticated expression will be developed.
The orbital integral is implemented as

$$
    G_b(\mathbf{k}, \mathbf{k}^\prime)
        = (a_{b,\mathbf{k}} a_{b,\mathbf{k^\prime}}
          + c_{b,\mathbf{k}} c_{b,\mathbf{k^\prime}} x )^2,
$$

where $x$ is the cosine of the angle between $\mathbf{k}$
and $\mathbf{k}^\prime)$, and $a$ and $c$ depend
on the *s* ($\phi_s$) and *p*-orbital projections ($\phi_p$)
as:

$$
\begin{align}
    a_{b,\mathbf{k}} = {}& \frac{\phi_{s,b,\mathbf{k}}}
                            {\sqrt{(\phi_{s,b,\mathbf{k}}^2 +
                              \phi_{p,b,\mathbf{k}}^2)}}, \\
    c_{b,\mathbf{k}} = {}& \sqrt{1 - a_{b,\mathbf{k}}^2}.
\end{align}
$$

The justification for the above form of the overlap integral is given in [^Rode].

Brillouin zone integration
--------------------------

All scattering rate equations depend on the Dirac delta function,
$\delta(E - E^\prime)$, which is 1 when the energy of the two states
$E$ and $E^\prime$ are equal and 0 otherwise.

Due to finite k-point sampling and numerical noise, it is unlikely that two
states will ever have exactly the same energy. To account for this, we replace
the Dirac function with a Gaussian distribution, according to

$$
    \frac{1}{\sigma \sqrt{2 \pi}} \exp{ \left ( \frac{E - E^\prime}{\sigma} \right )^2}
$$

where $\sigma$ is the broadening width.

The overall scattering rate at k-point, $\mathbf{k}$, and band, $b$,
can therefore be calculated as a discrete summation over all k-points in the
Brillouin zone. I.e.,


$$
    s_b(\mathbf{k}) = \frac{\Omega}{N_\mathrm{kpts}}
        \sum_{\mathbf{k} \neq \mathbf{k}^\prime}^{\mathbf{k}^\prime}
        s_b(\mathbf{k}, \mathbf{k}^\prime),
$$

where $N_\mathrm{kpts}$ is the total number of k-points in the full
Brillouin zone and $\Omega$ is the reciprocal lattice volume.

The methodology for combining the scattering rates for multiple scattering
mechanisms is given in the [theory section](theory.md).

[^Rode]: Rode, D. L. *Low-field electron transport. Semiconductors and semimetals* **10**, (Elsevier, 1975).

[^Shockley]: Shockley, W. & others. *Electrons and holes in semiconductors: with applications to transistor electronics.* (van Nostrand New York, 1950).


[^Bardeen]: Bardeen, J. & Shockley, W. *Deformation potentials and mobilities in non-polar crystals.* Phys. Rev. **80**, 72-80 (1950).

[^Dingle]: Dingle, R. B. *XCIV. Scattering of electrons and holes by charged donors and acceptors in semiconductors.* London, Edinburgh, Dublin Philos. Mag. J. Sci. **46**, 831-840 (1955).


[^Frohlich]: Fröhlich, H. *Electrons in lattice fields.* Adv. Phys. **3**, 325–361 (1954).

[^Conwell]: Conwell, E. M. *High Field Transport in Semiconductors.* Academic Press, New York (1967).

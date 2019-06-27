Scattering rates
================

AMSET approximates electron scattering rates using common materials properties
in combination with information contained in a DFT band structure calculation.
For every scattering mechanism, the scattering rates are calculated at each
band and k-point in the band structure. This is achieved through the
differential scattering rate, :math:`s_b(\mathbf{k}, \mathbf{k}^\prime)`, which
gives the rate of scattering from k-point, :math:`\mathbf{k}`, to a second
k-point, :math:`\mathbf{k}^\prime`, in band :math:`b`.

The overall scattering rate for each k-point and
band is obtained by integrating the differential scattering rates over the
full Brillouin zone. Umklapp-type scattering is included by considering
periodic boundary conditions.

**Note:** *currently, AMSET only models intraband scattering. Interband
scattering will likely be added in a future release.*

In this section we report the differential scattering rate equation and
references for each mechanism. More information about how carrier
transport properties are calculated is given in the `theory section <theory>`_.

Overview
--------

Below, we give a summary of each mechanism, its abbreviation, and the material
parameters needed to calculate it.

============================================  =====  =========================================  =========
Mechanism                                     Code   Requires                                   Type
============================================  =====  =========================================  =========
`Acoustic deformation potential scattering`_  ACD    *n*- and *p*-type deformation potential,   Elastic
                                                     elastic constant
`Ionized impurity scattering`_                IMP    static dielectric constant                 Elastic
`Piezoelectric scattering`_                   PIE    static dielectric constant, piezoelectric  Elastic
                                                     coefficient.
`Polar optical phonon scattering`_            POP    polar optical phonon frequency, static     Inelastic
                                                     and high-frequency dielectric constants
============================================  =====  =========================================  =========

The differential scattering rate equations include
:math:`G_b(\mathbf{k}, \mathbf{k}^\prime)`, the `overlap integral`_,
and :math:`\delta`, the `Dirac delta function <#brillouin-zone-integration>`_.
A description of these factors is given at the bottom of this page.

Acoustic deformation potential scattering
-----------------------------------------

The acoustic deformation potential differential scattering rate is given by

.. math::

   s_b(\mathbf{k}, \mathbf{k}^\prime) =
        \frac{e^2 k_\mathrm{B}T E_\mathrm{d}^2}{4 \pi^2 \hbar C_\mathrm{el}}
        G_b(\mathbf{k}, \mathbf{k}^\prime) \delta ( E - E^\prime ),


where :math:`E_\mathrm{d}` is the acoustic-phonon deformation-potential,
and :math:`C_\mathrm{el}` is the elastic constant.

Notes
~~~~~

- *Abbreviation:* ACD
- *Type:* Elastic
- *References:* [Bardeen]_, [Shockley]_, [Rode]_
- *Requires:* ``deformation_potential``, ``elastic_constant``

Ionized impurity scattering
---------------------------

The ionized impurity differential scattering rate is given by

.. math::

   s_b(\mathbf{k}, \mathbf{k}^\prime) =
        \frac{e^4 N_\mathrm{imp}}{4 \pi^2 \hbar \epsilon_\mathrm{s}^2}
        \frac{G_b(\mathbf{k}, \mathbf{k}^\prime)}
             {(\left | \mathbf{k} - \mathbf{k}^\prime \right | ^2 + \beta^2)^2}
        \delta ( E - E^\prime ),

where :math:`\epsilon_\mathrm{s}` is the static dielectric constant,
:math:`N_\mathrm{imp}` is the concentration of ionized impurities
(i.e., :math:`N_\mathrm{holes} + N_\mathrm{electrons}`),
and :math:`\beta` is the inverse screening length, defined as

.. math::

    \beta^2 = \frac{e^2}{\epsilon_\mathrm{s}  k_\mathrm{B} T}
        \int (\mathbf{k} / \pi)^2 f(1-f) \,\mathrm{d}\mathbf{k}.

where :math:`f` is the Fermi dirac distribution given in the
`theory section <theory>`_.

Notes
~~~~~

- Abbreviation: IMP
- Type: Elastic
- References: [Dingle]_, [Rode]_
- Requires: ``static_dielectric``

Piezoelectric scattering
------------------------

The piezoelectric differential scattering rate is given by

.. math::

   s_b(\mathbf{k}, \mathbf{k}^\prime) =
        \frac{e^2 k_\mathrm{B} T P_\mathrm{pie}^2}{4 \pi \hbar \epsilon_\mathrm{s}}
        \frac{G_b(\mathbf{k}, \mathbf{k}^\prime)}
             {\left | \mathbf{k} - \mathbf{k}^\prime \right | ^2 }
        \delta ( E - E^\prime ),

where :math:`\epsilon_\mathrm{s}` is the static dielectric constant and
:math:`P_\mathrm{pie}` is the dimensionless piezoelectric coefficient.

Notes
~~~~~

- Abbreviation: PIE
- Type: Elastic
- References: [Rode]_
- Requires: ``piezoelectric_coefficient``, ``static_dielectric``

Polar optical phonon scattering
-------------------------------

The polar optical phonon differential scattering rate is given by

.. math::

   s_b(\mathbf{k}, \mathbf{k}^\prime) =
        {}& \frac{e^2 \omega_\mathrm{po}}{8 \pi^2}
        \left (\frac{1}{\epsilon_\infty} - \frac{1}{\epsilon_\mathrm{s}}\right)
        G_b(\mathbf{k}, \mathbf{k}^\prime) \\
        {}& \times \begin{cases}
            \delta ( E - E^\prime + \hbar \omega_\mathrm{po})(N_\mathrm{po} + 1), & \text{emission},\\
            \delta ( E - E^\prime - \hbar \omega_\mathrm{po})(N_\mathrm{po}), & \text{absorption},\\
         \end{cases}

where :math:`\omega_\mathrm{po}` is the polar optical phonon frequency,
:math:`\epsilon_\infty` is the high-frequency dielectric constant,
and :math:`N_\mathrm{po}` is the phonon density of states. The
:math:`-\hbar \omega_\mathrm{po}` and :math:`+\hbar \omega_\mathrm{po}` terms
correspond to scattering by phonon absorption and emission, respectively.

The phonon density of states is given by the Bose-Einstein distribution,
according to

.. math::

    N_\mathrm{po} = \frac{1}{\exp (\hbar \omega_\mathrm{po} / k_\mathrm{B} T) - 1}.

Notes
~~~~~

- Abbreviation: POP
- Type: Inelastic
- References: [Frohlich]_, [Conwell]_, [Rode]_
- Requires: ``pop_frequency``, ``static_dielectric``, ``high_frequency_dielectric``

Overlap integral
----------------

Each differential scattering rate equation depends on the integral overlap,
:math:`G_b(\mathbf{k}, \mathbf{k}^\prime)`, which gives the degree of
orbital overlap between a k-point, :math:`\mathbf{k}` and a second
k-point, :math:`\mathbf{k}^\prime`, in band :math:`b`.

In general, calculating the overlap integral between two k-points requires
access to the wavefunctions of the states of interest. For interpolated band
structures, this poses a problem as the wavefunctions now must also be interpolated.

In AMSET, we use an approximation for the orbital integral based on the projected
orbital contributions. Currently, a simple expression for the overlap is used,
however, in future releases a more sophisticated expression will be developed.
The orbital integral is implemented as

.. math::

    G_b(\mathbf{k}, \mathbf{k}^\prime)
        = (a_{b,\mathbf{k}} a_{b,\mathbf{k^\prime}}
          + c_{b,\mathbf{k}} c_{b,\mathbf{k^\prime}} x )^2,

where :math:`x` is the cosine of the angle between :math:`\mathbf{k}`
and :math:`\mathbf{k}^\prime)`, and :math:`a` and :math:`c` are depend
on the *s* (:math:`\phi_s`) and *p*-orbital projections (:math:`\phi_p`)
as:

.. math::

    a_{b,\mathbf{k}} = {}& \frac{\phi_{s,b,\mathbf{k}}}
                            {\sqrt{(\phi_{s,b,\mathbf{k}}^2 +
                              \phi_{p,b,\mathbf{k}}^2)}}, \\
    c_{b,\mathbf{k}} = {}& \sqrt{1 - a_{b,\mathbf{k}}^2}.

The justification for the above form of the overlap integral is given in [Rode]_.

Brillouin zone integration
--------------------------

All scattering rate equations depend on the Dirac delta function,
:math:`\delta(E - E^\prime)`, which is 1 when the energy of the two states
:math:`E` and :math:`E^\prime` are equal and 0 otherwise.

Due to finite k-point sampling and numerical noise, it is unlikely that two
states will ever have exactly the same energy. To account for this, we replace
the Dirac function with a Gaussian distribution, according to

.. math::

    \frac{1}{\sigma \sqrt{2 \pi}} \exp{ \left ( \frac{E - E^\prime}{\sigma} \right )^2}

where :math:`\sigma` is the broadening width.

The overall scattering rate at k-point, :math:`\mathbf{k}`, and band, :math:`b`,
can therefore be calculated as a discrete summation over all k-points in the
Brillouin zone. I.e.,

.. math::

    s_b(\mathbf{k}) = \frac{\Omega}{N_\mathrm{kpts}}
        \sum_{\mathbf{k} \neq \mathbf{k}^\prime}^{\mathbf{k}^\prime}
        s_b(\mathbf{k}, \mathbf{k}^\prime),

where :math:`N_\mathrm{kpts}` is the total number of k-points in the full
Brillouin zone and :math:`\Omega` is the reciprocal lattice volume.

The methodology for combining the scattering rates for multiple scattering
mechanisms is given in the `theory section <theory>`_.

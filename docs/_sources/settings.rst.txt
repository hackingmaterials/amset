Settings
========

Whether using AMSET via the command-line or python API, the primary controls
are contained in the settings file or dictionary. An example AMSET settings file
is given :ref:`here <example-settings>`.

The settings are grouped into sections. The description for each section and
settings parameter is given below.

All settings are also controllable via command-line flags. The corresponding
command-line interface options are also detailed below. Any settings specified
via the command-line will override those in the settings file.

.. contents::
   :local:
   :backlinks: None

General settings
----------------

The ``general`` section contains the main settings that control the AMSET run,
including interpolation settings and temperature/doping ranges.


``doping``
~~~~~~~~~~

*Command-line option:* ``-d, --doping``

Controls which doping levels (carrier concentrations) to calculate.
Concentrations given with the unit cm\ :sup:`–3`. Can be specified directly as a comma
separated list, e.g.::

    1E13, 1E14, 1E15, 1E16

Alternatively, ranges can be specified using the syntax ``start:stop:num_steps``,
e.g.::

    1E13:1E20:8

Negative concentrations indicate holes (*p*-type doping), positive concentrations
indicate electrons (*n*-type doping).


``temperatures``
~~~~~~~~~~~~~~~~

*Command-line option:* ``-d, --temperatures``

Controls which temperatures to calculate. Temperatures given in Kelvin. Can be
specified directly as a comma separated list, e.g.::

    300,400,500

Alternatively, ranges can be specified using the syntax ``start:stop:num_steps``,
e.g.::

    300:1000:8


``interpolation_factor``
~~~~~~~~~~~~~~~~~~~~~~~~

*Command-line option:* ``-i, --interpolation-factor``

Interpolation factor controlling the interpolation density. Larger numbers
indicate greater density. The number of k-points in the interpolated band
structure will be roughly equal to ``interpolation_factor`` times the number
of k-points in the DFT calculation. This is the primary option for controlling
the accuracy of the calculated scattering rates. **Transport properties should
be converged with respect to this parameter.**

``num_extra_kpoints``
~~~~~~~~~~~~~~~~~~~~~

*Command-line option:* ``-n, --num-extra-kpoints``

Number of additional k-points to add near the Fermi level in regions of k-space
with large band velocities. Densifying the band structure at these points can
often improve the speed of convergence for very disperse band structures.

When extra k-points are added, the k-point weights are determined by: i)
calculating the Voronoi tessellation for the k-points, ii) calculating the
Voronoi cell volumes. This calculation can add some additional computational
expense.

See `fd_tol`_ for controlling where to add the extra k-points.

``scattering_type``
~~~~~~~~~~~~~~~~~~~

*Command-line option:* ``-s, --scattering-type``

Which scattering mechanisms to calculation. If set to ``auto``, the scattering
mechanisms will automatically be determined based on the specified material
parameters. Alternatively, a comma separated list of scattering mechanism
can be specified. Options include:

- ``ACD`` (acoustic deformation potential scattering)
- ``IMP`` (ionized impurity scattering)
- ``PIE`` (piezoelectric scattering)
- ``POP`` (polar optical phonon scattering)

For example, ``ACD,IMP,POP``. The scattering mechanism will only be calculated
if all the required material parameters for that mechanism are set. See the
`scattering section <scattering>`_ of the documentation for more details.

``scissor``
~~~~~~~~~~~

*Command-line option:* ``-s, --scissor``

The amount to scissor the band gap, in eV. Positive values indicate band gap
opening, negative values indicate band gap narrowing. Has no effect for metallic
systems.

``bandgap``
~~~~~~~~~~~

*Command-line option:* ``-b, --bandgap``

Set the band gap to this value, in eV. Will automatically determine and apply the
correct band gap scissor for the specified band gap. Cannot be used in
combination with the `scissor`_  option. Has no effect for metallic systems.


Material settings
-----------------

The ``material`` section holds all materials properties required to calculate
the scattering rates.

``high_frequency_dielectric``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Command-line option:* ``--high-frequency-dielectric``

The high-frequency dielectric constant, in units of :math:`\epsilon_0`.

*Required for:* POP

``static_dielectric``
~~~~~~~~~~~~~~~~~~~~~

*Command-line option:* ``--static-dielectric``

The static dielectric constant, in units of :math:`\epsilon_0`.

*Required for:* IMP, PIE, POP

``elastic_constant``
~~~~~~~~~~~~~~~~~~~~

*Command-line option:* ``--elastic-constant``

The direction averaged elastic constant, in GPa.

*Required for:* ACD

``deformation_potential``
~~~~~~~~~~~~~~~~~~~~~~~~~

*Command-line option:* ``--deformation-potential``

The volume deformation potential, in eV. Can be given as a comma separated
list of two values for the VBM and CBM, respectively, e.g.::

    8.6, 7.4

Or a single value to use for all bands in metals.

*Required for:* ACD

``piezoelectric_coefficient``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*command-line option:* ``--piezoelectric-coefficient``

The direction averaged piezoelectric coefficient (unitless).

*Required for:* PIE

``acceptor_charge``
~~~~~~~~~~~~~~~~~~~

*Command-line option:* ``--acceptor-charge``

The charge of acceptor defects, in units of electron charge.

*Required for:* IMP

``donor_charge``
~~~~~~~~~~~~~~~~

*Command-line option:* ``--donor-charge``

The charge of donor defects, in units of electron charge.

*Required for:* IMP

``pop_frequency``
~~~~~~~~~~~~~~~~~

*Command-line option:* ``--pop-frequency``

The polar optical phonon frequency, in THz. Generally, it is ok to take the
highest optical phonon frequency at the Gamma point.

*Required for:* POP

Performance settings
--------------------

The ``performance`` section controls internal AMSET settings that will affect
the speed and accuracy of calculated properties.

``gauss_width``
~~~~~~~~~~~~~~~

*Command-line option:* ``--gauss-width``

The gaussian width (sigma) that is used to approximate the delta function when
calculating scattering rates, in eV. Larger values will lead to scattering
between greater numbers of k-points, leading to an artificial increase in the
scattering rate. Smaller values require denser k-point meshes to converge the
scattering rate. In general, the default value of 0.001 eV is acceptable in
most cases.

``energy_cutoff``
~~~~~~~~~~~~~~~~~

*Command-line option:* ``--energy-cutoff``

The energy cut-off used to determine which bands to include in the interpolation
and scattering rate calculation, in eV.

``fd_tol``
~~~~~~~~~~

*Command-line option:* ``--fd-tol``

The Fermi–Dirac derivative tolerance that controls where extra k-points are
added. Given as a percentage from 0 to 1. Larger values indicate that the
k-points will be added to regions of the band structure with large Fermi–Dirac
derivative contributions. Smaller values will spread the k-points over more of
the Brillouin zone.

Additional details of the densification process will be added soon.

``ibte_tol``
~~~~~~~~~~~~

*Command-line option:* ``--ibte-tol``

Parameter to control when the iterative Boltzmann transport equation is
considered converged. Given as a percent from 0 to 1.

**Not yet tested**

``max_ibte_iter``
~~~~~~~~~~~~~~~~~

*Command-line option:* ``--max-ibte-iter``

Maximum number of iterations for solving the iterative Boltzmann transport
equation.

**Not yet tested**

``dos_estep``
~~~~~~~~~~~~~

*Command-line option:* ``--dos-estep``

The energy step for the calculated density of states, in eV. Controls the
accuracy when determining the position of the Fermi level.

``dos_width``
~~~~~~~~~~~~~

*Command-line option:* ``--dos-width``

Broadening width by which to smear the density of states, in eV. It is
recommended to leave this as ``None``, e.g., no broadening.

``symprec``
~~~~~~~~~~~

*Command-line option:* ``--symprec``

The symmetry finding tolerance, in Å.

``nworkers``
~~~~~~~~~~~~

*Command-line option:* ``--nworkers``

Number of processors to use.

Output settings
---------------

The output section controls the output files and logging.

``calculate_mobility``
~~~~~~~~~~~~~~~~~~~~~~

*Command-line option:* ``--no-calculate-mobility``

Whether to calculate *n*- and *p*-type carrier mobilities. Has no effect
for metallic systems where mobility is not well defined.

``separate_scattering_mobilities``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Command-line option:* ``--no-separate-scattering-mobilities``

Whether to report the individual scattering rate mobilities. I.e., the mobility
if only that scattering mechanism were present.

``file_format``
~~~~~~~~~~~~~~~

*Command-line option:* ``--file-format``

The output file format. Options are: ``json``, ``yaml``, and ``txt``.

Default is ``json``.


``write_input``
~~~~~~~~~~~~~~~

*Command-line option:* ``--write-input``

Whether to write the input settings to a file called ``amset_settings.yaml``.

``write_mesh``
~~~~~~~~~~~~~~

*Command-line option:* ``--write-mesh``

Whether to write the full k-dependent properties to disk. Properties include
the band energy, velocity and scattering rate.

**Note:** for large values of `interpolation_factor`_ this option can use a large
amount of disk space.

``log_error_traceback``
~~~~~~~~~~~~~~~~~~~~~~~

*Command-line option:* ``--log-error-traceback``

Whether to log the full error traceback rather than just the error message. If
you find a problem with AMSET, please enable this option and provide the AMSET
developers with the full crash report.

``print_log``
~~~~~~~~~~~~~

*Command-line option:* ``--no-log``

Whether to print log messages.

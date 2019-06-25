.. title:: Settings

Settings
========

Whether using AMSET via the command-line or python API, the primary controls
are contained in the settings file. An example AMSET settings file is given
:doc:`here </example_settings.yaml>`.

The settings are grouped into sections. The description for each section and
settings parameter is given below.

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


``temperatures``
~~~~~~~~~~~~~~~

*Command-line option:* ``-d, --temperatures``


``interpolation_factor``
~~~~~~~~~~~~~~~~~~~~~~~~

*Command-line option:* ``-i, --interpolation-factor``

Something about interpolation factor

``num_extra_kpoints``
~~~~~~~~~~~~~~~~~~~~~

*Command-line option:* ``-n, --num-extra-kpoints``

``scattering_type``
~~~~~~~~~~~~~~~~~~~

*Command-line option:* ``-s, --scattering-type``

``scissor``
~~~~~~~~~~~

*Command-line option:* ``-s, --scissor``

``bandgap``
~~~~~~~~~~~

*Command-line option:* ``-b, --bandgap``


Material settings
-----------------

The ``material`` section holds all materials properties required to calculate
the scattering rates.

``high_frequency_dielectric``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Command-line option:* ``--high-frequency-dielectric``

``static_dielectric``
~~~~~~~~~~~~~~~~~~~~~

*Command-line option:* ``--static-dielectric``

``elastic_constant``
~~~~~~~~~~~~~~~~~~~~

*Command-line option:* ``--elastic-constant``

``deformation_potential``
~~~~~~~~~~~~~~~~~~~~~~~~~

*Command-line option:* ``--deformation-potential``

``piezoelectric_coefficient``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*command-line option:* ``--piezoelectric-coefficient``

``acceptor_charge``
~~~~~~~~~~~~~~~~~~~

*Command-line option:* ``--acceptor-charge``

``donor_charge``
~~~~~~~~~~~~~~~~

*Command-line option:* ``--donor-charge``

``pop_frequency``
~~~~~~~~~~~~~~~~~

*Command-line option:* ``--pop-frequency``


Performance settings
--------------------

The ``performance`` section controls internal AMSET settings that will affect
the speed and accuracy of the results.

``gauss_width``
~~~~~~~~~~~~~~~

*Command-line option:* ``--gauss-width``

``energy_cutoff``
~~~~~~~~~~~~~~~~~

*Command-line option:* ``--energy-cutoff``

``fd_tol``
~~~~~~~~~~

*Command-line option:* ``--fd-tol``

``g_tol``
~~~~~~~~~

*Command-line option:* ``--g-tol``

``max_g_iter``
~~~~~~~~~~~~~~

*Command-line option:* ``--max-g-iter``

``dos_estep``
~~~~~~~~~~~~~

*Command-line option:* ``--dos-estep``

``dos_width``
~~~~~~~~~~~~~

*Command-line option:* ``--dos-width``

``symprec``
~~~~~~~~~~~

*Command-line option:* ``--symprec``

``nworkers``
~~~~~~~~~~~~

*Command-line option:* ``--nworkers``



Output settings
---------------

The output section controls the output files and logging.

``calculate_mobility``
~~~~~~~~~~~~~~~~~~~~~~

*Command-line option:* ``--no-calculate-mobility``

``separate_scattering_mobilities``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Command-line option:* ``--no-separate-scattering-mobilities``

``file_format``
~~~~~~~~~~~~~~~

*Command-line option:* ``--file-format``

``write_input``
~~~~~~~~~~~~~~~

*Command-line option:* ``--write-input``

``write_mesh``
~~~~~~~~~~~~~~

*Command-line option:* ``--write-mesh``

``log_error_traceback``
~~~~~~~~~~~~~~~~~~~~~~~

*Command-line option:* ``--log-error-traceback``

``print_log``
~~~~~~~~~~~~~

*Command-line option:* ``--no-log``

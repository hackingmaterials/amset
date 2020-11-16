# Settings

Whether using AMSET via the command-line or python API, the primary controls
are contained in the settings file or dictionary. An example AMSET settings file
is given [here](https://github.com/hackingmaterials/amset/blob/master/examples/GaAs/settings.yaml).

The settings are grouped into sections. The description for each section and
settings parameter is given below.

All settings are also controllable via command-line flags. The corresponding
command-line interface options are also detailed below. Any settings specified
via the command-line will override those in the settings file.


## General settings

These settings control the AMSET run, including interpolation density and 
temperature/doping ranges.

### `doping`

!!! quote ""
    *Command-line option:* `-d, --doping`

    Controls which doping levels (carrier concentrations) to calculate.
    Concentrations given with the unit cm<sup>–3</sup>. Can be specified directly as a comma
    separated list, e.g.:

    ```python
    1E13,1E14,1E15,1E16,1E17,1E18,1E19,1E20
    ```

    Alternatively, ranges can be specified using the syntax `start:stop:num_steps`.
    For example, the same doping concentrations as above can be written::

    ```python
    1E13:1E20:8
    ```

    Negative concentrations indicate electrons (*n*-type doping), positive concentrations
    indicate holes (*p*-type doping).
    
    Default: `{{ doping }}`

### `temperatures`

!!! quote ""
    *Command-line option:* `-t, --temperatures`

    Controls which temperatures to calculate. Temperatures given in Kelvin. Can be
    specified directly as a comma separated list, e.g.::

    ```python
    300,400,500,600,700,800,900,1000
    ```

    Alternatively, ranges can be specified using the syntax `start:stop:num_steps`,
    For example, the same temperature range as above can be written::

    ```python
    300:1000:8
    ```
    
    Default: `{{ temperatures }}`

### `interpolation_factor`

!!! quote ""
    *Command-line option:* `-i, --interpolation-factor`

    Interpolation factor controlling the interpolation density. Larger numbers
    indicate greater density. The number of k-points in the interpolated band
    structure will be roughly equal to `interpolation_factor` times the number
    of k-points in the DFT calculation. This is the primary option for controlling
    the accuracy of the calculated scattering rates. **Transport properties should
    be converged with respect to this parameter.**

    Default: `{{ interpolation_factor }}`

### `scattering_type`

!!! quote ""
    *Command-line option:* `-s, --scattering-type`

    Which scattering mechanisms to calculate. If set to `auto`, the scattering
    mechanisms will automatically be determined based on the specified material
    parameters. Alternatively, a comma separated list of scattering mechanism
    can be specified. Options include:

    - `ACD` (acoustic deformation potential scattering)
    - `IMP` (ionized impurity scattering)
    - `PIE` (piezoelectric scattering)
    - `POP` (polar optical phonon scattering)
    - `CRT` (constant relaxation time)
    - `MFP` (mean free path scattering)

    For example, `ACD,IMP,POP`. The scattering mechanism will only be calculated
    if all the required material parameters for that mechanism are set. See the
    `scattering section <scattering>`_ of the documentation for more details.

    Default: `{{ scattering_type }}`

### `wavefunction_coefficients`

!!! quote ""
    *Command-line option:* `-w, --wavefunction-coefficients`

    Path to wavefunction coefficients file. The coefficients can be extracted
    from a VASP WAVECAR using the command:
   
    ```bash
    amset wave
    ``` 

    This command also requires the vasprun.xml to be in the same folder.

    Default: `{{ wavefunction_coefficients }}`
    
### `use_projections`

!!! quote ""
    *Command-line option:* `--use-projections`
    
    Use projections to calculate wavefunction overlap. This can often result in very
    poor performance, and so is **not recommended**. 
    
    In order to use projections, the VASP calculation must be performed with
    `LORBIT = 11`. 

    Default: `{{ use_projections }}`
    
### `scissor`

!!! quote ""
    *Command-line option:* `-s, --scissor`

    The amount to scissor the band gap, in eV. Positive values indicate band gap
    opening, negative values indicate band gap narrowing. Has no effect for metallic
    systems.

### `bandgap`

!!! quote ""
    *Command-line option:* `-b, --bandgap`

    Set the band gap to this value, in eV. Will automatically determine and apply the
    correct band gap scissor for the specified band gap. Cannot be used in
    combination with the [`scissor`](#scissor) option. Has no effect for metallic systems.

### `zero_weighted_kpoints`

!!! quote ""
    *Command-line option:* `-z, --zero-weighted-kpoints`

    How to handle "zero-weighted" k-points if they are present in the calculation. 
    Options are:
    
    - keep: Keep zero-weighted k-points in the band structure.
    - drop: Drop zero-weighted k-points, keeping only the weighted k-points.
    - prefer: Drop weighted-kpoints if zero-weighted k-points are present
      in the calculation (useful for cheap hybrid calculations).

### `free_carrier_screening`

!!! quote ""
    *Command-line option:* `--free-carrier-screening`
    
    Whether free carriers will screen polar optical phonon and piezoelectric
    scattering rates. This modifies the matrix elements from a 
    $`\frac{1}{\left | \mathbf{q} + \mathbf{G} \right |}`$
    dependence to a 
    $`\frac{1}{\left | \mathbf{q} + \mathbf{G} \right | + \beta_\infty}`$
    dependence, where $`\beta_\infty`$ is the inverse screening length that depends
    on the temperature, carrier concentration, and high-frequency dielectric constant.
    
    This can result in a large reduction in the scattering rates at high carrier
    concentrations.
    
    Default: `{{ free_carrier_screening }}`
    

## Material settings

These settings control the materials properties required to calculate the 
scattering rates.

### `high_frequency_dielectric`

!!! quote ""
    *Command-line option:* `--high-frequency-dielectric`

    The high-frequency dielectric constant, in units of $`\epsilon_0`$.
    Can be given as a 3x3 tensor or a single isotropic value.

    Required for: POP, PIE

### `static_dielectric`

!!! quote ""
    *Command-line option:* `--static-dielectric`

    The static dielectric constant, in units of $`\epsilon_0`$.
    Can be given as a 3x3 tensor or a single isotropic value.

    Required for: IMP, POP

### `elastic_constant`

!!! quote ""
    *Command-line option:* `--elastic-constant`

    The elastic constants as the full 3x3x3x3 tensor or 6x6 Voigt form, in GPa.
    
    Alteratively, a single averaged value can be given (not recommended).

    Required for: ACD, PIE

### `deformation_potential`

!!! quote ""
    *Command-line option:* `--deformation-potential`
    
    Path to file containing deformation potentials for all bands, generated
    using `amset deform read`.
    
    Alternatively, Can be given as a comma separated list of two deformation potentials
    for the VBM and CBM, respectively in eV, e.g.:

    ```python
    8.6, 7.4
    ```

    Or a single value to use for all bands in metals.

    Required for: ACD

### `piezoelectric_constant`

!!! quote ""
    *command-line option:* `--piezoelectric-constant`

    The piezoelectric constants ($`\mathbf{e}`$) in C/m<sup>2</sup> given as either the 
    full 3x3x3 tensor or the 3x6 Voigt form.

    Required for: PIE

### `acceptor_charge`

!!! quote ""
    *Command-line option:* `--acceptor-charge`

    The charge of acceptor defects, in units of electron charge.

    Required for: IMP

    Default: `{{ acceptor_charge }}`

### `donor_charge`

!!! quote ""
    *Command-line option:* `--donor-charge`

    The charge of donor defects, in units of electron charge.

    Required for: IMP

    Default: `{{ donor_charge }}`

### `pop_frequency`

!!! quote ""
    *Command-line option:* `--pop-frequency`

    The polar optical phonon frequency, in THz. Generally, it is ok to take the
    highest optical phonon frequency at the Gamma point.

    Required for: POP
    
### `mean_free_path`

!!! quote ""
    *Command-line option:* `--mean-free-path`

    Basic version of boundary scattering in which the scattering rate is set to
    $`v_g / L`$, where $`v_g`$ is the group velocity and $`L`$ is the mean free
    path in nm.
    
    Required for: MFP
    
### `constant_relaxation_time`

!!! quote ""
    *Command-line option:* `--constant-relaxation-time`

    A constant relaxation time to use as the minimum relaxation time for
    all k-points.
    
    It is not recommended to use this option in conjunction with any other
    scattering rates. Instead, this should be used to compare against
    results calculated in the constant relaxation time approximation.
    
    Required for: CRT


## Performance settings

These settings control the speed and accuracy of calculated properties. In 
general the defaults should give converged values.

### `energy_cutoff`

!!! quote ""
    *Command-line option:* `--energy-cutoff`

    The energy cut-off used to determine which bands to include in the interpolation
    and scattering rate calculation, in eV.

    Default: `{{ energy_cutoff }}`

### `fd_tol`

!!! quote ""
    *Command-line option:* ``--fd-tol``

    The Fermi–Dirac derivative tolerance that controls which k-points to calculate
    the scattering for. Given as a percentage from 0 to 1. Larger values indicate
    that the fewer k-points will be calculated, smaller values indicate a larger
    portion of the Brillouin zone will be calculated.

    Default: `{{ fd_tol }}`

### `dos_estep`

!!! quote ""
    *Command-line option:* `--dos-estep`

    The energy step for the calculated density of states and transport density
    of states, in eV. Controls the accuracy of determining the position of the 
    Fermi level and transport properties. Smaller is better but can quickly
    get more expensive.

    Default: `{{ dos_estep }}`

### `symprec`

!!! quote ""
    *Command-line option:* `--symprec`

    The symmetry finding tolerance, in Å.

    Default: `{{ symprec }}`

### `nworkers`

!!! quote ""
    *Command-line option:* `--nworkers`

    Number of processors to use. `-1` indicates to use all available
    processors.
    
    When using multiprocessing it is recommended to run `export OMP_NUM_THREADS=1` before
    running amset.
    
    Default: `{{ nworkers }}`
    
### `cache_wavefunction`

!!! quote ""
    *Command-line option:* `--cache-wavefunction`

    Cache interpolated wavefunction coefficients. This means that the coefficients
    for each band and k-point on the Fourier interpolated k-point mesh are only 
    calculated once. While this can yield a significant speed-up, it also massively
    increases memory requirements, especially if using a low value of `fd_tol`, or if
    the system contains very flat bands.
    
    If memory issues occur, it is recommended to set `cache_wavefunction` to `False`.
    
    Default: `{{ cache_wavefunction }}`


## Output settings

These settings control the output files and logging.

### `calculate_mobility`

!!! quote ""
    *Command-line option:* ``--calculate-mobility/--no-calculate-mobility``

    Whether to calculate *n*- and *p*-type carrier mobilities. Has no effect
    for metallic systems where mobility is not well defined.

    Default: `{{ calculate_mobility }}`

### `separate_scattering_mobilities`

!!! quote ""
    *Command-line option:* `--separate-mobility/--no-separate-mobility`

    Whether to report the individual scattering rate mobilities. I.e., the mobility
    if only that scattering mechanism were present.

    Default: `{{ separate_mobility }}`

### `file_format`

!!! quote ""
    *Command-line option:* `--file-format`

    The output file format. Options are: `json`, `yaml`, and `txt`.
    
    Note, `write_mesh=True` is not supported using the `txt` format.

    Default: `{{ file_format }}`

### `write_input`

!!! quote ""
    *Command-line option:* `--write-input/--no-write-input`

    Whether to write the input settings to a file called `amset_settings.yaml`.

    Default: `{{ write_input }}`

### `write_mesh`

!!! quote ""
    *Command-line option:* `--write-mesh/--no-write-mesh`

    Whether to write the full k-dependent properties to disk. Properties include
    the band energy, velocity and scattering rate. Only k-points in the 
    irreducible wedge are included.

    **Note:** for large values of [interpolation_factor](#interpolation_factor) 
    his option can use a large amount of disk space.

    Default: `{{ write_mesh }}`

### `print_log`

!!! quote ""
    *Command-line option:* ``--print-log/--no-log``

    Whether to print log messages.

    Default: `{{ print_log }}`

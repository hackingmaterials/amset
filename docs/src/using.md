# Getting started

AMSET can be used from the command-line as a standalone program or from the
Python API. In both cases, the primary input is a `vasprun.xml` file from a
uniform band structure calculation (i.e., on a regular k-point grid and not
along high-symmetry lines).

Temperature and doping ranges, scattering rates, and calculation
parameters are controlled through the settings file. More details on the
available settings are provided in the [settings section](settings.md) of the
documentation. An example settings file is given 
[here](https://github.com/hackingmaterials/amset/blob/master/examples/GaAs/settings.yaml).

## From the command-line

AMSET can be run from the command-line using the `amset run` command. The help
menu listing a summary of the command-line options can be printed using:


```bash
amset run -h
```

By default, AMSET will look for a `vasprun.xml` file and `settings.yaml`
file in the current directory. A different directory can be specified using
the `directory` option, e.g.:

```bash
amset run --directory path/to/files
```

Any settings specified via the command line will override those in the settings
file. For example, the interpolation factor can be easily controlled using:

```bash
amset run --interpolation-factor 20
```


!!! info "Obtaining best performance"
    To obtain the best performance, it is recommended to run `export OMP_NUM_THREADS=1`
    before running AMSET.

## From the Python API

Greater configurability is available when running AMSET from the Python API.
For example, the following snippet will look for a `vasprun.xml` and
`settings.yaml` file in the current directory, then run AMSET.

```python
from amset.core.run import Runner

if __name__ == "__main__":
    runner = Runner.from_directory(directory='.')
    amset_data = runner.run()
```

The API allows for easy convergence of parameters. For example,
the following snippet will run AMSET using multiple interpolation factors.

```python
from amset.core.run import Runner

settings = {'interpolation_factor': 5}

if __name__ == "__main__":
    outputs = []
    for i_factor in range(10, 100, 10):
        settings["interpolation_factor"] = i_factor

        runner = Runner.from_directory(directory='.', settings_override=settings)
        outputs.append(runner.run())
```

When running AMSET from the API, it is not necessary to use a settings file
at all. Instead the settings can be passed as a dictionary. For example:

```python
from amset.core.run import Runner

settings = {
    "interpolation_factor": 150,
    "doping": [1e15, 1e16, 1e17, 1e18],
    "temperatures": [300],

    "deformation_potential": (6.5, 6.5),
    "elastic_constant": 190,
    "static_dielectric": 13.1,
}

if __name__ == "__main__":
    runner = Runner.from_vasprun("vasprun.xml.gz", settings)
    amset_data = runner.run()
```

# Installation

## From source

To install AMSET from source, first clone the repository from GitHub, then
install using pip:

```bash
git clone https://github.com/hackingmaterials/amset.git
cd amset
pip install .
```

If not installing from inside a virtual environment or conda environment, you
may need to specify to install as a *user* via:

```bash
pip install . --user
```

## Installing AMSET on Mac

Due to the way BolzTraP2 must be compiled, the following snippet is required to install AMSET on Mac:

```bash
MACOSX_DEPLOYMENT_TARGET=10.9 pip install amset
```

## Installing AMSET on NERSC

The BolzTraP2 dependency requires some configuration to be installed properly on
CRAY systems. Accordingly, AMSET can be installed using:

```bash
CXX=icpc CC=icc pip install amset
```

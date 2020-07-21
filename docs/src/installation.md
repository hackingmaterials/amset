# Installation

AMSET is a Python 3 package and requires a
[typical scientific Python stack](https://www.scipy.org/about.html).
AMSET can be installed using the Python package manager "Pip",
which will automatically setup other Python packages as required:

```
pip install amset
```
    
If not installing from inside a virtual environment or conda environment, you
may need to specify to install as a *user* via:

```bash
pip install amset --user
```

## Developer Installation

To install an editable version of AMSET, first clone the repository from 
GitHub, then install using pip:

```bash
git clone https://github.com/hackingmaterials/amset.git
cd amset
pip install -e .
```

## Installing AMSET on NERSC

The BolzTraP2 dependency requires some configuration to be installed properly on
CRAY systems. Accordingly, AMSET can be installed using:

```bash
CXX=icpc CC=icc pip install amset
```

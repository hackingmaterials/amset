
# <img alt="amset" src="docs_rst/source/_static/logo-01.png" width="250">

Ab initio Model for Mobility and Seebeck coefficient using Boltzmann Transport equation. AMSET (a.k.a aMoBT) in Python is currently in development and is not functional. If you are intereted in the MATLAB source code of the basic model (aMoBT), please contact alireza@lbl.gov; the basic model is not appropriate for anisotropic or multiple-valley band structures.


Interested in contributing? See our [contribution guidelines](https://github.com/hackingmaterials/amset/blob/master/CONTRIBUTING.md)

### Installing AMSET on NERSC

The BolzTraP2 dependency requires some configuration to be installed properly on CRAY systems. Accordingly, AMSET can be installed using:

```bash
CXX=icc CRAYPE_LINK_TYPE=shared pip install amset
```
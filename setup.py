import os
import platform
import sys
from distutils.sysconfig import get_config_vars
from distutils.version import LooseVersion
from pathlib import Path

from setuptools import find_packages, setup

from amset import __version__

reqs_raw = Path("requirements.txt").read_text()
reqs_list = [r.replace("==", ">=") for r in reqs_raw.split("\n")]

module_dir = os.path.dirname(os.path.abspath(__file__))


# The below snippet was taken from the Pandas setup.py script.
# It should make it easy to install BolzTraP2 without additional
# configuration.

# For mac, ensure extensions are built for macos 10.9 when compiling on a
# 10.9 system or above, overriding distuitls behaviour which is to target
# the version that python was built for. This may be overridden by setting
# MACOSX_DEPLOYMENT_TARGET before calling setup.py
if sys.platform == "darwin":
    if "MACOSX_DEPLOYMENT_TARGET" not in os.environ:
        current_system = platform.mac_ver()[0]
        python_target = get_config_vars().get(
            "MACOSX_DEPLOYMENT_TARGET", current_system
        )
        if LooseVersion(current_system) >= "10.9" > LooseVersion(python_target):
            os.environ["MACOSX_DEPLOYMENT_TARGET"] = "10.9"


with open("README.md", "r") as file:
    long_description = file.read()

if __name__ == "__main__":
    setup(
        name="amset",
        version=__version__,
        description="AMSET is a tool to calculate carrier transport properties "
        "from ab initio calculation data",
        long_description=long_description,
        url="https://github.com/hackingmaterials/amset",
        author="Alex Ganose",
        author_email="aganose@lbl.gov",
        license="modified BSD",
        keywords="conductivity scattering seebeck dft vasp",
        packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
        package_data={
            "amset": [
                "defaults.yaml", "plot/amset_base.mplstyle", "plot/revtex.mplstyle"
            ]
        },
        data_files=["LICENSE", "requirements.txt"],
        zip_safe=False,
        install_requires=reqs_list,
        extras_require={
            'docs': [
                'mkdocs==1.1.2',
                'mkdocs-material==5.4.0',
                'mkdocs-minify-plugin==0.3.0',
                'mkdocs-macros-plugin==0.4.9',
                'markdown-include==0.5.1',
                'markdown-katex==202006.1021',
            ],
            'all-electron': ['pawpyseed==0.6.3'],
            'dev': [
                'coverage==5.1',
                'codacy-coverage==1.3.11',
                'pycodestyle==2.6.0',
                'mypy==0.782',
                'pydocstyle==5.0.2',
                'flake8==3.8.3',
                'pylint==2.5.3',
            ]
        },
        classifiers=[
            "Programming Language :: Python :: 3.6",
            "Development Status :: 4 - Beta",
            "Intended Audience :: Science/Research",
            "Intended Audience :: System Administrators",
            "Intended Audience :: Information Technology",
            "Operating System :: OS Independent",
            "Topic :: Other/Nonlisted Topic",
            "Topic :: Scientific/Engineering",
        ],
        test_suite="nose.collector",
        tests_require=["nose"],
        entry_points={"console_scripts": ["amset = amset.tools.cli:cli"]},
    )

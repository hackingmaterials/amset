import platform
import sys
import os

from setuptools import setup, find_packages

from distutils.sysconfig import get_config_vars
from distutils.version import LooseVersion


module_dir = os.path.dirname(os.path.abspath(__file__))


# The below snippet was taken from the Pandas setup.py script.
# It should make it easy to install BolzTraP2 without additional
# configuration.

# For mac, ensure extensions are built for macos 10.9 when compiling on a
# 10.9 system or above, overriding distuitls behaviour which is to target
# the version that python was built for. This may be overridden by setting
# MACOSX_DEPLOYMENT_TARGET before calling setup.py
if sys.platform == 'darwin':
    if 'MACOSX_DEPLOYMENT_TARGET' not in os.environ:
        current_system = platform.mac_ver()[0]
        python_target = get_config_vars().get('MACOSX_DEPLOYMENT_TARGET',
                                              current_system)
        if (LooseVersion(python_target) < '10.9' and
                LooseVersion(current_system) >= '10.9'):
            os.environ['MACOSX_DEPLOYMENT_TARGET'] = '10.9'


with open('README.md', 'r') as file:
    long_description = file.read()

if __name__ == "__main__":
    setup(
        name='amset',
        version='0.1.0',
        description="AMSET is a tool to calculate carrier transport properties "
                    "from ab initio calculation data",
        long_description=long_description,
        url='https://github.com/hackingmaterials/amset',
        author="Alex Ganose",
        author_email='aganose@lbl.gov',
        license='modified BSD',
        keywords='conductivity scattering seebeck dft vasp',
        packages=find_packages(),
        package_data={},
        data_files=['LICENSE', 'requirements-optional.txt'],
        zip_safe=False,
        install_requires=['numpy', 'pymatgen>=2019.5.8', 'scipy', 'monty',
                          'matplotlib', 'BoltzTraP2',
                          'spglib>=1.12.2', "scikit-learn", "tqdm",
                          "memory_profiler", "numexpr"],
        extras_require={'docs': ['sphinx']},
        classifiers=['Programming Language :: Python :: 3.6',
                     'Development Status :: 4 - Beta',
                     'Intended Audience :: Science/Research',
                     'Intended Audience :: System Administrators',
                     'Intended Audience :: Information Technology',
                     'Operating System :: OS Independent',
                     'Topic :: Other/Nonlisted Topic',
                     'Topic :: Scientific/Engineering'],
        test_suite='nose.collector',
        tests_require=['nose'],
        entry_points={'console_scripts': ['amset = amset.cli:main']}
    )

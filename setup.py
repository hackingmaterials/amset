from setuptools import setup, find_packages

import os

module_dir = os.path.dirname(os.path.abspath(__file__))

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
        install_requires=['numpy', 'pymatgen', 'scipy', 'monty',
                          'matplotlib', 'BoltzTraP2',
                          'spglib>=1.12.2', "scikit-learn", "tqdm",
                          "memory_profiler"],
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

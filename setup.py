from setuptools import find_packages, setup

from amset import __version__

with open("README.md", "r") as file:
    long_description = file.read()

if __name__ == "__main__":
    setup(
        name="amset",
        version=__version__,
        description="AMSET is a tool to calculate carrier transport properties "
        "from ab initio calculation data",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/hackingmaterials/amset",
        author="Alex Ganose",
        author_email="a.ganose@imperial.ac.uk",
        license="modified BSD",
        keywords="mobility conductivity seebeck scattering lifetime rates dft vasp",
        packages=find_packages(),
        package_data={
            "amset": [
                "defaults.yaml",
                "interpolation/quad.json",
                "plot/amset_base.mplstyle",
                "plot/revtex.mplstyle",
            ]
        },
        data_files=["LICENSE"],
        zip_safe=False,
        install_requires=[
            "pymatgen>=2022.0.16",
            "scipy",
            "monty",
            "matplotlib",
            "BoltzTraP2",
            "tqdm",
            "tabulate",
            "memory_profiler",
            "spglib",
            "click",
            "sumo",
            "h5py",
            "pyFFTW",
            "interpolation",
            "numba",
        ],
        setup_requires=[
            "numpy",
        ],
        extras_require={
            "docs": [
                "mkdocs==1.4.2",
                "mkdocs-material==9.1.2",
                "mkdocs-minify-plugin==0.6.2",
                "mkdocs-macros-plugin==0.7.0",
                "markdown-include==0.8.1",
                "markdown-katex==202112.1034",
            ],
            "tests": ["pytest==7.2.2", "pytest-cov==4.0.0"],
            "all-electron": ["pawpyseed==0.7.1"],
            "dev": [
                "coverage==7.2.1",
                "codacy-coverage==1.3.11",
                "pycodestyle==2.10.0",
                "mypy==0.991",
                "pydocstyle==6.3.0",
                "flake8==5.0.4",
                "pylint==2.15.10",
                "black==22.10.0",
                "pre-commit==3.1.1",
            ],
        },
        python_requires=">=3.8",
        classifiers=[
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Development Status :: 4 - Beta",
            "Intended Audience :: Science/Research",
            "Intended Audience :: System Administrators",
            "Intended Audience :: Information Technology",
            "Operating System :: OS Independent",
            "Topic :: Other/Nonlisted Topic",
            "Topic :: Scientific/Engineering",
        ],
        tests_require=["pytest"],
        entry_points={
            "console_scripts": [
                "amset = amset.tools.cli:cli",
            ]
        },
    )

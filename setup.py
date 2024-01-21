import os
import pathlib
import shutil
import subprocess

import numpy
import setuptools
from setuptools import find_packages, setup

from amset import __version__

with open("README.md", "r") as file:
    long_description = file.read()

if "AMSET_USE_CMAKE" in os.environ and os.environ["AMSET_USE_CMAKE"].lower() == "false":
    use_cmake = False
else:
    use_cmake = True


def _run_cmake(build_dir):
    build_dir.mkdir()
    args = [
        "cmake",
        "-S",
        ".",
        "-B",
        "_build",
        "-DCMAKE_INSTALL_PREFIX=.",
    ]
    cmake_output = subprocess.check_output(args)
    print(cmake_output.decode("utf-8"))
    subprocess.check_call(["cmake", "--build", "_build", "-v"])
    return cmake_output


def _clean_cmake(build_dir):
    if build_dir.exists():
        shutil.rmtree(build_dir)


def _get_params_from_site_cfg():
    """Read extra_compile_args and extra_link_args.

    Examples
    --------
    # For macOS
    extra_compile_args = -fopenmp=libomp
    extra_link_args = -lomp -lopenblas

    # For linux
    extra_compile_args = -fopenmp
    extra_link_args = -lgomp  -lopenblas -lpthread

    """
    params = {
        "define_macros": [],
        "extra_link_args": [],
        "extra_compile_args": [],
        "extra_objects": [],
        "include_dirs": [],
    }
    use_mkl_lapacke = False
    use_threaded_blas = False

    site_cfg_file = pathlib.Path.cwd() / "site.cfg"
    if not site_cfg_file.exists():
        return params

    with open(site_cfg_file) as f:
        lines = [line.strip().split("=", maxsplit=1) for line in f]

        for line in lines:
            if len(line) < 2:
                continue
            key = line[0].strip()
            val = line[1].strip()
            if key not in params:
                continue
            if key == "define_macros":
                pair = val.split(maxsplit=1)
                if pair[1].lower() == "none":
                    pair[1] = None
                params[key].append(tuple(pair))
            else:
                if "mkl" in val:
                    use_mkl_lapacke = True
                if "openblas" in val:
                    use_threaded_blas = True
                params[key] += val.split()

    if use_mkl_lapacke:
        params["define_macros"].append(("MKL_LAPACKE", None))
    if use_threaded_blas:
        params["define_macros"].append(("MULTITHREADED_BLAS", None))

    print("=============================================")
    print("Parameters found in site.cfg")
    for key, val in params.items():
        print(f"{key}: {val}")
    print("=============================================")
    return params


def _get_extensions(build_dir):
    """Return python extension setting.

    User customization by site.cfg file
    -----------------------------------
    See _get_params_from_site_cfg().

    Automatic search using cmake
    ----------------------------
    Invoked by environment variable unless PHONO3PY_USE_CMAKE=false.

    """
    params = _get_params_from_site_cfg()
    extra_objects_amst = []

    if not use_cmake or not shutil.which("cmake"):
        print("** Setup without using cmake **")
        sources_amset = [
            "c/_amset.c",
        ]
    else:
        print("** Setup using cmake **")
        use_mkl_lapacke = False
        found_extra_link_args = []
        found_extra_compile_args = []
        sources_amset = ["c/_amset.c"]
        cmake_output = _run_cmake(build_dir)
        found_flags = {}
        found_libs = {}
        for line in cmake_output.decode("utf-8").split("\n"):
            for key in ["BLAS", "LAPACK", "OpenMP"]:
                if f"{key} libs" in line and len(line.split()) > 3:
                    found_libs[key] = line.split()[3].split(";")
                if f"{key} flags" in line and len(line.split()) > 3:
                    found_flags[key] = line.split()[3].split(";")
        for key, value in found_libs.items():
            found_extra_link_args += value
            for element in value:
                if "libmkl" in element:
                    use_mkl_lapacke = True
        for key, value in found_flags.items():
            found_extra_compile_args += value
        if use_mkl_lapacke:
            params["define_macros"].append(("MKL_LAPACKE", None))

        libamst = list((pathlib.Path.cwd() / "_build").glob("*amst.*"))
        if libamst:
            print("=============================================")
            print(f"AMSET library: {libamst[0]}")
            print("=============================================")
            extra_objects_amst += [str(libamst[0])]

        params["extra_link_args"] += found_extra_link_args
        params["extra_compile_args"] += found_extra_compile_args

        print("=============================================")
        print("Parameters found by cmake")
        print("extra_compile_args: ", found_extra_compile_args)
        print("extra_link_args: ", found_extra_link_args)
        print("define_macros: ", params["define_macros"])
        print("=============================================")
        print()

    extensions = []
    params["define_macros"].append(("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"))
    params["include_dirs"] += ["c", numpy.get_include()]

    extensions.append(
        setuptools.Extension(
            "amset._amset",
            sources=sources_amset,
            extra_link_args=params["extra_link_args"],
            include_dirs=params["include_dirs"],
            extra_compile_args=params["extra_compile_args"],
            extra_objects=params["extra_objects"] + extra_objects_amst,
            define_macros=params["define_macros"],
        )
    )

    return extensions


if __name__ == "__main__":
    build_dir = pathlib.Path.cwd() / "_build"
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
                "mkdocs==1.5.2",
                "mkdocs-material==9.1.21",
                "mkdocs-minify-plugin==0.7.1",
                "mkdocs-macros-plugin==1.0.4",
                "markdown-include==0.8.1",
                "markdown-katex==202112.1034",
            ],
            "tests": ["pytest==7.4.0", "pytest-cov==4.1.0"],
            "all-electron": ["pawpyseed==0.7.1"],
            "dev": ["pre-commit==3.3.3"],
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
        ext_modules=_get_extensions(build_dir),
    )
    _clean_cmake(build_dir)

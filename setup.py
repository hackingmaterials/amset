from pathlib import Path

from setuptools import find_packages, setup

from amset import __version__

reqs_raw = Path("requirements.txt").read_text()
reqs_list = [r.replace("==", ">=") for r in reqs_raw.split("\n")]

with open("README.md", "r") as file:
    long_description = file.read()

if __name__ == "__main__":
    setup(
        name="amset",
        version=__version__,
        description="AMSET is a tool to calculate carrier transport properties "
        "from ab initio calculation data",
        long_description=long_description,
        long_description_content_type='text/markdown',
        url="https://github.com/hackingmaterials/amset",
        author="Alex Ganose",
        author_email="aganose@lbl.gov",
        license="modified BSD",
        keywords="mobility conductivity seebeck scattering lifetime rates dft vasp",
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
                'coverage==5.2',
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

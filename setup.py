#!/usr/bin/env python

from setuptools import setup, find_packages
import os
import multiprocessing, logging  # AJ: for some reason this is needed to not have "python setup.py test" freak out

module_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    setup(
        name='Amset',
        version='0.1.0',
        description='Amset is Ab initio Mobility, Seebeck, and Transport',
        long_description=open(os.path.join(module_dir, 'README.rst')).read(),
        url='https://github.com/hackingmaterials/Amset',
        author='Alireza Faghaninia',
        author_email='alireza@lbl.gov',
        license='modified BSD',
        packages=find_packages(),
        #package_data={'amset.example_module': ['*.txt'], 'amset.flask_site': ['static/images/*', 'static/css/*', 'static/js/*', 'templates/*']},
        package_data={},
        zip_safe=False,
        #install_requires=['six>=1.5.2'],
        install_requires=[],
        #extras_require={'plotting':['matplotlib>=1.1.1'},
        extras_require={},
        classifiers=['Programming Language :: Python :: 2.7',
                     'Development Status :: 4 - Beta',
                     'Intended Audience :: Science/Research',
                     'Intended Audience :: System Administrators',
                     'Intended Audience :: Information Technology',
                     'Operating System :: OS Independent',
                     'Topic :: Other/Nonlisted Topic',
                     'Topic :: Scientific/Engineering'],
        test_suite='nose.collector',
        tests_require=['nose'],
        scripts=[]
        #scripts=[os.path.join('scripts', f) for f in os.listdir(os.path.join(module_dir, 'scripts'))]
    )

Installation
============

From Pypi
---------

From source
-----------

Installing AMSET on NERSC
~~~~~~~~~~~~~~~~~~~~~~~~~

The BolzTraP2 dependency requires some configuration to be installed properly on
 CRAY systems. Accordingly, AMSET can be installed using:

.. code-block:: bash

    CXX=icc CRAYPE_LINK_TYPE=shared pip install amset

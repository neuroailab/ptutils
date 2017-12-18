Installation
============

Let's get started using PTUtils to train deep learning models!

Requirements
~~~~~~~~~~~~

.. csv-table::
   :header: "Ubuntu", "OSX", "Description"
   :widths: 20, 20, 40
   :escape: ~

   python-pip, pip, Tool to install python dependencies
   python-virtualenv (*), virtualenv (*), Allows creation of isolated environments ((*): This is required only for Python 2.7 installs. With Python3: test for presence of ``venv`` with ``python3 -m venv -h``)
   libhdf5-dev, h5py, Enables loading of hdf5 formats
   pymongo, pymongo, Python interface to MongoDB.

Installation
~~~~~~~~~~~~

We recommend installing PTUtils within a `virtual
environment <http://docs.python-guide.org/en/latest/dev/virtualenvs/>`__
to ensure a self-contained environment. To install PTUtils within an
already existing virtual environment, see the System-wide Install section.

.. code-block:: bash

    git clone https://github.com/neuroailab/ptutils.git
    cd ptutils;
    python setup.py install

Or, if you would prefer to install using pip:

.. code-block:: bash

    pip install git+https://github.com/neuroailab/ptutils.git

To activate the virtual environment, type

.. code-block:: bash

    . .venv/bin/activate

You will see the prompt change to reflect the activated environment. When you are finished, remember to deactivate the environment

.. code-block:: bash

    deactivate

Congratulations, you have installed PTUtils! Next, we recommend you learn
how to run models using PTUtils and walk through the MNIST multilayer
perceptron tutorial.
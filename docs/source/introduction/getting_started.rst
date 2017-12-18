Getting Started
===============

We will take our first steps with PTUtils by training a multilayer perceptron (MLP)
on the `MNIST <http://yann.lecun.com/exdb/mnist/>`_  dataset. We demonstrate the basic
functionality of PTUtils by saving and loading experiments.

Configuring MongoDB
~~~~~~~~~~~~~~~~~~~

Before we can begin training MNIST, we must locate a running instance of
MongoDB and determine the port it is listening to (by default, MongoDB
is set to listen on port ``27017``).

To verify that an instance of MongoDB is running on your machine, type

.. code-block:: bash

    $ps aux | grep mongod

which should produce an output resembling

.. code-block:: bash

    root      8030  0.3  0.8 13326208376 1165680 ? Sl   Aug07 170:37 /usr/bin/mongod -f /etc/mongod.conf

Look for the lines:

::

    ...
    # network interfaces
    net:
      port: 27017
    ...

in the mongod configuration file (``/etc/mongod.conf`` above) to determine the port,

These tests show basic procedures for training, validating, and extracting features from
models.

Note about MongoDB:

The tests require a MongoDB instance to be available on the port defined by "testport" in
the code below. This db can either be local to where you run these tests (and therefore
on 'localhost' by default) or it can be running somewhere else and then by ssh-tunneled on
the relevant port to the host where you run these tests. [That is, before testing, you'd run

.. code-block:: bash

         ssh -f -N -L  [testport]:localhost:[testport] [username]@mongohost.xx.xx

on the machine where you're running these tests. ``[mongohost]`` is the where the mongodb
instance is running.

Saving an Experiment to MongoDB
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now that we have a database ready to store our training results, let's specify
how they will be saved. Critically, any experiment executed by
PTUtils can be uniquely identified by the following 5-tuple:

+--------------+------------------------------------------+
| ``host``     | Hostname where database connection lives |
+--------------+------------------------------------------+
| ``port``     | Port where database connection lives     |
+--------------+------------------------------------------+
| ``dbname``   | Name of database for storage             |
+--------------+------------+-----------------------------+
| ``collname`` | Name of the database collection          |
+--------------+------------+-----------------------------+
| ``exp_id``   | Experiment id descriptor                 |
+--------------+------------------------------------------+

The variables host/port/dbname/coll/exp_id control the location of the saved
data for the run, in increasing order of specificity.

.. note::
  When choosing these, consider that:

        1.  If a given host/port/dbname/coll/exp_id already has saved checkpoints,
            then any new call to start training with these same location variables
            will start to train from the most recent saved checkpoint.  If you mistakenly
            try to start training a new model with different variable names or structure
            from that existing checkpoint, an error will be raised as the model will be
            incompatiable with the saved variables (unless a remapping between the old 
            model and new model is specified), 

        2.  When choosing what dbname, coll, and exp_id, to use, keep in mind that mongodb
            queries only operate over a single collection.  So if you want to analyze
            results from a bunch of experiments together using mongod queries, you should
            put them all in the same collection, but with different exp_ids.  If, on the
            other hand, you never expect to analyze data from two experiments together,
            you can put them in different collections or different databases.  Choosing
            between putting two experiments in two collections in the same database
            or in two totally different databases will depend on how you want to organize
            your results and is really a matter of preference.

        3.  The sum of the lengths of the dbname and collname must not exceed 64 characters. This
            is a constraint from MongoDB, not PTUtils.

Specifying an Experiment
~~~~~~~~~~~~~~~~~~~~~~~~

We specify the configuration for our experiment in a ::class::dict called ``params``. The contents
of this dictionary are passed to the ``init`` method of the ``::class::Runner`` class, which is 
PTUtils primary class for running experiments. Once the ``Runner`` object has been initialized with
the ``params`` dictionary, it can be used to train and test experiments.

.. code-block:: python

    import sys
    import time
    import pymongo as pm
    import re

    import torch
    import torch.nn as nn

    sys.path.insert(0, '../')
    import ptutils

    LOG_LEVEL = 'WARNING'
    MONGO_PORT = 27017
    CUDA = 0

    class MNIST(torch.nn.Module, ptutils.base.Base):
        def __init__(self, **kwargs):
            super(MNIST, self).__init__()
            ptutils.base.Base.__init__(self, **kwargs)

            self.layer1 = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=5, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2))
            self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=5, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2))
            self.fc = nn.Linear(7 * 7 * 32, 10)

        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            return out


    class Criterion(nn.CrossEntropyLoss, ptutils.base.Base):
        def __init__(self, **kwargs):
            super(Criterion, self).__init__()
            ptutils.base.Base.__init__(self, **kwargs)


    params = {
        'func': ptutils.runner.Runner,
        'name': 'MNISTRunner',
        'exp_id': 'mnist_training',
        'description': 'The \'Hello, World!\' of deep learning',

        # Define Model Params
        'model': {
            'func': ptutils.model.Model,
            'name': 'MNIST',
            'use_cuda': True,
            'devices': CUDA,

            'net': {
                'func': MNIST,
                'name': 'mnist'},
            'criterion': {
                'func': Criterion,
                'name': 'crossentropy'},
            'optimizer': {
                'func': ptutils.optimizer.Optimizer,
                'name': 'sgd_optimizer',
                'algorithm': 'SGD',
                'params': None,
                'defaults': {
                    'momentum': 0.9,
                    'lr': 0.05}}},

        # Define DataProvider Params
        'dataprovider': {
            'func': ptutils.data.MNISTProvider,
            'name': 'MNISTProvider',
            'n_threads': 4,
            'batch_size': 64,
            'modes': ('train', 'test')},

        # Define DBInterface Params
        'dbinterface': {
            'func': ptutils.database.MongoInterface,
            'name': 'mongo',
            'port': MONGO_PORT,
            'host': 'localhost',
            'database_name': 'ptutils_test',
            'collection_name': 'ptutils_test'},

        'train_params': {
            'num_steps': 50
        },

        'validation_params': {
        },

        'save_params': {
            'metric_freq': 25,
            'val_freq': 10},

        'load_params': {
            'restore': False,
            'dbinterface': {
                'func': ptutils.database.MongoInterface,
                'name': 'mongo',
                'port': MONGO_PORT,
                'host': 'localhost',
                'database_name': 'ptutils_test',
                'collection_name': 'ptutils_test'},
            'query': {'exp_id': exp_id},
            'restore_params': None,
            'restore_mapping': None
            }
        }

    # Initialize the runner
    runner = ptutils.runner.Runner.init(**params)
    # Train the model
    runner.train()
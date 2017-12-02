Getting Started
===============

**"Hello World" - TFUtils style**

We will take our first steps with TFUtils by training the cononical *MNIST*
network and saving the results to a MongoDB database (The following assumes
you have successfully completed the installation process).

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
      port: 29101
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

Specifying an Experiment
~~~~~~~~~~~~~~~~~~~~~~~~

Now that we have a database ready to store our training results, let's specify
how they will be saved. Critically, any experiment executed by
TFUtils can be uniquely identified by the following 5-tuple:

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
data for the run, in order of increasing specificity.

.. note::
  When choosing these, consider that:

        1.  If a given host/port/dbname/coll/exp_id already has saved checkpoints,
            then any new call to start training with these same location variables
            will start to train from the most recent saved checkpoint.  If you mistakenly
            try to start training a new model with different variable names, or structure,
            from that existing checkpoint, an error will be raised, as the model will be
            incompatiable with the saved variables.

        2.  When choosing what dbname, coll, and exp_id, to use, keep in mind that mongodb
            queries only operate over a single collection.  So if you want to analyze
            results from a bunch of experiments together using mongod queries, you should
            put them all in the same collection, but with different exp_ids.  If, on the
            other hand, you never expect to analyze data from two experiments together,
            you can put them in different collections or different databases.  Choosing
            between putting two experiments in two collections in the same database
            or in two totally different databases will depend on how you want to organize
            your results and is really a matter of preference.

With this in mind, let us specify our ``save_params``:

.. code-block:: python

   """Training MNIST with ptutils

   The 'Hello, World!' of deep learning.

   """
   import sys

   import torch
   import torch.nn as nn

   sys.path.insert(0, '../')
   import ptutils


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


   class Sequential(nn.Sequential, ptutils.base.Base):
       pass


   # Experiment Params
   params = {
       'func': ptutils.runner.Runner,
       'name': 'MNISTRunner',
       'exp_id': 'mnist_example',
       'description': 'The \'Hello, Wordl!\' of deep learning',
       'Notes':
           """
           This is a simple experiment to demonstrate the most common
           way in which a user will interact with ptutils. Typically,
           a user will specify a single model, dbinterface and
           dataprovider class to be run by the default runner class.

           You can add arbitrary attributes to all instances of the
           ptutils.base.Base class, such as these notes. Here would
           be a good place to capture any thoughts or ideas about
           the experiment that will be run.
           """,

       # Define Model Params
       'model': {
           'func': ptutils.model.Model,
           'name': 'MNIST',
           'use_cuda': True,
           'devices': 0,

           'net': {
               'func': MNIST,
               'name': 'mnist',
               'layer3': Sequential(
                   nn.Conv2d(1, 16, (5, 5)))},
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
           'batch_size': 128,
           'modes': ('train', 'test')},

       # Define DBInterface Params
       'dbinterface': {
           'func': ptutils.database.MongoInterface,
           'name': 'mongo',
           'port': 27017,
           'host': 'localhost',
           'database_name': 'ptutils',
           'collection_name': 'ptutils'},

       'train_params': {
           'num_steps': 5},

       'validation_params': {},

       'save_params': {
           'metric_freq': 1},

       'load_params': {
           'restore': True,
           'restore_params': None,
           'restore_mapping': None}}


   runner = ptutils.runner.Runner.from_params(**params)
   runner.train_from_params()


   print "done"
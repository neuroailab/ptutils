"""ptutils tests.

These tests are meant to reflect the tests in the tfutils repository, e.g.
test_training(), test_training_save(), and test_validation().

These are higher level test than the test in test.py, but they will be integrated
into the same testing suite once they are finish.

Note about MongoDB:
The tests require a MongoDB instance to be available on the port defined by "testport" in
the code below.   This db can either be local to where you run these tests (and therefore
on 'localhost' by default) or it can be running somewhere else and then by ssh-tunneled on
the relevant port to the host where you run these tests.  [That is, before testing, you'd run
         ssh -f -N -L  [testport]:localhost:[testport] [username]@mongohost.xx.xx
on the machine where you're running these tests.   [mongohost] is the where the mongodb
instance is running.


"""

from __future__ import division, print_function, absolute_import

import os
import re
import sys
import time
import errno
import shutil
import logging
import pymongo as pm
import unittest
import numpy as np
from bson.objectid import ObjectId

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, '../')
# from ptutils import base, data, error, model, runner, database
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


class Sequential(nn.Sequential, ptutils.base.Base):
    pass


def test_training():
    """Illustrate training.

    This test illustrates how basic training is performed using the
    ptutils.runner.train_from_params function.  This is the first in a sequence of
    interconnected tests. It creates a pretrained model that is used by
    the next few tests (test_validation and test_feature_extraction).
    As can be seen by looking at how the test checks for correctness, after the
    training is run, results of training, including (intermittently) the full
    variables needed to re-initialize the tensorflow model, are stored in a
    MongoDB.

    """
    # set up the parameters
    params = {
        'func': ptutils.runner.Runner,
        'name': 'MNISTRunner',
        'exp_id': "mnist_example_test",
        'description': 'The \'Hello, World!\' of deep learning',
        # 'use_cuda': True,
        # 'devices': CUDA,
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
            'batch_size': 128,
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
            'num_steps': 50,
            'train': True},

        'validation_params': {},

        'save_params': {
            'metric_freq': 25},

        'load_params': {
            'restore': False,
            'dbinterface': {
                'func': ptutils.database.MongoInterface,
                'name': 'mongo',
                'port': MONGO_PORT,
                'host': 'localhost',
                'database_name': 'ptutils_test',
                'collection_name': 'ptutils_test'},
            'load_query': {'exp_id': "mnist_example_test"},
            'restore_params': None,
            'restore_mapping': None}}

    # clear database
    conn = pm.MongoClient(host=params['dbinterface']['host'],
                          port=params['dbinterface']['port'])
    conn[params['dbinterface']['database_name']][params['dbinterface']['collection_name']].delete_many({'exp_id': params['exp_id']})
    assert conn[params['dbinterface']['database_name']][params['dbinterface']['collection_name']].find({'exp_id': params['exp_id']}).count() == 0
    # actually run the training
    runner = ptutils.runner.Runner.from_params(**params)
    runner.train_from_params()

    # test if the number of saved documents is correct: (num_steps / metric_freq) + 1 for initial save
    assert runner.dbinterface.collection.find({'exp_id': params['exp_id']}).count() == (params['train_params']['num_steps'] / params['save_params']['metric_freq']) + 1

    # run another 50 steps of training on the same experiment id.
    params['train_params']['num_steps'] = 100
    params['load_params']['restore'] = True
    runner = ptutils.runner.Runner.from_params(**params)
    runner.train_from_params()

    # test if results are as expected -- should this be plus 1?
    print("params['train_params']['num_steps']/params['save_params']['metric_freq']", runner.train_params['num_steps'] // params['save_params']['metric_freq'])
    print("runner.dbinterface.collection.find({'exp_id': params['exp_id']}).count()", runner.dbinterface.collection.find({'exp_id': params['exp_id']}).count())
    assert runner.dbinterface.collection.find({'exp_id': params['exp_id']}).count() == (runner.train_params['num_steps'] // params['save_params']['metric_freq']) + 2  # there have been two initial saves now.
    assert runner.dbinterface.collection.distinct('exp_id')[0] == params['exp_id']


    # TODO: this won't work in our current setup. we need to figure out whether we want to
    # replicate the tfutils loading structure, i.e. whether there should be a separate load exp_id
    # run 500 more steps but save to a new experiment id.
    params['train_params']['num_steps'] = 1500
    params['load_params'] = {'exp_id': 'training0'}
    params['save_params']['exp_id'] = 'training1'

    # base.train_from_params(**params)
    # assert conn[testdbname][testcol + '.files'].find({'exp_id': 'training1',
    #                                                   'saved_filters': True}).distinct('step') == [1200, 1400]

def test_validation():
    """Illustrate validation.
    This is a test illustrating how to compute performance on a trained model on a new dataset,
    using the test_from_params function. This test assumes that test_training function
    has run first (to provide a pre-trained model to validate).
    After the test is run, results from the validation are stored in the MongoDB.
    (The test shows how the record can be loaded for inspection.)
    """
    # specify the parameters for the validation
    params = {}

    params['model_params'] = {'func': model.mnist_tfutils}

    params['load_params'] = {'host': testhost,
                             'port': testport,
                             'dbname': testdbname,
                             'collname': testcol,
                             'exp_id': 'training0'}

    params['save_params'] = {'exp_id': 'validation0'}

    params['validation_params'] = {'valid0': {'data_params': {'func': data.MNIST,
                                                              'batch_size': 100,
                                                              'group': 'test',
                                                              'n_threads': 4},
                                              'queue_params': {'queue_type': 'fifo',
                                                               'batch_size': 100},
                                              'num_steps': 10,
                                              'agg_func': utils.mean_dict}}

    # check that the results are correct
    conn = pm.MongoClient(host=testhost,
                          port=testport)

    conn[testdbname][testcol + '.files'].delete_many({'exp_id': 'validation0'})

    # actually run the model
    base.test_from_params(**params)

    # ... specifically, there is now a record containing the validation0 performance results
    assert conn[testdbname][testcol + '.files'].find({'exp_id': 'validation0'}).count() == 1
    # ... here's how to load the record:
    r = conn[testdbname][testcol + '.files'].find({'exp_id': 'validation0'})[0]
    asserts_for_record(r, params, train=False)

    # ... check that the recorrectly ties to the id information for the
    # pre-trained model it was supposed to validate
    assert r['validates']
    idval = conn[testdbname][testcol + '.files'].find({'exp_id': 'training0'})[50]['_id']
    v = conn[testdbname][testcol + '.files'].find({'exp_id': 'validation0'})[0]['validates']
    assert idval == v

if __name__ == '__main__':
    test_training()

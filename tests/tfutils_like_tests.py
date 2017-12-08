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


def setup_params(exp_id=None):
    params = {
        'func': ptutils.runner.Runner,
        'name': 'MNISTRunner',
        'exp_id': exp_id,
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
            'exp_id': exp_id,
            'restore_params': None,
            'restore_mapping': None}}
    return params


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
    # Set up the parameters.
    exp_id = 'mnist_training'
    new_exp_id = 'mnist_training1'
    params = setup_params(exp_id)

    # Clear database.
    conn = pm.MongoClient(host=params['dbinterface']['host'], port=params['dbinterface']['port'])
    conn[params['dbinterface']['database_name']][params['dbinterface']['collection_name']].delete_many({'exp_id': params['exp_id']})
    conn[params['dbinterface']['database_name']][params['dbinterface']['collection_name']].delete_many({'exp_id': new_exp_id})
    conn[params['dbinterface']['database_name']][params['dbinterface']['collection_name']].drop()
    assert conn[params['dbinterface']['database_name']][params['dbinterface']['collection_name']].find({'exp_id': params['exp_id']}).count() == 0

    # Actually run the training.
    runner = ptutils.runner.Runner.init(**params)
    runner.train_from_params()

    # Test if the number of saved documents is correct: (num_steps / metric_freq) + 1 for initial save.
    assert runner.dbinterface.collection.find({'exp_id': params['exp_id']}).count() == (params['train_params']['num_steps'] / params['save_params']['metric_freq']) + 1

    # Run another 50 steps of training on the same experiment id.
    params['train_params']['num_steps'] = 100
    params['load_params']['restore'] = True

    runner = ptutils.runner.Runner.init(**params)
    runner.train_from_params()

    # Test if results are as expected -- should this be plus 2?
    print("params['train_params']['num_steps']/params['save_params']['metric_freq']", runner.train_params['num_steps'] // params['save_params']['metric_freq'])
    print("runner.dbinterface.collection.find({'exp_id': params['exp_id']}).count()", runner.dbinterface.collection.find({'exp_id': params['exp_id']}).count())
    assert runner.dbinterface.collection.find({'exp_id': params['exp_id']}).count() == (runner.train_params['num_steps'] // params['save_params']['metric_freq']) + 2  # there have been two initial saves now.
    assert runner.dbinterface.collection.distinct('exp_id')[0] == params['exp_id']

    # Run 100 more steps but save to a new experiment id.
    params['exp_id'] = new_exp_id
    params['train_params']['num_steps'] = 200

    runner = ptutils.runner.Runner.init(**params)
    runner.train_from_params()

    assert runner.dbinterface.collection.find({'exp_id': params['exp_id']}).count() == 5


if __name__ == '__main__':
    runner = test_training()

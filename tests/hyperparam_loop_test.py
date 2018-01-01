from __future__ import division, print_function, absolute_import

import sys
import time
import pymongo as pm
import re

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, '../')
import ptutils

LOG_LEVEL = 'WARNING'
MONGO_PORT = 27017
CUDA = 2


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
                    'lr': np.random.uniform()}}},

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
    return params

def setup_params_hp(exp_id, param_dict_list):

    params = {
        'func': ptutils.runner.HyperParameterStatisticRunner,
        'name': 'HPRUN',
        'param_dict_list': param_dict_list,
        'num_iterations_per_param': 2,
        'exp_id': exp_id,

        # Define DBInterface Params
        'dbinterface': {
            'func': ptutils.database.MongoInterface,
            'name': 'mongo',
            'port': MONGO_PORT,
            'host': 'localhost',
            'database_name': 'ptutils_test',
            'collection_name': 'ptutils_test'},

        'load_params': {
            'restore': True,
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
    exp_id = 'mnist_training_hyper_param'
    new_exp_id = 'new_mnist_training'
    
    params_list = [setup_params('hp_'+str(i)) for i in range(5)]
    params = setup_params_hp(exp_id, params_list)

    # # Clear database.
    # conn = pm.MongoClient(host=params['dbinterface']['host'], port=params['dbinterface']['port'])
    # conn[params['dbinterface']['database_name']][params['dbinterface']['collection_name']].delete_many({'exp_id': params['exp_id']})
    # conn[params['dbinterface']['database_name']][params['dbinterface']['collection_name']].delete_many({'exp_id': new_exp_id})
    # conn[params['dbinterface']['database_name']][params['dbinterface']['collection_name']].drop()
    # assert conn[params['dbinterface']['database_name']][params['dbinterface']['collection_name']].find({'exp_id': params['exp_id']}).count() == 0
    # assert conn[params['dbinterface']['database_name']][params['dbinterface']['collection_name']].find({'exp_id': new_exp_id}).count() == 0

    # actually run the training.
    runner = ptutils.runner.HyperParameterStatisticRunner.init(**params)
    runner.train()
    

    # time.sleep(10)  # Wait for the morst recent record to finsh being saved to db.

    # # Test if the number of saved documents is correct: (num_steps / metric_freq).
    # assert runner.dbinterface.collection.find({'exp_id': params['exp_id']}).count() == (params['train_params']['num_steps'] // params['save_params']['metric_freq'])

    # # Run another 50 steps of training on the same experiment id.
    # params['train_params']['num_steps'] = 100
    # params['load_params']['restore'] = True

    # # Revive the experiment using little more than load_params.
    # revive_params = {k: v for k, v in params.items() if k in ['func', 'exp_id', 'load_params', 'train_params']}

    # runner = ptutils.runner.Runner.init(**revive_params)
    # runner.train()

    # time.sleep(10)  # Wait for the most recent record to finsh being saved to db.

    # # Test if the number of saved documents is correct: (num_steps / metric_freq).
    # assert runner.dbinterface.collection.find({'exp_id': params['exp_id']}).count() == (
    #     runner.train_params['num_steps'] // params['save_params']['metric_freq'])
    # assert runner.dbinterface.collection.distinct('exp_id')[0] == params['exp_id']

    # # Run 100 more steps but save to a new experiment id and on different GPU.
    # params['exp_id'] = new_exp_id
    # previous_num_steps = params['train_params']['num_steps']
    # params['train_params']['num_steps'] = 200
    # params['model']['devices'] = 1
    # expected_num_records = (params['train_params']['num_steps'] - previous_num_steps) // params['save_params']['metric_freq']

    # runner = ptutils.runner.Runner.init(**params)
    # runner.train()

    # time.sleep(1)  # Wait for the morst recent record to finsh being saved to db.
    # assert runner.dbinterface.collection.find({'exp_id': params['exp_id']}).count() == expected_num_records
test_training()




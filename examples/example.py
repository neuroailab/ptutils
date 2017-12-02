"""Training MNIST with ptutils.

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
    'exp_id': "mnist_example_test",
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
        'devices': 1,

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
        'port': 27017,
        'host': 'localhost',
        'database_name': 'ptutils',
        'collection_name': 'ptutils'},

    'train_params': {
        'num_steps': 50,
        'train': True},

    'validation_params': {},

    'save_params': {
        'metric_freq': 25},

    'load_params': {
        'restore': True,
        'restore_params': None,
        'restore_mapping': None}}


runner = ptutils.runner.Runner.from_params(**params)
runner.train_from_params()

runner = ptutils.runner.Runner.from_params(**params)
runner.train_params['num_steps'] = 100
runner.train_from_params()
runner.dbinterface.collection.drop()
"""Training MNIST with ptutils.

The 'Hello, World!' of deep learning.

"""
import sys
import logging

import torch
import torch.nn as nn

sys.path.insert(0, '../')
import ptutils

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel('DEBUG')


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


class AverageMeter(ptutils.base.Base):
    """Compute and stores the average and current value."""

    def __init__(self, **kwargs):
        super(AverageMeter, self).__init__(**kwargs)
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class OnlineLossAggregator(ptutils.base.Base):

    def __init__(self, **kwargs):
        super(OnlineLossAggregator, self).__init__(**kwargs)
        self.loss_meter = AverageMeter()

    def __repr__(self):
        return 'Monitor'

    @staticmethod
    @ptutils.runner.Runner.after('step')
    def average_loss(runner):
        try:
            runner.online_aggregator.loss_meter.update(runner.model._loss.data[0])
        except AttributeError:
            pass
        log.info('step: {}; loss {:.4f} ({:.4f})'.format(runner.global_step,
                                                         runner.online_aggregator.loss_meter.val,
                                                         runner.online_aggregator.loss_meter.avg))

    # @staticmethod
    # @ptutils.model.Model.after('forward')
    # def method1(model):
        # try:
        # print(model)
        # self.loss_meter.update(model._loss.data[0])
        # except AttributeError:
        # pass
        # print('Loss {:.4f} ({:.4f})'.format(self.loss_meter.val,
        # self.loss_meter.avg))


# @ptutils.runner.Runner.after('step')
# def print_base(base):
#     print('----')
#     print(base)
#     print('----')


CUDA = 0
USE_CUDA = True
MONGO_PORT = 27017
exp_id = 'mnist_example'

# Experiment Params
params = {
    'func': ptutils.runner.Runner,
    'name': 'MNISTRunner',
    'exp_id': exp_id,
    'description': 'The \'Hello, World!\' of deep learning',

    # Define Model Params
    'model': {
        'func': ptutils.model.Model,
        'name': 'MNIST',
        'use_cuda': USE_CUDA,
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

    'online_aggregator': {
        'func': OnlineLossAggregator,
    },

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


runner = ptutils.runner.Runner.init(params)
runner.train()
runner.dbinterface.collection.drop()


import torch
import torch.nn as nn
import ptutils


class MNIST(torch.nn.Module, ptutils.base.Base):

    def __init__(self, **kwargs):
        super(MNIST, self).__init__()
        ptutils.base.Base(**kwargs)

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


class Criterion(ptutils.base.Base):
    pass


class Optimizer(ptutils.base.Base):
    pass


class Runner(ptutils.base.Base):
    pass


class DBInterface(ptutils.base.Base):
    pass


class Dataprovider(ptutils.base.Base):
    pass


# Experiment Params
params = {
    'func': ptutils.base.Runner,
    'name': 'MNIST Example',
    'exp_id': 'mnist_example',
    'description': 'The \'Hello, Wordl!\' of deep learning',

    # Define Model Params
    'model': {
        'func': ptutils.model.Model,
        'use_cuda': True,
        'name': 'MNIST',
        'devices': [0, 1],
        'net': {
            'func': MNIST,
            'name': 'mnist'},
        'criterion': {
            'func': Criterion,
            'name': 'crossentropy'},
        'optimizer': {
            'func': Optimizer,
            'name': 'Adam'}},

    # Define DataProvider Params
    'dataprovider': {
        'func': Dataprovider,
        'name': 'ImageNet'},

    # Define DBInterface Params
    'dbinterface': {
        'func': ptutils.database.MongoInterface,
        'name': 'mongo',
        'port': 27017,
        'host': 'localhost',
        'database_name': 'ptutils_db',
        'collection_name': 'ptutils_coll'}
}

runner = Runner.from_params(**params)

print(runner.to_params())

# new_runner = Runner.from_params(**to_params)

# print(new_runner)

# for name, param in params.items():
#     print('name: {} param: {}'.format(name, param))

# for name, param in model_params.items():
#     print('name: {} param: {}'.format(name, param))

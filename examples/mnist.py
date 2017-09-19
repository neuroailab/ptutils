"""Training MNIST with ptutils

The hello world example of deep learning.
"""
import torch.nn as nn
import torch.optim as optim

from ptutils.base import Runner
from ptutils.model import Model, ConvMNIST, FcMNIST


class MNISTModel(Model):

    def __init__(self, *args, **kwargs):
        super(MNISTModel, self).__init__(*args, **kwargs)
        self.net = MNIST()
        self.learning_rate = 1e-3
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), self.learning_rate)


class MNISTRunner(Runner):

    def __init__(self):
        super(MNISTTrainer, self).__init__()
        self.exp_id = 'my_experiment'
        self.model = MNISTModel()
        self.datastore = MongoDatastore('test_mnist', 'testcol')
        self.datasource = mnist.MNISTSource()

    def step(self, input, target):
        super(MNISTTrainer, self).step(input, target)

        # Save anything you would like
        self.datastore.save({'step': self.global_step,
                             'loss': self.model._loss.data[0]})

    def run(self):
        print(trainer.to_params())
        super(MNISTTrainer, self).run()


params = {
    'name': 'mnist_trainer',
    'exp_id': 'my_experiment',
    'model': {MNISTModel: {}},
    'my_datastore': {MongoDatastore: {'database_name': 'test_mnist',
                                      'collection_name': 'testcol'}},
    'my_datasource': {mnist.MNISTSource: {}}}


mnist_trainer = Trainer.from_params(params)
# OR
trainer = MNISTTrainer()
# trainer.run()

import sys

import torch.nn as nn
import torch.optim as optim
sys.path.insert(0, '.')
sys.path.insert(0, '..')
from ptutils.session import Session, Config
from ptutils.model import AlexNet, CNN, CIFARConv
from ptutils.database import MongoInterface
from ptutils.data import MNISTProvider, CIFARProvider

DB = 'ptutils_DEBUG'
COL = 'CIFAR'

config = {}
config['model'] = CIFARConv()
config['criterion'] = nn.CrossEntropyLoss()
config['optimizer'] = optim.Adam(config['model'].parameters())
config['data_provider'] = CIFARProvider()
config['db_interface'] = MongoInterface(db_name=DB, collection_name=COL)
config['run'] = {'num_epochs': 50,
                 'batch_size': 128}
sess = Session(config)
sess.run()

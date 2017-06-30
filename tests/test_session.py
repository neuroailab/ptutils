import sys

import torch
import torch.nn as nn
import torch.optim as optim
sys.path.insert(0, '.')
sys.path.insert(0, '..')
from ptutils.session import Session, Configuration
from ptutils.model import AlexNet, CNN, CIFARConv, Criterion, Optimizer
from ptutils.database import MongoInterface, DBInterface
from ptutils.data import MNISTProvider, ImageNetProvider, ImageNet, HDF5DataReader


db_config = {
    'port': 27017,
    'hostname': 'localhost',
    'db_name': 'ptutils',
    'collection_name': 'ptutils'}

# run_config = {'use_cuda': False}

# config_dict = {
#     CNN: {},
#     MNISTProvider: {},
#     MongoInterface: db_config,
#     nn.CrossEntropyLoss: {},
#     Optimizer: {}
# }


sess = Session()
sess.model = CNN()
sess.criterion = nn.CrossEntropyLoss()
sess.optimizer = optim.Adam(sess.model.parameters())
sess.data = MNISTProvider()
sess.dbinterface = MongoInterface(**db_config)
sess.status


sess.default_run()

# # DB = 'ptutils_DEBUG'
# # COL = 'CIFAR'

# config = {}
# config['model'] = CIFARConv()
# config['criterion'] = nn.CrossEntropyLoss()
# config['optimizer'] = optim.Adam(config['model'].parameters())
# config['data_provider'] = CIFARProvider()
# # config['data_provider'] = None
# config['db_interface'] = MongoInterface(db_name=DB, collection_name=COL)
# config['run'] = {'num_epochs': 50,
#                  'batch_size': 128}
# sess = Session(config)
# # sess.run()

# model_config = {
#     'num_classes': 10}

# dataset_config = {
#     ImageNet: {
#         HDF5DataReader: {},
#         'data_source': 'path/to/daa',
#         'transform': None}}

# config_dict = {
#     DBInterface: db_config,
#     AlexNet: model_config,
#     ImageNetProvider: dataset_config,
#     'session_name': 'test_session',
#     'description': 'a basic session to test ptutils',
#     'session_id': 'sess_0',
#     'use_cuda': False}

# sess_config_attr = Session()
# sess_config_attr.config = config
# # print(sess_attr.config)
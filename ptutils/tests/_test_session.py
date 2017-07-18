import sys
import pymongo as pm

import torch
import torch.nn as nn
import torch.optim as optim
sys.path.insert(0, '.')
sys.path.insert(0, '..')
# from ptutils.base import *
from ptutils.module import *
from ptutils.utils import sonify
from ptutils.session import Session
from ptutils.database import MongoInterface, DBInterface
from ptutils.model import AlexNet, CNN, CIFARConv, Criterion, Optimizer
from ptutils.data import MNISTProvider, ImageNetProvider, ImageNet, HDF5DataReader

port = 27017
db_name = 'ptutils'
hostname = 'localhost'
collection_name = 'ptutils'

config_file = 'config.yml'

config = {
    'sess_id': 0,
    'sess_name': 'ptutils_session',
    'description': 'A ptutils test session'
}

run_config = {
    'use_cuda': True,
}
dbinterface_config = {
    'port': port,
    'hostname': hostname,
    'db_name': db_name,
    'collection_name': collection_name}

save_config = {
    'save_valid_freq': 20,
    'save_filters_freq': 200,
    'cache_filters_freq': 100}

load_config = {
    'do_restore': True,
}

optimizer_config = {
    'optimizer': optim.Adam,
}

criterion_config = {
    'criterion': nn.CrossEntropyLoss,
}

config.update({
    CNN: {},
    MNISTProvider: {},
    Optimizer: optimizer_config,
    Criterion: criterion_config,
    MongoInterface: dbinterface_config,
})

# delete old database if it exists
conn = pm.MongoClient(host=hostname, port=port)
conn.drop_database(db_name)


c = Configuration(config_file)
sess = Session(c)

# candidates = sess._DEFAULT['DBInterface'].load({'sess_id': 0})
# print(candidates)

# sess.default_run()

# config['run'] = {'num_epochs': 50,
#                  'batch_size': 128}
# sess = Session(config)
# sess.run()

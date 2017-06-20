import sys

import torch.nn as nn
import torch.optim as optim
sys.path.insert(0, '.')
sys.path.insert(0, '..')
from ptutils.base import *
from ptutils.database import DBInterface, MongoInterface

class MongoConfig(dict):

    def __init__(self, config):
        super(MongoConfig, self).__init__()
        self.m_class = MongoInterface
        for key, value in config.items():
            self[key] = value

    def configure(self):
        return DBInterface(self)

config = {
    'port': 27017,
    'hostname': 'localhost',
    'db_name': 'ptutils_db_DEBUG',
    'collection_name': 'ptutils_col_DEBUG',
}

c = MongoConfig(config)
print(c)
mdn = c.configure()
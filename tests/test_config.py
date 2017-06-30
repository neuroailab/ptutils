import sys
import pprint
from collections import Mapping, MutableMapping, OrderedDict, defaultdict

import torch.nn as nn
import torch.optim as optim
sys.path.insert(0, '.')
sys.path.insert(0, '..')
from ptutils.base import Module, State
from ptutils.model import AlexNet
from ptutils.session import Session
from ptutils.utils import frozendict
from ptutils.data import ImageNetProvider, ImageNet, HDF5DataReader
from ptutils.database import DBInterface, MongoInterface


class Configuration(State):
    __name__ = 'config'

    def __init__(self, config_dict):
        super(Configuration, self).__init__()
        self._configs = OrderedDict()
        self._properties = OrderedDict()
        self._config_dict = config_dict
        self._hash = None
        for key, value in config_dict.items():
            if isinstance(key, type):
                if isinstance(value, dict):
                    self._configs[key.__name__] = (key, Configuration(value))

            elif isinstance(value, dict):
                key, value = self._find_configs(value)
                if isinstance(value, dict):
                    self._configs[key.__name__] = (key, Configuration(value))
            else:
                self[key] = value

    def configure(self):
        for name, (cls_, config) in self._configs.items():
            config.configure()
            self[name] = cls_(**config)
        return self

    # def state(self):
        # return self._config_dict

    def _find_configs(self, dict_):
        for key, value in dict_.items():
            if isinstance(key, type):
                return key, value
            if isinstance(value, dict):
                key, value = self._find_configs(value)
        return key, value

    def __setattr__(self, name, value):
            object.__setattr__(self, name, value)

    def __repr__(self):
        return pprint.pformat(self._config_dict)

    def __hash__(self):
        if self._hash is None:
            h = 0
            for key, value in self.items():
                h ^= hash((key, value))
            self._hash = h
        return self._hash


# db_config = {
#     'port': 27017,
#     'hostname': 'localhost',
#     'db_name': 'ptutils_db_DEBUG',
#     'collection_name': 'ptutils_col_DEBUG',}
# model_config = {'num_classes': 10,}
# dataset_config = {ImageNet: {'trouble': {'even_more_trouble': {HDF5DataReader: {},
#         'data_source': 'path_to_source'}}}}
# config = {'session_name': 'session_name',
#           'description': 'description of session (optional)',
#           'session_id': 'session id number (unique to the session',
#           MongoInterface: db_config,
#           AlexNet: model_config,
#           ImageNetProvider: dataset_config,
#           }


# c = Configuration(config)
# print(c.configure())
# print(c())
# print(c.state())

# sess = Session(c)
# print(type(c))
# print(type(sess))

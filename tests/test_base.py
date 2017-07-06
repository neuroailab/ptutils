import sys
import copy
import pprint
from collections import defaultdict, OrderedDict

import torch.nn as nn
import torch.optim as optim
sys.path.insert(0, '.')
sys.path.insert(0, '..')
from ptutils.base import *
from ptutils.model import AlexNet
from ptutils.session import Session
from ptutils.data import ImageNetProvider, ImageNet, HDF5DataReader
from ptutils.database import DBInterface, MongoInterface



# class Configuration(State, Module):
#     __name__ = 'configuration'

#     def __init__(self, config_dict):
#         State.__init__(self)
#         Module.__init__(self)
#         self._configs = OrderedDict()
#         self._config_dict = State(config_dict)
#         self._hash = None
#         self.state = State()
#         for key, value in config_dict.items():
#             if isinstance(key, type):
#                 if isinstance(value, dict):
#                     self._configs[key.__name__] = (key, Configuration(value))
#                     self.state[key.__name__] =  State(value)

#             elif isinstance(value, dict):
#                 key, value = self._find_configs(value)
#                 if isinstance(value, dict):
#                     self._configs[key.__name__] = (key, Configuration(value))
#                     self.state[key.__name__] = State(value)
#             else:
#                 self[key] = value

#         self.configure()

#     def configure(self):
#         for name, (cls_, config) in self._configs.items():
#             config.configure()
#             self[name] = cls_(**config)
#         return self

#     def state(self):
#         return SafeState(self.state)

#     def _find_configs(self, dict_):
#         for key, value in dict_.items():
#             if isinstance(key, type):
#                 return key, value
#             if isinstance(value, dict):
#                 key, value = self._find_configs(value)
#         return key, value

#     def __call__(self):
#         self.configure()

#     def __setattr__(self, name, value):
#             object.__setattr__(self, name, value)

#     # def __repr__(self):
#     #     return self._config_dict.__repr__()

#     def __hash__(self):
#         if self._hash is None:
#             h = 0
#             for key, value in self.items():
#                 h ^= hash((key, value))
#             self._hash = h
#         return self._hash

if __name__ == '__main__':
    db_config = {
        'port': 27017,
        'hostname': 'localhost',
        'db_name': 'ptutils_db_DEBUG',
        'collection_name': 'ptutils_col_DEBUG'}

    model_config = {
        'num_classes': 10}

    dataset_config = {
        ImageNet: {
            HDF5DataReader: {},
            'data_source': 'path/to/daa',
            'transform': None}}

    config_dict = {
        DBInterface: db_config,
        AlexNet: model_config,
        ImageNetProvider: dataset_config,
        'session_name': 'session_name',
        'description': 'description of session (optional)',
        'session_id': 'session id number (unique to the session'}

    # STATE -------------------------------------------------------------------
    state = State({'dict_key': 'dict_value',
                   'outer_dict_key': {'inner_dict_key': 'inner_dict_value'}},
                  keyword_arg1='keyword_value1',
                  keyword_arg2='keyword_value2')

    state.dot_set = 'test_dot_set'
    state['key_set'] = 'test_key_set'
    state.dict_dot_set = {'test_dict_dot_set_key': 'test_dict_dot_set_value'}
    state['dict_key_set'] = {'test_dict_key_set_key': 'test_dict_key_set_value'}
    state['key_set'] = 'test_key_set'
    state.ref = 'test_ref'
    state['ref'] = state.ref

    assert isinstance(state, dict)
    assert isinstance(state, Module)
    assert isinstance(state.dict_key, str)
    assert isinstance(state['dict_key'], str)
    assert isinstance(state.outer_dict_key, dict)
    assert isinstance(state['outer_dict_key'], dict)
    assert isinstance(state.outer_dict_key, State)
    assert isinstance(state['outer_dict_key'], State)
    assert isinstance(state.outer_dict_key, Module)
    assert isinstance(state['outer_dict_key'], Module)
    assert isinstance(state.dict_dot_set, dict)
    assert isinstance(state['dict_key_set'], dict)
    assert isinstance(state.dict_dot_set, State)
    assert isinstance(state['dict_key_set'], State)
    assert isinstance(state.dict_dot_set, Module)
    assert isinstance(state['dict_key_set'], Module)

    assert state.dict_key == state['dict_key']
    assert state.outer_dict_key['inner_dict_key'] == state['outer_dict_key'].inner_dict_key
    assert state.keyword_arg1 == state['keyword_arg1']
    assert state.dot_set == 'test_dot_set'
    assert state['key_set'] == 'test_key_set'
    assert state['ref'] == 'test_ref'
    state['ref'] = 'new_ref'
    assert state.ref == 'new_ref'
    assert state == state() and state == state.state() and state == state.load_state()

    # SONIFIED STATE -----------------------------------------------------------

    sonified_state = SonifiedState(config_dict)
    assert len(set(mape(type, sonified_state.keys()))) == 1
    assert str in set(map(type, sonified_state.keys()))
    print(sonified_state)

    # CONFIGURATION ------------------------------------------------------------
    config = Configuration(config_dict)
    config_dot_state = config.state()

    assert isinstance(config, State)
    assert isinstance(config, Module)
    assert isinstance(config_dot_state, State)
    assert sorted(config._configs.keys()) == ['AlexNet', 'DBInterface',
                                              'ImageNetProvider']

    config_dot_configure = config.configure()

    assert isinstance(config_dot_configure, Configuration)
    assert config.keys() == ['DBInterface', 'description', 'session_id',
                             'session_name', 'AlexNet', 'ImageNetProvider']

    # STATE -> CONFIGURATION ---------------------------------------------------

    state_config_dict = State(config_dict)
    config_state_config_dict = Configuration(state_config_dict)
    config_state_config_dict.configure()

    assert isinstance(state_config_dict, dict)
    assert isinstance(state_config_dict, State)
    assert sorted(state_config_dict.keys()) == sorted(config_dict.keys())

    assert isinstance(config_state_config_dict, State)
    assert isinstance(config_state_config_dict, Module)
    assert isinstance(config_state_config_dict, Configuration)

    assert config_state_config_dict.keys() == ['DBInterface', 'description',
                                               'session_id', 'session_name',
                                               'AlexNet', 'ImageNetProvider']

    m = Module(config_state_config_dict)

    assert m._modules.keys() == ['Configuration', 'DBInterface',
                                 'AlexNet', 'ImageNetProvider']
    assert m._properties.keys() == ['name', 'description',
                                    'session_id', 'session_name']

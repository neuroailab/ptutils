import yaml
import json
import cPickle as pickle

CONFIG_TYPES = {'yml': {'file': yaml.load, 'data': yaml.load},
                'yaml': {'file': yaml.load, 'data': yaml.load},
                'json': {'file': json.load, 'data': json.loads},
                'pkl': {'file': pickle.load, 'data': pickle.loads}}

# from .base import *

# _CORE_MODULES = {
#     Model.__dict__.get('__name__', Model.__name__): Model,
#     DBInterface.__dict__.get('__name__', DBInterface.__name__): DBInterface,
#     DataProvider.__dict__.get('__name__', DataProvider.__name__): DataProvider,
# }

# BASE_MODULES = {
#     Model.__dict__.get('__name__', Model.__name__): Model,
#     DBInterface.__dict__.get('__name__', DBInterface.__name__): DBInterface,
#     DataProvider.__dict__.get('__name__', DataProvider.__name__): DataProvider,
# }

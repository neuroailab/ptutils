import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')
from ptutils.base import *
from ptutils.model import CIFARConv
from ptutils.session import Session
from ptutils.data import DataProvider
from ptutils.database import DBInterface, MongoInterface


# Creating properties

# Anonymous property: it doesn't have a name attribute
# prop = Property('property_data')
# print(prop.name) --> None

# Named property
# prop1 = Property('property_data', name='Property_Name')
# print(prop1.name) --> 'Property_Name'

# class test(str):
    # def __init__(self, property):
        # super(test, self).__init__()

# t = test('db_name')
# DB = 'ptutils_DEBUG'
prop_group = PropertyGroup({'db_name': 'ptutils_db_DEBUG'},
                           {'collection_name': 'ptutils_col_DEBUG'})
# print(prop_group)


db = MongoInterface(db_name='ptutils_db_DEBUG',
                    collection_name='ptutils_col_DEBUG')
# dp = DataProvider(Module(), Module(name='m2'), prop1, name='dp_name')
# dp = DataProvider(Property('dp_prop_value', name='prop1'))
# Generate a session configuration
config = {'db_interface': DBInterface(PropertyGroup({'db_name': 'prop1_data'},
                                                    {'prop2_name': 'prop2_data'})),
          'model': CIFARConv(),
          'data_provider': DataProvider(PropertyGroup({'prop_name', 'prop_data'}))}

# mod1 = Module(name='mod1_name')
# mod2 = DBInterface()
# mod3 = Module()
# mod = Module(mod1, mod2, mod3)

sess = Session(config)
# print(sess._properties)
# print(sess._modules)


# db_prop = Property('db_prop')
# sess_prop = Property('sess_prop')

# db = Module(test_db_prop=db_prop)
# sess = Module(db=db, test_sess_prop=sess_prop)


# OLD
# mdb = MongoInterface(db_name='db_name',
#                      collection_name='collection_name')
# mdb.config = Property({'host': 'localhost',
#                        'port': 27017,
#                        'db_name': 'ptutils_db_name_DEBUG',
#                        'collection_name': 'putils_collection_name_DEBUG'})
# # data_provider = CIFARProvider()
# data_provider.config = Property({'datasets': ['CIFAR10', 'CIFAR100'],
#                                  'modes': ['train', 'test'],
#                                  'data_reader': 'ImageFolder'})
# model = CIFARConv()

# sess = Session()
# sess.config = {'session_id': 1,
#                'description': 'Test Description',
#                'status': {'num_run': 1,
#                           'run_id': {'init': '[new/resume/restart]',
#                                      'start_date': 'test_start_date',
#                                      'state': '[pending/in-progress/complete]',
#                                      'progress': ' 0 \% complete',
#                                      'eta': 'test_eta',
#                                      'errors': '[error1, error2, error3, ...]',
#                                      'end_date': 'test_end_date',
#                                      'outcome': '[success/failure/other]'}},
#                'history': '(optional) {instance_prop1: val, intance_prop2: val, ...}',
#                'subsessions': '[subsess1, subsess2, subsess3, ...]'}
# sess.model = model
# sess.db_interface = mdb
# sess.data_provider = data_provider

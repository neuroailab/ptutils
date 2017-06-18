import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')
from ptutils.base import *
from ptutils.model import CNN
from ptutils.session import Session
from ptutils.database import DBInterface, MongoInterface
from ptutils.data import DataProvider, MNISTProvider, Dataset

model = CNN()
data = MNISTProvider()
db = MongoInterface(db_name='ptutils_DEBUG', collection_name='ptutils_DEBUG')

sess = Session()
sess.db = db
sess.data = data
sess.model = model

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

# import sys
# sys.path.insert(0, '.')
# sys.path.insert(0, '..')
# import torch
# from ptutils.base import *
# from ptutils.session import Session
# from ptutils.model import Model, CNN, AlexNet
# from ptutils.database import DBInterface, MongoInterface
# from ptutils.data import DataProvider, MNISTProvider, Dataset

# model = CNN()
# data = MNISTProvider()
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters())
# db = MongoInterface(db_name='ptutils_DEBUG', collection_name='ptutils_DEBUG')
# # alexnet = AlexNet()
# # model = Model({'cnn': cnn, 'alexnet': alexnet})

# sess = Session()
# sess.db = db
# sess.data = data
# sess.model = model
# sess.criterion = criterion
# sess.optimizer = optimizer

# # These are all session configs
# config = {}
# # Make type: SessionProperty
# # config['sess'] or
# config['info'] = {
#     'session_name': 'session_name',
#     'description': 'description of session (optional)',
#     'session_id': 'session id number (unique to the session',
# }

# # Make type: SessionProperty
# config['save'] = {
#     'save_valid_freq': 20,
#     'save_filters_freq': 200,
#     'cache_filters_freq': 100,
# }

# # Make type: SessionProperty or ModelProperty
# config['train'] = {
#     'group': 'train',
#     'num_gpus': 4,
#     'num_steps': 500,
#     'num_epochs': 10,
#     'num_threads': 4,
#     'batch_size': 100,
# }

# config['criterion'] = {
#     'staircase': True,
#     'decay_rate': 0.95,
#     'learning_rate': 0.05,
#     'decay_steps': 'num_batches_per_epoch',
# }

# config['optimizer'] = {
#     'staircase': True,
#     'decay_rate': 0.95,
#     'learning_rate': 0.05,
#     'decay_steps': 'num_batches_per_epoch',
# }

# config['validation'] = {
#     'num_gpus': 4,
#     'num_steps': 10,
#     'group': 'test',
#     'num_epochs': 10,
#     'num_threads': 4,
#     'batch_size': 100,
#     'agg_func': 'utils.mean_dict',
# }

# config['db_interface'] = {
#     'port': 27017,
#     'hostname': 'localhost',
#     'db_name': 'ptutils_db_DEBUG',
#     'collection_name': 'ptutils_col_DEBUG',
# }

# config['data_provider'] = {
#     'provider_name': 'CustomProvider',
#     'provide_modes': 'TBD: Likely dict of kwargs',
#     'datasets': {
#         'CIFAR': {
#             'transform': None,
#             'data_format': 'ImageFolder',
#             'data_source': 'path/to/data',
#             'data_reader': 'ImageFolerReader'},
#         'ImageNet': {
#             'data_format': 'HDF5',
#             'data_source': 'path/to/data',
#             'data_reader': 'HDF5DataReader',
#             'transform': {
#                 'ToTensor': True,
#                 'RandomSizedCrop': 224,
#                 'RandomHorizontalFlip': True,
#                 'Normalize': {
#                     'std': [0.229, 0.224, 0.225],
#                     'mean': [0.485, 0.456, 0.406]
#                 }
#             }
#         }
#     }
# }

# # model.state_dict().keys()
# # NOT RECOMMENDED
# config['model'] = [
#     'cnn.layer1.0.weight',
#     'cnn.layer1.0.bias',
#     'cnn.layer1.1.weight',
#     'cnn.layer1.1.bias',
#     'cnn.layer1.1.running_mean',
#     'cnn.layer1.1.running_var',
#     'cnn.layer2.0.weight',
#     'cnn.layer2.0.bias',
#     'cnn.layer2.1.weight',
#     'cnn.layer2.1.bias',
#     'cnn.layer2.1.running_mean',
#     'cnn.layer2__1.running_var',
#     'cnn.fc.weight',
#     'cnn.fc.bias',
# ]

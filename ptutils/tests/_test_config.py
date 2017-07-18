import re
import sys
import yaml
import pprint

import torch.nn as nn
import torch.optim as optim
sys.path.insert(0, '.')
sys.path.insert(0, '..')
from ptutils.model import AlexNet
from ptutils.session import Session
from ptutils.utils import frozendict
# from ptutils.module import Module, State, Configuration
from ptutils.base import Module, Configuration
from ptutils.database import DBInterface, MongoInterface
from ptutils.data import ImageNetProvider, ImageNet, HDF5DataReader


CONFIG_FILE = 'resources/config.yml'

c = Configuration(CONFIG_FILE)
sess = Session(c)
# print(c)
# print(sess)

import sys
import copy
import pprint
from collections import defaultdict, OrderedDict

import torch.nn as nn
import torch.optim as optim
sys.path.insert(0, '.')
sys.path.insert(0, '..')
from ptutils.module import *

m = Module()
n = Module()
l = Module()

l.l = 'l'
l.j = 'j'
n.l = l
n.k = 'k'
m.n = n
m.l = 'l'
s = m.state()

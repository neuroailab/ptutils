"""ptutils model.py

Encapsulates a neural network model, criterion and optimizer.

"""
import random

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.parallel import data_parallel

from ptutils.base import Base

class Model(Base):

    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)

        # Core
        # self._net = None
        # self._criterion = None
        # self._optimizer = None

        # GPU and dtype business
        # This should go in the params 'specification (spec)'
        self._dtype = 'float'
        self._devices = None
        self._use_cuda = torch.cuda.is_available()
        # self._use_cuda = False

        # self.model = torch.nn.DataParallel(self.model).cuda()

        # use_cuda = torch.cuda.is_available()
        # dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

        # if use_cuda:
        #     self.model.cuda()
        #     self.criterion.cuda()

        # # Select mode
        # if mode == 'train':
        #     self.model.train()
        #     volatile = False
        # else:
        #     self.model.eval()
        #     volatile = True

        self._loss = None

    def output(self, input):
        input_var = Variable(input)
        if self._devices is not None:
            return data_parallel(self.net, input_var, list(self._devices))
        else:
            return self.net(input_var)

    def loss(self, output, target):
        target_var = Variable(target)
        loss = self.criterion(output, target_var)
        return loss

    def compute_gradients(self, loss=None):
        loss.backward()

    def apply_gradients(self):
        self.optimizer.step()

    def optimize(self, loss=None):
        self.compute_gradients(loss=loss)
        self.apply_gradients()
        self.optimizer.zero_grad()

    def forward(self, input, target):
        output = self.output(input)
        self._loss = self.loss(output, target)
        self.optimize(self._loss)

    @property
    def net(self):
        return self._bases['net']

    @net.setter
    def net(self, value):
        self._bases['net'] = value

    @property
    def criterion(self):
        return self._bases['criterion']

    @criterion.setter
    def criterion(self, value):
        self._bases['criterion'] = value

    @property
    def optimizer(self):
        return self._bases['optimizer']

    @optimizer.setter
    def optimizer(self, value):
        self._bases['optimizer'] = value


class MNIST(nn.Module):

    def __init__(self, devices=1):
        super(MNIST, self).__init__()

        self.devices = devices
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class DynamicNet(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H)
        self.middle_linear = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.input_linear(x).clamp(min=0)
        for _ in range(random.randint(0, 3)):
            h_relu = self.middle_linear(h_relu).clamp(min=0)
        y_pred = self.output_linear(h_relu)
        return y_pred


class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, self.num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


class CIFARConv(nn.Module):

    def __init__(self, num_classes=10):
        super(CIFARConv, self).__init__()
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def reset_classifier(self, num_classes=10):
        self.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 4 * 4)
        x = self.classifier(x)
        return x


class CIFARConvOld(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFARConv, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 20, kernel_size=5, padding=2),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(20, 20, kernel_size=5, padding=2),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(4 * 4 * 20, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def re_init_fc(self, num_classes=10):
        self.fc = nn.Linear(4 * 4 * 20, num_classes)

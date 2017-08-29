"""ptutils base module.

"""
from __future__ import print_function

import os
import re
import logging
import collections

import torch
import torch.nn as nn
import torch.optim as optim

from error import StepError

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel('DEBUG')

if 'PTUTILS_HOME' in os.environ:
    PTUTILS_HOME = os.environ['PTUTILS_HOME']
else:
    PTUTILS_HOME = os.path.join(os.environ['HOME'], '.ptutils')

DEFAULT_LOAD_PARAMS = {'do_restore': True}
DEFAULT_LOSS_PARAMS = {'func': nn.CrossEntropyLoss}
DEFAULT_OPTIMIZER_PARAMS = {'func': optim.SGD,
                            'momentum': 0.9,
                            'lr': 0.05}


class Base(object):

    def __init__(self, *args, **kwargs):
        self._bases = collections.OrderedDict()
        self._params = collections.OrderedDict()

        self.name = kwargs.get('name', type(self).__name__.lower())

        for i, arg in enumerate(args):

            if isinstance(arg, Base):
                self.__setattr__(arg._name, arg)

            if isinstance(arg, collections.Mapping):
                for key, value in arg.items():
                    setattr(self, key, value)

        if kwargs:
            for key, value in kwargs.items():
                setattr(self, key, value)

    # @property
    # def name(self):
        # return self._name
        # self._params['name']

    # @name.setter
    # def name(self, value):
        # self._name = value
        # self._params['name'] = value

    def to_params(self):
        params = collections.OrderedDict()
        state_dict = collections.OrderedDict()
        for name, param in self._params.items():
            if param is not None:
                params[name] = param
        for name, base in self._bases.items():
            try:
                params[name] = base.to_params()
            except AttributeError as params_error:
                try:
                    state_dict[name] = base.state_dict().keys()
                except AttributeError as state_error:
                    log.warning(params_error + state_error)
        return params

    @classmethod
    def from_params(cls, params):
        for key, value in params.items():
            if isinstance(key, type):
                if isinstance(value, collections.Mapping):
                    return key.from_params(value)
            elif isinstance(value, collections.Mapping):
                params[key] = cls.from_params(value)
        return cls(**params)

    def to_state(self, destination=None, prefix=''):
        """Return a dictionary containing a whole state of the module."""
        if destination is None:
            destination = collections.OrderedDict()
        for name, base in self._bases.items():
            if isinstance(base, torch.nn.Module):
                base.state_dict(destination, prefix + name + '.')
            else:
                base.to_state(destination, prefix + name + '.')
        return destination

    def from_state(self, state, restore_params=None, param_mapping=None):
        """Restore base to state."""
        own_state = self.to_state()

        # Determine which params to restore from state.
        if restore_params is None:
            restore_params = state.keys()
        elif isinstance(restore_params, re._pattern_type):
            restore_params = [name for name in state.keys()
                              if restore_params.match(name)]
        elif isinstance(restore_params, list):
            restore_params = [name for name in state.keys()
                              if name in restore_params]
        else:
            raise TypeError('restore_params ({}) unsupported.'
                            .format(type(restore_params)))

        if param_mapping is None:
            # Use identity mapping if None.
            param_mapping = {name: name for name in state.keys()}
        else:
            # print('before')
            # print(param_mapping)
            param_mapping.update({name: name for name in state.keys()
                                  if name not in param_mapping})
            # print('after')
            # print(param_mapping)
        for name, param in state.items():
            if name in restore_params:
                own_state[param_mapping[name]].copy_(param)

    def restore_state(self, state, restore_params):
        """Filter state params for those to be restored.

        Args:
            state (dict): A state_dict.
            restore_params (list[str] or regex): Specifies params to restore.

        Returns:
            TYPE: Description.

        Raises:
            TypeError: restore_params type is unsupported.

        """
        if restore_params is None:
            restore_params = state.keys()
        elif isinstance(restore_params, re._pattern_type):
            return [name for name in state.keys()
                    if restore_params.match(name)]
        elif isinstance(restore_params, list):
            return [name for name in state.keys()
                    if name in restore_params]
        raise TypeError('restore_params ({}) unsupported.'.format(type(restore_params)))

    def __setattr__(self, name, value):
        if isinstance(value, (Base, torch.nn.Module)):
            self._bases[name] = value
        else:
            if not name.startswith('_'):
                self._params[name] = value
        object.__setattr__(self, name, value)

    def __getattr__(self, name):
        if name in self._bases:
            return self._bases[name]
        elif name in self._params:
            return self._params[name]
        elif name in self.__dict__:
            return self.__dict__[name]
        else:
            raise AttributeError("'{}' object has no attribute '{}'"
                                 .format(type(self).__name__, name))

    def __setitem__(self, name, item):
        self.__setattr__(name, item)

    def __getitem__(self, name):
        return self.__getattr__(name)

    def __repr__(self):
        """Return module string representation."""
        repstr = '{} ({}): ('.format(type(self).__name__, self.name)
        if self._bases:
            repstr += '\n'
        for name, base in self._bases.items():
            basestr = base.__repr__()
            basestr = _addindent(basestr, 2)
            repstr += '  {}\n'.format(basestr)
        repstr = repstr + ')'
        return repstr


class Runner(Base):

    def __init__(self, model=None, dbinterface=None, dataprovider=None, **kwargs):
        super(Runner, self).__init__(model=None,
                                     dbinterface=None,
                                     dataprovider=None,
                                     **kwargs)

        # Core
        self._model = None
        self._dbinterface = None
        self._dataprovider = None

        self._exp_id = None
        self._global_step = 0
        self._state_dict = collections.OrderedDict()

        # Params
        self._save_params = None
        self._load_params = None

        self._loss_params = None
        self._train_params = None
        self._optimizer_params = None

    @classmethod
    def from_params(cls, **params):
        runner = cls()
        runner.save_params = params.get('save_params', None)
        runner.load_params = params.get('load_params', None)

        model_params = params.get('model_params', None)
        runner.get_model(**model_params)

        dbinterface_params = params.get('dbinterface_params', None)
        runner.get_dbinterface(**dbinterface_params)

        dataprovider_params = params.get('dataprovider_params', None)
        runner.get_dataprovider(**dataprovider_params)

        runner._params['loss_params'] = params.get('loss_params', DEFAULT_LOSS_PARAMS)
        runner._params['optimizer_params'] = params.get('optimizer_params', DEFAULT_OPTIMIZER_PARAMS)
        return runner

    def predict(self):
        pass

    def test(self):
        pass

    def test_from_params(self):
        pass

    @property
    def dbinterface(self):
        return self._bases['dbinterface']

    @dbinterface.setter
    def dbinterface(self, value):
        self._bases['dbinterface'] = value

    def get_dbinterface(self, func, **dbinterface_params):
        self._params['dbinterface_params'] = dbinterface_params
        self.dbinterface = func(**dbinterface_params)

    @property
    def exp_id(self):
        return self._exp_id

    @exp_id.setter
    def exp_id(self, value):
        self._exp_id = value

    @property
    def global_step(self):
        return self._global_step

    @global_step.setter
    def global_step(self, value):
        if value <= self._global_step:
            raise StepError('The global step should have been incremented.')
        elif value > (self._global_step + 1):
            raise StepError('The global step can only be incremented by one.')
        else:
            self._global_step = value

    def step(self, input, target):
        self.global_step += 1
        """Define a single step of an experiment.

        This must increment the global step. A common use case
        will be to simply make a forward pass update the model.

        Formally, this will call model.forward(), whose output should
        be used by the dataprovider to provide the next batch of data.

        """
        output = self.model(input, target)
        # print('step: {}; loss: {}'.format(self.global_step,
                                          # self.model._loss.data[0]))

    def train(self, dataloader):
        """Define the primary training loop.

        The default is to just step the trainer.

        """
        # Step the Trainer
        for input, target in dataloader:
            self.step(input, target)

            # You may want to do additional computation
            # in between steps.

    def train_from_params(self):
        """Run the execution of an experiment.

        This is the primary entrance to the Trainer class.

        """
        assert self.exp_id is not None, 'Must provide and exp_id'
        # if self.load_params['do_restore']:
            # self.load_run()

        # Do any initialization needed here
        input = self.datasource.provide()

        # Start the main training loop.
        self.train(input)

        # Perhaps you do validation at this point

        # Do any cleanup needed to conclude the experiment.

    def load_run(self):
        params = self.dbinterface.load({'exp_id': self.exp_id})
        if params is not None:
            return self.from_params(**params)
        else:
            return self

    @property
    def model(self):
        """Get the model."""
        return self._bases['model']

    @model.setter
    def model(self, value):
        self._bases['model'] = value

    def get_model(self, func, **model_params):
        self._params['model_params'] = model_params
        self.model = func(**model_params)

    @property
    def save_params(self):
        """Get the model."""
        return self._params['save_params']

    @save_params.setter
    def save_params(self, value):
        self._params['save_param'] = value

    @property
    def load_params(self):
        """Get the model."""
        return self._params['load_params']

    @load_params.setter
    def load_params(self, value):
        self._params['load_param'] = value

    @property
    def optimizer(self):
        """Get the optimizer."""
        return self._bases['optimizer']

    @optimizer.setter
    def optimizer(self, value):
        if isinstance(value, str) or callable(value):
            self.get_optimizer(value)
        elif isinstance(value, dict):
            self.get_optimizer(**value)
        else:
            raise NotImplementedError

    def get_optimizer(self, func, param_groups=None, **optimizer_params):
        """Build the optimizer for training.

        Args:
            func (str or callable): Optimizer to .
            param_groups (None, optional): Description.
            **optimizer_params: Description.

        Returns:
            TYPE: Description.

        Raises:
            NotImplementedError: Description.

        """
        if isinstance(func, str):
            optimizer_class = getattr(torch.optim, func, None)
            assert optimizer_class is not None, "Optimizer {} not found.".format(
                func)
        elif callable(func) and isinstance(func, type):
            optimizer_class = func
        elif isinstance(func, torch.optim.Optimizer):
            self._optimizer = func
            return self
        else:
            raise NotImplementedError
        param_groups = self.model.parameters() if param_groups is None else param_groups
        self._optimizer = optimizer_class(param_groups, **optimizer_params)
        return self

    @property
    def dataprovider(self):
        return self._bases['dataprovider']

    @dataprovider.setter
    def dataprovider(self, value):
        self._bases['dataprovider'] = value

    def get_dataprovider(self, func, **dataprovider_params):
        self._params['dataprovider_params'] = dataprovider_params
        self.dataprovider = func(**dataprovider_params)


def _addindent(string, numSpaces):
    s = string.split('\n')
    # dont do anything for single-line stuff
    if len(s) == 1:
        return string
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s

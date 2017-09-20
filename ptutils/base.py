"""ptutils base module.

This module defines the base ptutils class that should subclassed by all
subsequent ptutils classes. This class generates useful information about
itself that can be used to recreate and resume experiments exactly as they
were.

"""
from __future__ import print_function

import os
import re
import logging
import collections

import torch
import torch.nn as nn
import torch.optim as optim

from error import StepError, ExpIdError, ParamError

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel('DEBUG')

if 'PTUTILS_HOME' in os.environ:
    PTUTILS_HOME = os.environ['PTUTILS_HOME']
else:
    PTUTILS_HOME = os.path.join(os.environ['HOME'], '.ptutils')

DEFAULT_LOAD_PARAMS = {'restore': False,
                       'restore_params': None,
                       'restore_mapping': None}

DEFAULT_LOSS_PARAMS = {'func': nn.CrossEntropyLoss}
DEFAULT_OPTIMIZER_PARAMS = {'func': optim.SGD,
                            'momentum': 0.9,
                            'lr': 0.05}


class Base(object):

    def __init__(self, *args, **kwargs):
        self._bases = collections.OrderedDict()
        self._params = collections.OrderedDict()

        self.devices = None
        self.use_cuda = False

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

    @property
    def name(self):
        return self._params['name']

    @name.setter
    def name(self, value):
        self._params['name'] = value

    @property
    def devices(self):
        return self._params['devices']

    @devices.setter
    def devices(self, value):
        self._params['devices'] = value

    @property
    def use_cuda(self):
        return self._params['use_cuda']

    @use_cuda.setter
    def use_cuda(self, value):
        self._params['use_cuda'] = value

    def to_params(self):
        params = collections.OrderedDict()
        params['func'] = self.__class__
        state_dict = collections.OrderedDict()
        for name, param in self._params.items():
            if param is not None:
                params[name] = param
        for name, base in self._bases.items():
            try:
                params[name] = base.to_params()
            except AttributeError as params_error:
                try:
                    params[name] = collections.OrderedDict({'func': base.__class__})
                    state_dict[name] = base.state_dict().keys()
                except AttributeError as state_error:
                    log.warning(str(params_error) + str(state_error))

        return params

    @classmethod
    def from_params(cls, **params):
        if 'func' in params:
            func = params['func']
            for key, value in params.items():
                if isinstance(value, dict):
                    params[key] = func.from_params(**value)
            return func(**params)
        else:
            return params

    @classmethod
    def _from_params(cls, **params):
        try:
            func = params['func']
        except KeyError:
            pass
            # raise ParamError('Param key \'func\' not provided.')
        else:
            for key, value in params.items():
                if isinstance(value, dict):
                    params[key] = func.from_params(**value)
            return func(**params)

    def to_state(self, destination=None, prefix=''):
        """Return a dictionary containing a whole state of the module.

        TODO: CAVEAT GOES HERE

        """
        if destination is None:
            destination = collections.OrderedDict()
        for name, base in self._bases.items():
            if isinstance(base, torch.nn.Module):
                try:
                    base.state_dict(destination, prefix + name + '.')
                except AttributeError as state_error:
                    log.warning(state_error)
            else:
                base.to_state(destination, prefix + name + '.')
        return destination

    def from_state(self, state, restore_params=None, restore_mapping=None):
        """Restore base to the state specified by `state`.

        Args:
            state (dict): Names mapped to parameters.
            restore_params (list[str] or regex, optional): Params to restore.
                If a list, elements must be the names of params to restore.
                If a regex, it must match all the param names to be restored.
                If None, attempts to restore all parameters.
                Defaults to None.
            restore_mapping (dict, optional): Maps old param names to new names.
                Defaults to None.

        Returns:
            Base: self.

        Raises:
            TypeError: restore_params type unsupported.

        """
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

        # Determine the restore params mapping.
        if restore_mapping is None:
            # Use identity mapping if None.
            restore_mapping = {name: name for name in state.keys()}
        else:
            restore_mapping.update({name: name for name in state.keys()
                                    if name not in restore_mapping})
        for name, param in state.items():
            if name in restore_params:
                own_state[restore_mapping[name]].copy_(param)

        return self

    def apply(self, fn):
        """Apply``fn`` recursively to every subbase as well as self.

        Typical use applying dataparallel to all modules.

        Args:
            fn (function or static method): Function to apply to all subbases.

        Returns:
            Base: self

        """
        for base in self._bases.values():
            base.apply(fn)
        fn(self)
        return self

    def cuda(self, devices=None):
        """Move all Bases to the GPU.

        Args:
            devices(list, optional): if specified, all parameters will be
                copied to that device

        """
        self.use_cuda = True
        if self.devices is None:
            self.devices = devices

        for base in self._bases.values():
            base.use_cuda = True
            if isinstance(base, torch.nn.Module):
                base.cuda()
            else:
                base.cuda(devices=self.devices)

    def cpu(self):
        """Move all Bases to the CPU."""
        self.use_cuda = False
        self.devices = None
        for base in self._bases.values():
            base.cpu()

    def cast(self, obj):

        if isinstance(obj, (list, tuple)):
            return type(obj)([self.cast(o) for o in obj])
        else:
            obj = obj.cuda() if self.use_cuda else obj
            return obj.type(self.dtype) if dtype is None else obj.type(dtype)

    def __setattr__(self, name, value):
        if isinstance(value, (Base, torch.nn.Module)):
            self._bases[name] = value
        else:
            # Allow this , just restrict for _params.
            if name not in ['_params', '_bases']:
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
    """Summary.

    Attributes:
        dataprovider (TYPE): Description.
        dbinterface (TYPE): Description.
        exp_id (TYPE): Description.
        global_step (int): Description.
        load_params (TYPE): Description.
        model (TYPE): Description.
        save_params (TYPE): Description.
        train_params (TYPE): Description.

    """

    def __init__(self,
                 exp_id,
                 model=None,
                 dbinterface=None,
                 dataprovider=None,
                 train_params=None,
                 save_params=None,
                 load_params=None,
                 **kwargs):
        """Initialize the :class:`Runner` class.

        Args:
            exp_id (str): Description.
            model (Model, optional): Description.
            dbinterface (DBInterface, optional): Description.
            dataprovider (DataProvider, optional): Description.
            train_params (dict, optional): Description.
            save_params (dict, optional): Description.
            load_params (dict, optional): Description.
            **kwargs: Additional attr required by runner.

        """
        super(Runner, self).__init__(**kwargs)

        # Core bases.
        self.model = model
        self.dbinterface = dbinterface
        self.dataprovider = dataprovider

        # Params.
        self.save_params = save_params
        self.load_params = load_params
        self.train_params = train_params

        self.exp_id = exp_id
        self.global_step = 0

# -- Runner Properties ---------------------------------------------------------

    @property
    def exp_id(self):
        return self._params['exp_id']

    @exp_id.setter
    def exp_id(self, value):
        self._params['exp_id'] = value

    @property
    def global_step(self):
        return self._params['global_step']

    @global_step.setter
    def global_step(self, value):
        # if value <= self._params['global_step']:
            # raise StepError('The global step should have been incremented.')
        if value > (self._params['global_step'] + 1):
            raise StepError('The global step can only be incremented by one.')
        elif value < 0:
            raise StepError('The global step cannot be negative.')
        else:
            self._params['global_step'] = value

    @property
    def model(self):
        """Get the model."""
        return self._bases['model']

    @model.setter
    def model(self, value):
        self._bases['model'] = value

    @property
    def dbinterface(self):
        return self._bases['dbinterface']

    @dbinterface.setter
    def dbinterface(self, value):
        self._bases['dbinterface'] = value

    @property
    def dataprovider(self):
        return self._bases['dataprovider']

    @dataprovider.setter
    def dataprovider(self, value):
        self._bases['dataprovider'] = value

    @property
    def save_params(self):
        """Get the save parameters."""
        return self._params['save_params']

    @save_params.setter
    def save_params(self, value):
        self._params['save_params'] = value

    @property
    def load_params(self):
        """Get the load parameters."""
        return self._params['load_params']

    @load_params.setter
    def load_params(self, value):
        self._params['load_params'] = value

    @property
    def train_params(self):
        """Get the train parameters."""
        return self._params['train_params']

    @train_params.setter
    def train_params(self, value):
        self._params['train_params'] = value

# -- Runner Methods ------------------------------------------------------------

    def step(self, prev_output):
        """Define a single step of an experiment.

        This must increment the global step. A common use case
        will be to simply make a forward pass update the model.

        Formally, this will call model.forward(), whose output should
        be used by the dataprovider to provide the next batch of data.

        """
        prev_output = None
        data = self.dataprovider.provide(prev_output)
        output = self.model.step(data)

        print('step: {}; loss: {}'.format(self.global_step,
                                          output['loss'].data[0]))
        return output

    def train(self):
        """Define the primary training loop.

        The default behavior is to step the trainer and
        save intermediate results.

        """
        model_output = None
        for step in range(self.train_params['num_steps']):
            model_output = self.step(model_output)

            # if self.global_step % self.save_params['metric_freq'] == 0:
            #     # Save desired results.
            #     record = {'exp_id': self.exp_id,
            #               'step': self.global_step,
            #               'loss': model_output['loss'].data[0]}
            #     self.dbinterface.save(record)
            # if val_freq % 0:
                # val_model_output = None
                # for val_step in self.validation_params['num_steps']
                    # val_model_output = self.validation_step(val_model_output)
            # You may want to do additional computation
            # in between steps.

            self.global_step += 1

    def train_from_params(self):
        """Run the execution of an experiment.

        This is the primary entrance to the Trainer class.

        """
        # Enforce that all Runners have an exp_id.
        if self.exp_id is None:
                raise ExpIdError('Cannot run an experiment without an exp_id')

        # Restore previous run.
        if self.load_params['restore']:
            self.load_run()

        # Prepare devices.
        self.cuda(devices=[0, 1])

        # Start the main training loop.
        self.train()

    def predict(self):
        # TODO
        pass

    def test(self):
        # TODO
        pass

    def test_from_params(self):
        # TODO
        pass

    def load_run(self):
        params = self.dbinterface.load({'exp_id': self.exp_id})
        # TODO: Raise exc if not found of exp_id collisions.
        if params is not None:
            return self.from_params(**params)
        else:
            return self


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

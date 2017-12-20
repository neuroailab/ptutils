"""ptutils base module.

This module defines the base ptutils class that should subclassed by all
subsequent ptutils classes. This class generates useful information about
itself that can be used to recreate and resume experiments exactly as they
were.

"""
from __future__ import print_function

import os
import re
import copy
import logging
import collections
from collections import Iterable
import torch
import torch.nn as nn
import torch.optim as optim

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel('DEBUG')


class Base(object):

    def __init__(self, *args, **kwargs):
        self._bases = collections.OrderedDict()
        self._params = collections.OrderedDict()

        self.devices = None
        self.use_cuda = False

        self.name = kwargs.get('name', type(self).__name__.lower())

        for i, arg in enumerate(args):

            if isinstance(arg, Base):
                self.__setattr__(arg.name, arg)

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
        """Generate dictionary representation of base.

        The params dict of a given base contains the following key-value
        pairs:
            'func' (cls): its class
            'name' (str): name of base
            'devices' (list[ints]): devices to which it belongs
            'use_cuda' (bool): whether base should be moved to its devices

        """
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
            # params is itself a base.
            func = params['func']
            for key, value in params.items():
                if isinstance(value, dict):
                    params[key] = func.from_params(**value)
            return func(**params)
        else:
            # params isn't a base.
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

    def _to_state(self, destination=None, prefix=''):
        """Return a dictionary containing a whole state of the module.

        TODO: CAVEAT GOES HERE

        """
        if destination is None:
            destination = collections.OrderedDict()
        for name, base in self._bases.items():
            if isinstance(base, (torch.nn.Module, torch.optim.Optimizer)):
                try:
                    base.state_dict(destination, prefix + name + '.')
                except TypeError as state_error:
                    log.warning(state_error)
                    try:
                        destination[name] = base.state_dict()
                    except TypeError:
                        pass
            else:
                base.to_state(destination, prefix + name + '.')
        return destination

    def to_state(self, destination=None, prefix=''):
        """Return a dictionary containing a whole state of the module."""
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

    def base_cuda(self, devices=None):
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
                base.cuda(self.devices)
            else:
                base.base_cuda(devices=self.devices)

    def base_cpu(self):
        """Move all Bases to the CPU."""
        self.use_cuda = False
        self.devices = None
        for base in self._bases.values():
            base.use_cuda = False
            if isinstance(base, torch.nn.Module):
                base.cpu()
            else:
                base.base_cpu()

    def cast(self, obj):

        if isinstance(obj, (list, tuple)):
            return type(obj)([self.cast(o) for o in obj])
        else:
            obj = obj.cuda() if self.use_cuda else obj #this cuda call may need to be changed
            return obj.type(self.dtype) if dtype is None else obj.type(dtype)

    def __setattr__(self, name, value):
        if isinstance(value, (Base, torch.nn.Module)):
            self._bases[name] = value
        elif isinstance(value, list):
            # Check to see if it's a list of bases
            # If so, make it covertly into a BaseList and then save it to _bases
            if isinstance(value[0], Base):
                baselist = BaseList(value)
                self._bases[name] = baselist
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

    
class BaseList(Base):
    """Holds subBases in a list. Modeled after the torch.nn.ModuleList

    BaseList can be indexed like a regular Python list, but basess it
    contains are properly registered, and will be visible by all Base methods.

    Arguments:
        bases (iterable, optional): an iterable of bases to add
    """

    def __init__(self, bases=None):
        super(BaseList, self).__init__()
        if bases is not None:
            self += bases

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        return self._bases[str(idx)]

    def __setitem__(self, idx, base):
        return setattr(self, str(idx), base)

    def __len__(self):
        return len(self._bases)

    def __iter__(self):
        return iter(self._bases.values())

    def __iadd__(self, bases):
        return self.extend(bases)

    def __dir__(self):
        keys = super(BaseList, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def append(self, base):
        r"""Appends a given module to the end of the list.

        Arguments:
            module (nn.Module): module to append
        """
        self.add_module(str(len(self)), base)
        return self

    def add_base(self, name, base):
        """Adds a child module to the current module.

        The module can be accessed as an attribute using the given name.

        Args:
            name (string): name of the child module. The child module can be
                accessed from this module using the given name
            parameter (Module): child module to be added to the module.
        """
        if not isinstance(base, Base) and base is not None:
            raise TypeError("{} is not a Module subclass".format(
                torch.typename(base)))
        if hasattr(self, name) and name not in self._bases:
            raise KeyError("attribute '{}' already exists".format(name))
        self._bases[name] = base 

    def extend(self, bases):
        r"""Appends modules from a Python iterable to the end of the list.

        Arguments:
            modules (iterable): iterable of modules to append
        """
        if not isinstance(bases, Iterable):
            raise TypeError("ModuleList.extend should be called with an "
                            "iterable, but got " + type(bases).__name__)
        offset = len(self)
        for i, bases in enumerate(bases):
            self.add_base(str(offset + i), bases)
        return self


def _addindent(string, numSpaces):
    s = string.split('\n')
    # Don't do anything for single-line stuff.
    if len(s) == 1:
        return string
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s

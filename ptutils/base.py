"""ptutils base module.

This module defines the base ptutils class that should subclassed by all
subsequent ptutils classes. This class generates useful information about
itself that can be used to recreate and resume experiments exactly as they
were.

"""
from __future__ import print_function

import re
import sys
import inspect
import logging
import traceback
import collections
from functools import wraps
from collections import Iterable, OrderedDict

import torch

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel('DEBUG')


def decorator(function):
    name = function.__name__
    def wrapped(self, *args, **kwargs):
        result = function(self, *args, **kwargs)
        try:
            self.TAG_STORE[type(self).__name__ + '.' + name](self)
        except KeyError:
            pass
        return result
    return wrapped


class MetaBase(type):
    def __new__(meta, class_name, bases, class_dict):
        for name, item in class_dict.items():
            if callable(item) and not item.__name__.startswith('_'):
            # if callable(item):
                class_dict[name] = decorator(item)
        return type.__new__(meta, class_name, bases, class_dict)


class Base(object):
    __metaclass__ = MetaBase
    TAG_STORE = {}

    def __init__(self, *args, **kwargs):

        self.devices = None
        self.name = kwargs.get('name', type(self).__name__.lower())
        if not hasattr(self, '_exclude_from_params'):
            self._exclude_from_params = []
        self._ptutils_tags = {}

        # this prevents the backend for torch.nn.Modules
        # from being serialized in the database
        self._exclude_from_params.append('_backend')
        self._exclude_from_params.append('_modules')

        for i, arg in enumerate(args):

            if isinstance(arg, Base):
                self.__setattr__(arg.name, arg)

            if isinstance(arg, collections.Mapping):
                for key, value in arg.items():
                    setattr(self, key, value)

        if kwargs:
            for key, value in kwargs.items():
                setattr(self, key, value)

    @classmethod
    def after(cls, tagged_method):
        TAG_STORE = '_ptutils_tags'

        def wrapper(method):
            cls.TAG_STORE[cls.__name__ + '.' + tagged_method] = method

            @wraps(method)
            def wrapped(self, *method_args, **method_kwargs):
                return method(self, *method_args, **method_kwargs)

            return wrapped

        return wrapper

    def to_params(self):
        # params_dict = self._to_params(self)
        # if ['func'] not in params_dict.keys():
        #     params_dict['func'] = type(self)
        # return params_dict
        
        return self._to_params(self)

    # @classmethod
    def _to_params(self, value):
        """Generate dictionary representation of base.

        The params dict of a given base contains the following key-value
        pairs:
            'func' (cls): its class
            'name' (str): name of base
            'devices' (list[ints]): devices to which it belongs
            'use_cuda' (bool): whether base should be moved to its devices

        """
        if isinstance(value, (Base, torch.nn.Module, torch.optim.Optimizer)):
            value.func = value.__class__
            # if hasattr(value, '_to_params'):
            #     return value._to_params({k: v for k, v in value.__dict__.items()
            #                              if k not in self._exclude_from_params})
            # else:
            #     return {'func': value.func}
            # if isinstance(value, torch.nn.Module):
            #     print('mod',value)
                
            # elif isinstance(value, torch.optim.Optimizer):
            #     print('optim', value)
            # elif isinstance(value, Base):
            #     print('Base', value)
            try:
                return value._to_params({k: v for k, v in value.__dict__.items()
                                         if k not in self._exclude_from_params})
            except AttributeError as e:
                # print('errored', value)
                return {'func': value.func}
                # return self._to_params({k: v for k, v in value.__dict__.items()
                                         # if k not in self._exclude_from_params})
        elif isinstance(value, (dict, OrderedDict)):
            dictfunc = type(value)
            return dictfunc({k: self._to_params(v) for k, v in value.items()
                    if isinstance(k, (str, unicode)) and k not in self._exclude_from_params})
        elif isinstance(value, list) and len(value) > 0:
            return [self._to_params(v) for v in value]
        elif isinstance(value, tuple) and len(value) > 0:
            return tuple(self._to_params(v) for v in value)
        
        else:
            return value

    @classmethod
    def from_params(cls, params):
        if isinstance(params, dict):
            if 'func' in params:  # Assume we are given a func dictionary
                func = params['func']
                d = {k: cls.from_params(v) for k, v in params.items()}
                # d = {}
                # for k,v in params.items():
                    
                #     try:
                #         d[k] = cls.from_params(v)
                #     except:
                #         d[k] = v
                try:
                    return func(**d)
                # except:
                except Exception, e:
                    log.error(str(e))
                    traceback.print_exc()
                    print('could not make: ', func)
                    sys.exit(1)
            else:
                # Othwerwise, call from_params on dict values
                return {k: cls.from_params(p) for k, p in params.items()}
        elif isinstance(params, list):
            return [cls.from_params(p) for p in params]
        elif isinstance(params, tuple):
            return tuple(cls.from_params(p) for p in params)
        
        else:
            # params isn't a base.
            return params

    def to_state(self, destination=None, prefix=''):
        """Return a dictionary containing a whole state of the module."""
        bases, _ = self._get_bases_and_params()
        if destination is None:
            destination = collections.OrderedDict()
        for name, base in bases.items():
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
        elif isinstance(restore_params, tuple):
            restore_params = tuple(name for name in state.keys()
                              if name in restore_params)
            
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

    def assign_devices(self, devices=None):
        """Recursively assign devices to children bases/modules.

        Args:
            devices(list, optional): if specified, all parameters will be
                copied to that device
        """
        if self.devices is None:
            self.devices = devices
        bases, _ = self._get_bases_and_params()
        for base in bases.values():
            try:
                base.assign_devices(devices=self.devices)
            except AttributeError:  # Allow non-base, native torch.nn.Modules
                pass

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
            obj = obj.cuda() if self.use_cuda else obj  # this cuda call may need to be changed
            return obj.type(self.dtype) if dtype is None else obj.type(dtype)

    def _get_bases_and_params(self):
        bases = {}
        params = {}

        if 'func' not in self.__dict__.keys():
            params['func'] = type(self)

        for key, value in self.__dict__.items():
            if isinstance(value, (Base, torch.nn.Module)) and key != 'parent':
                bases[key] = value
            elif isinstance(value, list) and len(value) > 0 and all(isinstance(x, Base) for x in value):
                value = BaseList(value)

                bases[key] = value
            else:
                params[key] = value
        return bases, params

    def __setitem__(self, name, item):
        self.__setattr__(name, item)

    def __getitem__(self, name):
        return self.__getattr__(name)

    # def __repr__(self):
        """Return module string representation."""
        # repstr = '{} ({}): ('.format(type(self).__name__, self.name)
        # if self._bases:
        #     repstr += '\n'
        # for name, base in self._bases.items():
        #     basestr = base.__repr__()
        #     basestr = _addindent(basestr, 2)
        #     repstr += '  {}\n'.format(basestr)
        # repstr = repstr + ')'
        # return str(self.to_params())

class BaseList2(Base):
    def __init__(self, bases=None, *args, **kwargs):
        for idx, base in enumerate(bases):
            setattr(self, 'base_'+str(idx), base)
        super(BaseList2, self).__init__(*args, **kwargs)
class BaseList(Base):
    """Hold subBases in a list. Modeled after the torch.nn.ModuleList.

    BaseList can be indexed like a regular Python list, but basess it
    contains are properly registered, and will be visible by all Base methods.

    Arguments:
        bases (iterable, optional): an iterable of bases to add
    """

    def __init__(self, bases=None, *args, **kwargs):

        self._bases = {}
        if bases is not None:
            self += bases
        super(BaseList, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        return self._bases['base' + str(idx)]

    def __setitem__(self, idx, base):
        return setattr(self, 'base' + str(idx), base)

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
        """Append a given module to the end of the list.

        Arguments:
            module (nn.Module): module to append
        """
        self.add_base('base' + str(len(self)), base)
        return self

    def add_base(self, name, base):
        """Add a child module to the current module.

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
        """Append modules from a Python iterable to the end of the list.

        Arguments:
            modules (iterable): iterable of modules to append
        """
        if not isinstance(bases, Iterable):
            raise TypeError("ModuleList.extend should be called with an "
                            "iterable, but got " + type(bases).__name__)
        offset = len(self)
        for i, bases in enumerate(bases):
            self.add_base('base' + str(offset + i), bases)
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

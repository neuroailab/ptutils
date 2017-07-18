""""ptutils base module.

This module contains the ptutils base class definitions:

    - `Module`
    - `State`
    - `Status`
    - `Configuration`

At the core of PTUtils is the `Module`class, the base class for all ptutils
classes that attempts to generalize PyTorch's existing `torch.nn.Module`.
A `Module` is an arbitrary, container-like class that fulfills three simple
requirements:

1. A `Module` must be callable. A `Module` may maintain any desired number
of public and private methods, although it must separately implement the
`__call__` method, which may simply map to one of its other methods.

2. A `Module` must implement a `state()` method that returns an instance of
a `State` module. This state module should reflect the current 'state' of the
module and can be explicity specified by the user.

3. A `Module` must implement a `load_state()` method that restores the module
to the state described by a given state module.

A `State` module (henceforth referred to as the state) serves as a specialized
'identity' module that preserves the following:

```python
    s = s(*args, **kwargs)
      = s.state(*args, **kwargs)
      = s.load_state(*args, **kwargs),
```
where `s` is an instance of the `State` class.

Critically, a module can register and operate other modules as regular
attributes, allowing users to nest them in a tree structure. All other,
non-module attributes are considered to be 'properties' of that module.
By default, the state module returned by a module's `state` method contains
the properties of that module and the state module

Enforcing this simple API attempts to address the notion that the environment
in which a neural network operates should be free to evolve dynamically just
as the network itself is.


"""

import os
import re
import copy
import json
import yaml
import pprint
import warnings
from collections import OrderedDict

import torch.nn as nn
from torch.autograd import Variable

from utils import parse_config, sonify


class Module(object):
    """Base class for all ptutils modules."""

    __name__ = 'module'
    __base__ = 'Module'

    def __init__(self, *args, **kwargs):
        self._properties = OrderedDict()
        self._modules = OrderedDict()
        self._state = State()

        for i, arg in enumerate(args):

            if isinstance(arg, Module):
                self.add_module(arg.__name__, arg)

            if isinstance(arg, dict):
                for key, value in arg.items():
                    setattr(self, key, value)

        if kwargs:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def register_property(self, name, prop):
        """Add a property to the module.

        The property can be accessed as an attribute using given name.

        """
        if '_properties' not in self.__dict__:
            raise AttributeError(
                "cannot assign property before Module.__init__() call")
        if prop is None:
            self._properties[name] = None
        else:
            self._properties[name] = prop

    def properties(self):
        """Return an iterator over module properties.

        Example:
            >>> for prop in model.properties():
            >>>     print(prop)

        """
        for name, prop in self.named_properties():
            yield prop

    def named_properties(self, memo=None, prefix=''):
        """Return an iterator over module property names and values.

        Example:
            >>> for name, prop in self.named_properties():
            >>>    if name in ['status']:
            >>>        print(prop)
        """
        if memo is None:
            memo = set()
        for name, p in self._properties.items():
            if p is not None and p not in memo:
                memo.add(p)
                yield prefix + ('.' if prefix else '') + name, p
        for mname, module in self.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            if not isinstance(module, Module) and isinstance(module, nn.Module):
                for name, p in module.named_parameters(memo, submodule_prefix):
                    yield name, p
            else:
                for name, p in module.named_properties(memo, submodule_prefix):
                    yield name, p

    def add_module(self, name, module):
        """Add a child module to the current module.

        The module can be accessed as an attribute using the given name.

        """
        if hasattr(self, name):
            raise KeyError("attribute already exists '{}'".format(name))
        if not isinstance(module, Module) and module is not None:
            raise TypeError("{} is not a Module subclass".format(type(module)))
        self._modules[name] = module

    def _add_nameless_module(self, module):
        """Add an unamed child module to the module using its class name.

        The module can be accessed as an attribute using the its class name.

        """
        cls_name = module.__class__.__name__
        warnings.warn('A nameless module was provided. ' +
                      ' Defaulting to its class name {}'.format(cls_name))
        self.add_module(cls_name, module)

    def modules(self):
        """return an iterator over all modules in the module.

        note:
            duplicate modules are returned only once. in the following
            example, ``mod`` will be returned only once.

            >>> mod = linear(2, 2)
            >>> net = nn.sequential(l, l)
            >>> for idx, m in enumerate(net.modules()):
            >>>     print(idx, '->', m)
            0 -> sequential (
              (0): linear (2 -> 2)
              (1): linear (2 -> 2)
            )
            1 -> linear (2 -> 2)
        """
        for name, module in self.named_modules():
            yield module

    def module_names(self):
        """return an iterator over all modules in the module.

        note:
            duplicate modules are returned only once. in the following
            example, ``mod`` will be returned only once.

            >>> mod = linear(2, 2)
            >>> net = nn.sequential(l, l)
            >>> for idx, m in enumerate(net.modules()):
            >>>     print(idx, '->', m)
            0 -> sequential (
              (0): linear (2 -> 2)
              (1): linear (2 -> 2)
            )
            1 -> linear (2 -> 2)
        """
        for name, module in self.named_modules():
            yield name

    def named_modules(self, memo=None, prefix=''):
        """return an iterator over all modules in the module.

        yields both the name of the module as well as the module itself.

        note:
            duplicate modules are returned only once. in the following
            example, ``l`` will be returned only once.

            >>> l = nn.linear(2, 2)
            >>> net = nn.sequential(l, l)
            >>> for idx, m in enumerate(net.named_modules()):
            >>>     print(idx, '->', m)
            0 -> ('', sequential (
              (0): linear (2 -> 2)
              (1): linear (2 -> 2)
            ))
            1 -> ('0', linear (2 -> 2))

        """
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                submodule_prefix = prefix + ('.' if prefix else '') + name
                for m in module.named_modules(memo, submodule_prefix):
                    yield m

    def children(self):
        """return an iterator over immediate children modules."""
        for name, module in self.named_children():
            yield module

    def children_names(self):
        """return an iterator over immediate children modules."""
        for name, module in self.named_children():
            yield name

    def named_children(self):
        """return an iterator over immediate children modules.

        yields both the name of the module as well as the module itself.

        example:
            >>> for name, module in model.named_children():
            >>>     if name in ['db_interface', 'data_provider']:
            >>>         print(module)

        """
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module

    def state(self, destination=None, prefix=''):
        """Return the State module defining the state of the Module.

        Example:
            >>> module.state().keys()
            ['config', 'model', 'db_interface', 'data_provider']

        """
        # TODO: STATE_DICT KEYS SHOULD CONTAIN STANDARDIZED
        # CLASS NAMES! NOT ATTRIBUTE NAMES...OTHERWISE,
        # MODULE EQUIVALENCY MAY BE DISRUPTED DUE TO ATTR NAMES

        if destination is None:
            destination = State()
        for name, prop in self._properties.items():
            if prop is not None:
                destination[prefix + name] = prop
        for name, module in self._modules.items():
            if module is not None:
                try:
                    module.state(destination, prefix + name + '.')
                except Exception:
                    module.state_dict(destination, prefix + name + '.')
        return destination

    def state2(self):
        state = State()
        for name, prop in self._properties.items():
            if prop is not None:
                state[name] = prop
        for name, module in self._modules.items():
            if module is not None:
                try:
                    state[name] = module.state()
                except Exception:
                    state[name] = module.state_dict()
        return state


    def load_state(self, state):
        """Copy properties from state into this module and its descendants.

        The keys of :attr:`state` must exactly match the keys
        returned by this module's :func:`state` function.

        TODO: BE MORE FLEXIBLE!

        Args:
            state (State): A State module.

        """
        own_state = self.state()
        for name, prop in state.items():
            if name not in own_state:
                raise KeyError('unexpected key "{}" in state'
                               .format(name))
            if isinstance(prop, Property):
                # backwards compatibility for serialized parameters
                prop = prop
            own_state[name].copy_(prop)

        missing = set(own_state.keys()) - set(state.keys())
        if len(missing) > 0:
            raise KeyError('missing keys in state: "{}"'.format(missing))

    def __getattr__(self, name):
        """Return module attribute `name`."""
        if '_properties' in self.__dict__:
            _properties = self.__dict__['_properties']
            if name in _properties:
                return _properties[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        if name in self.__dict__:
            return self.__dict__[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def __setattr__(self, name, value):
        """Parse value type and register modules and properties as needed."""
        def remove_from(*dicts):
            for d in dicts:
                if name in d:
                    del d[name]

        modules = self.__dict__.get('_modules')
        if isinstance(value, Module):
            if modules is None:
                raise AttributeError(
                    "cannot assign module before Module.__init__() call")
            modules[name] = value
        elif modules is not None and name in modules:
            if value is not None:
                raise TypeError("cannot assign '{}' as child module '{}' "
                                "(ptutils.Module or None expected)"
                                .format(type(value), name))
            modules[name] = value

        else:
            object.__setattr__(self, name, value)
            if not name.startswith('_'):
                self.register_property(name, value)

    def __delattr__(self, name):
        """Remove module attribute `name`."""
        if name in self._properties:
            del self._properties[name]
        elif name in self._modules:
            del self._modules[name]
        else:
            object.__delattr__(self, name)

    def __repr__(self):
        """Return module string representation."""
        tmpstr = self.__class__.__name__ + ' (\n'
        for key, module in self._modules.items():
            modstr = module.__repr__()
            modstr = _addindent(modstr, 2)
            tmpstr = tmpstr + '  (' + str(key) + '): ' + modstr + '\n'
        tmpstr = tmpstr + ')'
        return tmpstr

    def __dir__(self):
        """Return module dir()."""
        module_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        properties = list(self._properties.keys())
        modules = list(self._modules.keys())
        keys = module_attrs + attrs + properties + modules
        return sorted(keys)

    def __getitem__(self, key):
        """Return module item."""
        return self.__getattr__(key)

    def __setitem__(self, name, value):
        """Specify module item."""
        self.__setattr__(name, value)


# class State(dict, Module):
class State(dict):
    """Base module for representing module state."""

    __name__ = 'state'

    def __init__(self, *args, **kwargs):
        """Initialize State module."""
        super(State, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v
        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def state(self, *args, **kwargs):
        """Return state."""
        return self

    def load_state(self, *args, **kwargs):
        """Load state."""
        return self

    def __call__(self, *args, **kwargs):
        """Return state."""
        return self

    def __getitem__(self, name):
        """Return item `name`."""
        return dict.__getitem__(self, name)

    def __setitem__(self, name, value):
        """Set item `name` to `value`."""
        if isinstance(value, dict):
            dict.__setitem__(self, name, State(value))
        else:
            dict.__setitem__(self, name, value)

    def __delattr__(self, name):
        """Delete attribute `name`."""
        return dict.__delitem__(self, name)

    def __getattr__(self, name):
        """Return attribute `name`."""
        return self.__getitem__(name)

    def __setattr__(self, name, value):
        """Set attribute `name` to `value`."""
        self.__setitem__(name, value)

    def __dir__(self):
        """Return dir."""
        return self.keys() + dir(dict(self))

    def __deepcopy__(self, memo):
        """Return deepcopy."""
        return State(copy.deepcopy(dict(self)))


class SonifiedState(State):
    __name__ = 'sonified_state'

    def __init__(self, *args, **kwargs):
        super(SonifiedState, self).__init__()
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v
        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __setitem__(self, name, value):
        if isinstance(name, type):
            if isinstance(value, type):
                dict.__setitem__(self, name.__name__,
                                 (sonify(name), sonify(value)))
                # dict.__setitem__(self, name.__name__, value.__name__)
            elif isinstance(value, dict):
                dict.__setitem__(self, name.__name__,
                                 (sonify(name), SonifiedState(value)))
                # dict.__setitem__(self, name.__name__, SonifiedState(value))
            else:
                dict.__setitem__(self, name.__name__,
                                 (sonify(name), sonify(value)))
                # dict.__setitem__(self, name.__name__, value)
        elif isinstance(value, type):
            dict.__setitem__(self, name, (sonify(name), sonify(value)))
            # dict.__setitem__(self, name, value.__name__)
        else:
            dict.__setitem__(self, name, (sonify(name), sonify(value)))
            # dict.__setitem__(self, name, value)


class Configuration(Module):
    """Configure a module or group of modules from a Configuration state.

    A Configuration module is a dictionary that specifies the configuration of
    its parent module.

    """

    __name__ = 'configuration'

    def __init__(self, config):
        """Initialize Configuration module."""
        super(Configuration, self).__init__()

        config_dict = parse_config(config)

        self._state = State(config_dict)
        self._configs = OrderedDict()

        for key, value in config_dict.items():
            if isinstance(key, type):
                if isinstance(value, dict):
                    name = key.__dict__.get('__name__', key.__name__)
                    self._configs[name] = (key, Configuration(value))
            elif isinstance(value, dict):
                key, value = self._find_configs(value)
                if isinstance(value, dict):
                    name = key.__dict__.get('__name__', key.__name__)
                    self._configs[name] = (key, Configuration(value))
            else:
                self[key] = value
        self.configure()

    def configure(self):
        for name, (cls_, config) in self._configs.items():
            config.configure()
            self[name] = cls_(**config._state)
        return self

    # def state(self, sonified=False):
    #     if sonified:
    #         return SonifiedState(self._state)
    #     else:
    #         return self._state

    def __call__(self):
        self.configure()

    def __repr__(self):
        return self._state.__repr__()

    def _find_configs(self, dict_):
        for key, value in dict_.items():
            if isinstance(key, type):
                return key, value
            if isinstance(value, dict):
                key, value = self._find_configs(value)
        return key, value

    def _parse(self, config):
        if isinstance(config, str):
            pattern = r'.(yml|yaml|json|pkl)$'
            m = re.search(pattern, config, flags=re.I)
            if m is not None:
                ext = m.group().lower()


class Status(Module):
    """Verify a Module's status, compatibility and progress.

    TODO: each module type specifies its own requirement
    """
    __name__ = 'status'

    def __init__(self, *args, **kwargs):
        super(Module, self).__init__()

    def verify(self, module=None):
        log.info('Determining module status ...')
        """Compare the config to the db to get status of the session."""
        # CALL _get_status whenever a module or property is added.

        # TODO:
        # (1) Query database for previous status.
        # (2) If it doesn't exist, create it.
        # (3) If it does exist, verify it.

        # Take module inventory.

        # Partition core session components by class.
        for name, module in self.named_modules():
            for core_name, core_module in self._CORE_MODULES.items():
                if isinstance(module, core_module):
                    if not self._DEFAULT[core_name]:
                        self._DEFAULT[core_name] = module
                        self._MISSING.pop(core_name)

        # valid sess if all components are present and config is compatible
        valid_sess = True if (not self._MISSING) else False

        # for db in db_interfaces:
        # # TODO: look for previous sessions
        #     candidates = db.load(self.config)
        #     for candidate in candidates:
        #     # TODO: Attempt to load missing components
        #     # TODO: Attempt to reuse previous experiements
        #         pass
        # # TODO: By scanning for previous sessions:
        # # should determine run number, new/

        if valid_sess:
            log.info('Current session is valid!')
        else:
            log.warning('Current session is NOT valid. Please review ' +
                        'the session\'s configuration before continuing')

        status = {
            'valid_session': valid_sess,
            'cuda_available': False,
            'use_cuda': False,
            'device_count': 1,
            'num_run': 0,
            'run_id': {
                'init': 'new',
                'start_date': datetime.datetime.now(),
                'state': 'pending',
                'progress': 0,
                'eta': 'N/A',
                'errors': [],
                'end_date': 'end date',
                'outcome': 'N/A'}}

        self.configuration['status'] = status
        # self._DEFAULT['DBInterface'].save(self.Configuration.state())
        return status
        pass


def _addindent(s_, numSpaces):
    s = s_.split('\n')
    # dont do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s

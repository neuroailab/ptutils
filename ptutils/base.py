""""
    Module containing base ptutils objects.
"""

from collections import OrderedDict


class Module(object):

    def __init__(self):
        self._modules = OrderedDict()

    def add_module(self, name, module):
        """Adds a child module to the current module.

        The module can be accessed as an attribute using the given name.
        """
        if hasattr(self, name):
            raise KeyError("attribute already exists '{}'".format(name))
        if not isinstance(module, Module) and module is not None:
            raise TypeError("{} is not a Module subclass".format(type(module)))
        self._modules[name] = module

    def modules(self):
        """Returns an iterator over all modules in the module.

        Note:
            Duplicate modules are returned only once. In the following
            example, ``mod`` will be returned only once.

            >>> mod = Linear(2, 2)
            >>> net = nn.Sequential(l, l)
            >>> for idx, m in enumerate(net.modules()):
            >>>     print(idx, '->', m)
            0 -> Sequential (
              (0): Linear (2 -> 2)
              (1): Linear (2 -> 2)
            )
            1 -> Linear (2 -> 2)
        """
        for name, module in self.named_modules():
            yield module

    def named_modules(self, memo=None, prefix=''):
        """Returns an iterator over all modules in the module, yielding
        both the name of the module as well as the module itself.

        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.

            >>> l = nn.Linear(2, 2)
            >>> net = nn.Sequential(l, l)
            >>> for idx, m in enumerate(net.named_modules()):
            >>>     print(idx, '->', m)
            0 -> ('', Sequential (
              (0): Linear (2 -> 2)
              (1): Linear (2 -> 2)
            ))
            1 -> ('0', Linear (2 -> 2))
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
        """Returns an iterator over immediate children modules."""
        for name, module in self.named_children():
            yield module

    def named_children(self):
        """Returns an iterator over immediate children modules, yielding both
        the name of the module as well as the module itself.

        Example:
            >>> for name, module in model.named_children():
            >>>     if name in ['db_interface', 'data_provider']:
            >>>         print(module)
        """
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module

    def state_dict(self, destination=None, prefix=''):
        """Returns a dictionary containing a whole state of the Module.

        All PTModule subclasses assigned as regular attributes are
        included. Keys are corresponding PTModule names.

        Example:
            >>> module.state_dict().keys()
            ['config', 'model', 'db_interface', 'data_provider']
        """
        if destination is None:
            destination = OrderedDict()
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination, prefix + name + '.')
        return destination

    def load_state_dict(self, state_dict):
        """Copies Modules from :attr:`state_dict` into
        this module and its descendants. The keys of :attr:`state_dict` must
        exactly match the keys returned by this module's :func:`state_dict()`
        function.

        Args:
            state_dict (dict): A dict containing modules.
        """
        own_state = self.state_dict()
        for name, module in state_dict.items():
            if name not in own_state:
                raise KeyError('unexpected key "{}" in state_dict'
                               .format(name))
            if isinstance(module, Module):
                own_state[name] = module.state_dict()

        missing = set(own_state.keys()) - set(state_dict.keys())
        if len(missing) > 0:
            raise KeyError('missing keys in state_dict: "{}"'.format(missing))

    def __getattr__(self, name):
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def __setattr__(self, name, value):
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

    def __delattr__(self, name):
        if name in self._modules:
            del self._modules[name]
        else:
            object.__delattr__(self, name)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        for key, module in self._modules.items():
            modstr = module.__repr__()
            modstr = _addindent(modstr, 2)
            tmpstr = tmpstr + '  (' + key + '): ' + modstr + '\n'
        tmpstr = tmpstr + ')'
        return tmpstr

    def __dir__(self):
        module_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        modules = list(self._modules.keys())
        keys = module_attrs + attrs + modules
        return sorted(keys)


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

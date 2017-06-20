""""
    Module containing base ptutils objects.
"""
import warnings
from collections import OrderedDict, MutableMapping

from  torch.autograd import Variable

from .utils import Map

DEFAULT_REQUIRES_SAVE = True
DEFAULT_SAVE_FREQ = 100

"""
TODO: REMOVE Property class!
If a module attr is not another module,
then it is a property! The property name is the attribute name
and the data is the attr value.
"""


class Module(object):

    __name__ = 'module'

    def __init__(self, *args, **kwargs):
        # TODO: UPDATE THIS TO REFLECT CHANGES TO PROPERTIES
        self.name = None
        self._modules = OrderedDict()
        self._properties = OrderedDict()
        if self.name is not None:
            self.__name__ = self.name

        for i, arg in enumerate(args):

            if isinstance(arg, Property) or isinstance(arg, Module):
                self._set_named_arg(arg, i)

            if isinstance(arg, dict):
                for key, value in arg.items():
                    setattr(self, key, value)

            if isinstance(arg, list):
                for j, value in enumerate(arg):
                    if isinstance(arg, Property) or isinstance(arg, Module):
                        self._set_named_arg(arg, i, j)

        if kwargs:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def _set_named_arg(self, arg, i, j=None):
        # TODO: ELMINATE THIS!
        if arg.name is not None:
            setattr(self, arg.name, arg)
        else:
            cls_name = arg.__class__.__name__
            if j is not None:
                default_name = '{}_{}_{}'.format(cls_name, i, j)
            else:
                default_name = '{}_{}'.format(cls_name.lower(), i)
            warnings.warn('{} does not have a name. Defaulting to: {}'.
                          format(cls_name, default_name))
            setattr(self, '{}_{}'.format(default_name, i), arg)

    def register_property(self, name, prop):
        """Adds a property to the module.

        The property can be accessed as an attribute using given name.
        """
        if '_properties' not in self.__dict__:
            raise AttributeError(
                "cannot assign property before Module.__init__() call")
        if prop is None:
            self._property[name] = None
        elif not isinstance(prop, Property):
            raise TypeError("cannot assign '{}' object to property '{}' "
                            "(ptutils.Property or None required)"
                            .format(type(prop), name))
        else:
            if prop.name is None:
                prop.name = name
            elif prop.name is not None and prop.name != name:
                warnings.warn('Property name ({}) '.format(prop.name) +
                              'does not match attr name ({}). '.format(name) +
                              'Proceed with caution ...')
            self._properties[name] = prop

    def properties(self):
        """Returns an iterator over module properties.

        Example:
            >>> for prop in model.properties():
            >>>     print(prop)
        """
        for name, prop in self.named_properties():
            yield prop

    def named_properties(self, memo=None, prefix=''):
        """Returns an iterator over module properties, yielding both the
        name of the properties as well as the property itself.

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
            for name, p in module.named_properties(memo, submodule_prefix):
                yield name, p

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

    def module_names(self):
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
            yield name

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

    def children_names(self):
        """Returns an iterator over immediate children modules."""
        for name, module in self.named_children():
            yield name

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

        # TODO: STATE_DICT KEYS SHOULD CONTAIN STANDARDIZED
        # CLASS NAMES! NOT ATTRIBUTE NAMES...OTHERWISE, 
        # MODULE EQUIVALENCY MAY BE DISRUPTED DUE TO ATTR NAMES

        if destination is None:
            destination = OrderedDict()
        for name, prop in self._properties.items():
            if prop is not None:
                destination[prefix + name] = prop.data
                destination[prefix + name] = prop
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination, prefix + name + '.')
        return destination

    def load_state_dict(self, state_dict):
        """Copies properties from :attr:`state_dict` into
        this module and its descendants. The keys of :attr:`state_dict` must
        exactly match the keys returned by this module's :func:`state_dict()`
        function.

        TODO: BE MORE FLEXIBLE!

        Args:
            state_dict (dict): A dict containing modules.
        """
        own_state = self.state_dict()
        for name, prop in state_dict.items():
            if name not in own_state:
                raise KeyError('unexpected key "{}" in state_dict'
                               .format(name))
            if isinstance(prop, Property):
                # backwards compatibility for serialized parameters
                prop = prop
            own_state[name].copy_(prop)

        missing = set(own_state.keys()) - set(state_dict.keys())
        if len(missing) > 0:
            raise KeyError('missing keys in state_dict: "{}"'.format(missing))

    def __getattr__(self, name):
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

        props = self.__dict__.get('_properties')
        if isinstance(value, Property):
            if props is None:
                raise AttributeError(
                    "cannot assign property before Module.__init__() call")
            remove_from(self.__dict__, self._modules)
            self.register_property(name, value)
        elif props is not None and name in props:
            if value is not None:
                raise TypeError("cannot assign '{}' as property '{}' "
                                "(ptutils.Property or None expected)"
                                .format(type(value), name))
            self.register_property(name, value)

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
        if name in self._properties:
            del self._properties[name]
        elif name in self._modules:
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
        properties = list(self._properties.keys())
        modules = list(self._modules.keys())
        keys = module_attrs + attrs + properties + modules
        return sorted(keys)

    def __getitem__(self, key):
        return self.__getattr__(key)

    def __setitem__(self, name, value):
        return self.__setattr__(name, value)


class Configuration(Module):
    """Configure a module or group of modules from a Configuration state.

    A Configuration module is a dictionary that specifies the configuration of
    its parent module. The property names are the dictionary keys and the
    properties are the dict values. If a key is a Mdoule class, then the
    corresponding value is the configuration of an instance of that module.

    Structure:
    config = {
    'property_name': 'property_value',
    'ptutils.Module.ClassName': sub_module_config,
    }

    # CONFIG.stat_dict() return its parent module's
    state_dict() without data (i.e. parameters)
    """

    def __init__(self, *args, **kwargs):
        super(Module, self).__init__()

    def configure(self):
        pass

    def state_dict(self, destination=None, prefix=''):
        if destination is None:
            destination = OrderedDict()
        for name, prop in self.named_properties.items():
            if issubclass(name, Module):
                pass
        pass

    def load_state_dict():
        pass


class Status(Module):
    """Verify a Module's status, compatibility and progress.

    """

    def __init__(self, *args, **kwargs):
        super(Module, self).__init__()

    def verify(self):
        pass


class Property(object):
    """A kind of Property that is to be considered a module property.

    Properties are arbitray python objects that exibit special behavior when
    used with :class:`Module`s - when they're assigned as Module attributes
    they are automatically added to the list of its properties, and will
    appear e.g. in :meth:`~Module.properties` iterator.

    Users should subclass the :class:`Property` class in anyway they please,
    and assign instances to a `Module` to define its state. These properties
    will populate a `Module`'s state_dict, which can then be saved automatically
    for the user.

    group related properties together e.g. status props, config props etc.
    each group can be assigned meta_properties about saving and logging
    requires_save, save_freq, log_to_tensorboard, to std out etc.
    """
    def __init__(self,
                 data,
                 name=None,
                 requires_save=DEFAULT_REQUIRES_SAVE,
                 save_freq=DEFAULT_SAVE_FREQ):
        self.data = data
        self.name = name
        self.requires_save = requires_save
        self.save_freq = save_freq

        if self.name is not None:
            self.__name__ = self.name

    def __repr__(self):
        return('{} with name {} containing:\n'.format(self.__class__.__name__,
                                                      self.name) +
               '{}\n'.format(self.data.__repr__()) +
               'Requires save: {}\n'.format(self.requires_save) +
               'Save Frequency: {}\n'.format(self.save_freq))


class Parameter(Property, Variable):
    """A kind of Property that is to be considered a module parameter.

    Parameters are :class:`~torch.autograd.Variable` subclasses, that have a
    very special property when used with :class:`Module` s - when they're
    assigned as Module attributes they are automatically added to the list of
    its parameters, and will appear e.g. in :meth:`~Module.parameters` iterator.
    Assigning a Variable doesn't have such effect. This is because one might
    want to cache some temporary state, like last hidden state of the RNN, in
    the model. If there was no such class as :class:`Parameter`, these
    temporaries would get registered too.

    Another difference is that parameters can't be volatile and that they
    require gradient by default.

    Arguments:
        data (Tensor): parameter tensor.
        requires_grad (bool, optional): if the parameter requires gradient. See
            :ref:`excluding-subgraphs` for more details.
    """
    def __new__(cls, data=None, requires_grad=True):
        return super(Parameter, cls).__new__(cls, data, requires_grad=requires_grad)

    def __repr__(self):
        return 'Parameter containing:' + self.data.__repr__()


class ModuleList(Module):
    """Holds submodules in a list.

    ModuleList can be indexed like a regular Python list, but modules it contains
    are properly registered, and will be visible by all Module methods.

    Arguments:
        modules (list, optional): a list of modules to add

    Example::

        class MyModule(ptutils.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.modules = ptutils.ModuleList([Module1, Module2,...]

            def display_modules(self):
                # ModuleList can act as an iterable, or be indexed using ints
                for i, m in enumerate(self.modules):
                    print({}th model is {}.format(i, m))
    """

    def __init__(self, modules=None):
        super(ModuleList, self).__init__()
        if modules is not None:
            self += modules

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        return self._modules[str(idx)]

    def __setitem__(self, idx, module):
        return setattr(self, str(idx), module)

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def __iadd__(self, modules):
        return self.extend(modules)

    def append(self, module):
        """Appends a given module at the end of the list.

        Arguments:
            module (nn.Module): module to append
        """
        self.add_module(str(len(self)), module)
        return self

    def extend(self, modules):
        """Appends modules from a Python list at the end.

        Arguments:
            modules (list): list of modules to append
        """
        if not isinstance(modules, list):
            raise TypeError("ModuleList.extend should be called with a "
                            "list, but got " + type(modules).__name__)
        offset = len(self)
        for i, module in enumerate(modules):
            self.add_module(str(offset + i), module)
        return self


class ModuleDict(Module):
    """Holds submodules in a dict.

    ModuleList can be indexed like a regular Python list, but modules it contains
    are properly registered, and will be visible by all Module methods.

    Arguments:
        modules (list, optional): a list of modules to add

    Example::

        class MyModule(ptutils.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.modules = ptutils.ModuleList([Module1, Module2,...]

            def display_modules(self):
                # ModuleList can act as an iterable, or be indexed using ints
                for i, m in enumerate(self.modules):
                    print({}th model is {}.format(i, m))
    """

    def __init__(self, modules=None):
        super(ModuleDict, self).__init__()
        if modules is not None:
            self += modules

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def __iadd__(self, modules):
        return self.extend(modules)

    def append(self, name, module):
        """Inserts a module into the dictionary with key `name`.

        Arguments:
            name (stf): module name and dict key
            module (nn.Module): module to append
        """
        self.add_module(str(len(self)), module)
        return self

    def extend(self, modules):
        """Appends modules from a Python list at the end.

        Arguments:
            modules (list): list of modules to append
        """
        if not isinstance(modules, list):
            raise TypeError("ModuleList.extend should be called with a "
                            "list, but got " + type(modules).__name__)
        offset = len(self)
        for i, module in enumerate(modules):
            self.add_module(str(offset + i), module)
        return self


class SubProperty(Property):

    def __init__(self):
        super(SubProperty, self).__init__()


class PropertyDict(Map):
    """Holds Properties in an enhanced dict that supports dot notation.

    Passing a python dict to the `PropertyGroup`'s init method automatically
    converts the values to properties named by the corresponding keys.

    Format:
        property = PropertyGroup['property_name']

        or equivalently,

        property = PropertyGroup.property_name

    Args:
        *args (dict): an arbitrary number of dictionaries whose keys
            are Property names and values are args to the Property
            init method.
        *kwargs: an arbitrary number of kwargs where the keywords
            are property names and the values are args to the
            Property init method.
    """

    def __init__(self, *args, **kwargs):
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = Property(v, name=k)

        if kwargs:
            for k, v in kwargs.items():
                self[k] = Property(v, name=k)

class PropertyList(Module):
    """Holds Properties in a list.

    PropertyLists can be indexed like a regular Python list, but poperties it contains
    are properly registered, and will be visible by all Module methods.

    Arguments:
        properties (list, optional): a list of :class:`ptutils.Property`` to add

    Example::

        class MyModule(ptutils.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.config = ptutils.PropertyList([ptutils.Property()])

            def forward(self, x):
                # PropertyLists can act as an iterable, or be indexed using ints
                for i, p in enumerate(self.config):
                    x = self.config[i // 2].mm(x) + p.mm(x)
                return x
    """

    def __init__(self, properties=None):
        super(PropertyList, self).__init__()
        if properties is not None:
            self += properties

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        return self._properties[str(idx)]

    def __setitem__(self, idx, prop):
        return self.register_property(str(idx), prop)

    def __len__(self):
        return len(self._properties)

    def __iter__(self):
        return iter(self._properties.values())

    def __iadd__(self, properties):
        return self.extend(properties)

    def append(self, property):
        """Appends a given property at the end of the list.

        Arguments:
            property (ptutils.Property): parameter to append
        """
        self.register_property(str(len(self)), property)
        return self

    def extend(self, properties):
        """Appends properties from a Python list at the end.

        Arguments:
            properties (list): list of properties to append
        """
        if not isinstance(properties, list):
            raise TypeError("Configuration.extend should be called with a "
                            "list, but got " + type(properties).__name__)
        offset = len(self)
        for i, prop in enumerate(properties):
            self.register_property(str(offset + i), prop)
        return self


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

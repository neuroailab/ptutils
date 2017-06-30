import copy
import collections


class Map(dict):
    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            dict.__setitem__(self, name, Map(value))
        else:
            dict.__setitem__(self, name, value)

    __delattr__ = dict.__delitem__

    def __dir__(self):
        return self.keys() + dir(dict(self))

    def __deepcopy__(self, memo):
        return Map(copy.deepcopy(dict(self)))


class NewMap(object):
    def __init__(self, *args, **kwargs):
        for arg in args:
            if isinstance(arg, dict):
                for key, value in arg.items():
                    setattr(self, key, value)

        if kwargs:
            for key, value in kwargs.items():
                setattr(self, key, value)
            # self.__dict__.update(kwargs)
            # self.set(k) = v

    # def __getattr__(self, attr):
    #     return self.get(attr)

    # # def __setattr__(self, name, value):
    # #     if isinstance(value, dict):
    # #         dict.__setitem__(self, name, Map(value))
    # #     else:
    # #         dict.__setitem__(self, name, value)
    # __delattr__ = dict.__delitem__

    # def __dir__(self):
    #     return self.keys() + dir(dict(self))

    # def __deepcopy__(self, memo):
    #     return Map(copy.deepcopy(dict(self)))


class frozendict(collections.Mapping):
    """
    An immutable wrapper around dictionaries that implements the complete :py:class:`collections.Mapping`
    interface. It can be used as a drop-in replacement for dictionaries where immutability is desired.

    from https://pypi.python.org/pypi/frozendict

    """

    dict_cls = dict

    def __init__(self, *args, **kwargs):
        self._dict = self.dict_cls(*args, **kwargs)
        self._hash = None

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def copy(self, **add_or_replace):
        return self.__class__(self, **add_or_replace)

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def __repr__(self):
        return '<%s %r>' % (self.__class__.__name__, self._dict)

    def __hash__(self):
        if self._hash is None:
            h = 0
            for key, value in self._dict.items():
                h ^= hash((key, value))
            self._hash = h
        return self._hash

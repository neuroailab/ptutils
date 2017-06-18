import copy


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
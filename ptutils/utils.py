import re
import os
import copy
import json
import inspect
import logging
import datetime
import collections
import pkg_resources

import git
import numpy as np
from bson.objectid import ObjectId

from tensorflow.python import DType

logging.basicConfig()
log = logging.getLogger('ptutils')


def version_info(module):
    """Gets version of a standard python module
    """
    if hasattr(module, '__version__'):
        version = module.__version__
    elif hasattr(module, 'VERSION'):
        version = module.VERSION
    else:
        pkgname = module.__name__.split('.')[0]
        try:
            info = pkg_resources.get_distribution(pkgname)
        except (pkg_resources.DistributionNotFound, pkg_resources.RequirementParseError):
            version = None
            log.warning('version information not found for %s -- what package is this from?' % module.__name__)
        else:
            version = info.version

    return {'version': version}


def version_check_and_info(module):
    """returns either git information or standard module version if not a git repo

    Args: - module (module): python module object to get info for.
    Returns: dictionary of info
    """
    srcpath = inspect.getsourcefile(module)
    try:
        repo = git.Repo(srcpath, search_parent_directories=True)
    except git.InvalidGitRepositoryError:
        log.info('module %s not in a git repo, checking package version' % module.__name__)
        info = version_info(module)
    else:
        info = git_info(repo)
    info['source_path'] = srcpath
    return info


def git_info(repo):
    """information about a git repo
    """
    if repo.is_dirty():
        log.warning('repo %s is dirty -- having committment issues?' % repo.git_dir)
        clean = False
    else:
        clean = True
    branchname = repo.active_branch.name
    commit = repo.active_branch.commit.hexsha
    origin = repo.remote('origin')
    urls = map(str, list(origin.urls))
    remote_ref = [_r for _r in origin.refs if _r.name == 'origin/' + branchname]
    if not len(remote_ref) > 0:
        log.warning('Active branch %s not in origin ref' % branchname)
        active_branch_in_origin = False
        commit_in_log = False
    else:
        active_branch_in_origin = True
        remote_ref = remote_ref[0]
        gitlog = remote_ref.log()
        shas = [_r.oldhexsha for _r in gitlog] + [_r.newhexsha for _r in gitlog]
        if commit not in shas:
            log.warning('Commit %s not in remote origin log for branch %s' % (commit,
                                                                              branchname))
            commit_in_log = False
        else:
            commit_in_log = True
    info = {'git_dir': repo.git_dir,
            'active_branch': branchname,
            'commit': commit,
            'remote_urls': urls,
            'clean': clean,
            'active_branch_in_origin': active_branch_in_origin,
            'commit_in_log': commit_in_log}
    return info


def make_mongo_safe(_d):
    """Makes a json-izable actually safe for insertion into Mongo."""
    klist = _d.keys()[:]
    for _k in klist:
        if hasattr(_d[_k], 'keys'):
            make_mongo_safe(_d[_k])
        if not isinstance(_k, str):
            _d[str(_k)] = _d.pop(_k)
        _k = str(_k)
        if '.' in _k:
            _d[_k.replace('.', '___')] = _d.pop(_k)


def sonify(arg, memo=None):
    """when possible, returns version of argument that can be
       serialized trivally to json format.
    """
    if memo is None:
        memo = {}
    if id(arg) in memo:
        rval = memo[id(arg)]

    if isinstance(arg, ObjectId):
        rval = arg
    elif isinstance(arg, datetime.datetime):
        rval = arg
    elif isinstance(arg, DType):
        rval = arg
    elif isinstance(arg, np.floating):
        rval = float(arg)
    elif isinstance(arg, np.integer):
        rval = int(arg)
    elif isinstance(arg, (list, tuple)):
        rval = type(arg)([sonify(ai, memo) for ai in arg])
    elif isinstance(arg, collections.OrderedDict):
        rval = collections.OrderedDict([(sonify(k, memo), sonify(v, memo))
                                        for k, v in arg.items()])
    elif isinstance(arg, dict):
        rval = dict([(sonify(k, memo), sonify(v, memo))
                     for k, v in arg.items()])
    elif isinstance(arg, (basestring, float, int, type(None))):
        rval = arg
    elif isinstance(arg, np.ndarray):
        if arg.ndim == 0:
            rval = sonify(arg.sum())
        else:
            rval = map(sonify, arg)  # N.B. memo None
    # -- put this after ndarray because ndarray not hashable
    elif arg in (True, False):
        rval = int(arg)
    elif callable(arg):
        mod = inspect.getmodule(arg)
        modname = mod.__name__
        objname = arg.__name__
        rval = version_check_and_info(mod)
        rval.update({'objname': objname,
                     'modname': modname})
        rval = sonify(rval)
    else:
        raise TypeError('sonify', arg)

    memo[id(rval)] = rval
    return rval


def jsonize(x):
    """returns version of x that can be serialized trivally to json format
    """
    try:
        json.dumps(x)
    except TypeError:
        return sonify(x)
    else:
        return x


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

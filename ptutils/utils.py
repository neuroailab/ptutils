import os
import re
import sys
import copy
import json
import yaml
import pkgutil
import inspect
import logging
import datetime
import pkg_resources
import cPickle as pickle

import git
import numpy as np
from bson.objectid import ObjectId

logging.basicConfig()
log = logging.getLogger('ptutils')

CONFIG_TYPES = {'yml': {'file': yaml.load, 'data': yaml.load},
                'yaml': {'file': yaml.load, 'data': yaml.load},
                'json': {'file': json.load, 'data': json.loads},
                'pkl': {'file': pickle.load, 'data': pickle.loads}}


def parse_config(config):
    """Parse input arguments to configuration modules."""
    if isinstance(config, dict):
        return config
    elif isinstance(config, str):
        # Load configuration file
        pattern = r'.(' + '|'.join(CONFIG_TYPES.keys()) + ')$'
        m = re.search(pattern, config, flags=re.I)
        if m is not None:
            type_ = m.group().lower()[1:]
            if os.path.isfile(config):
                return load_file(config, type_)
        else:
            for t in CONFIG_TYPES.keys():
                out = load_data(config, t)
                if out is not None:
                    return out
            raise ValueError('Invalid configuration format format: {}'.format(config))


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--params', type=json.loads, default=None)
    parser.add_argument('-g', '--gpu', default='0', type=str)
    args = vars(parser.parse_args())
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    for p in filter(lambda x: x.endswith('_func'), args):
        modname, objname = args[p].rsplit('.', 1)
        mod = importlib.import_module(modname)
        args[p] = getattr(mod, objname)
    return args


def load_file(config_file, type_):
    with open(config_file) as data:
        return CONFIG_TYPES[type_]['file'](data)


def load_data(data, type_):
    try:
        return CONFIG_TYPES[type_]['data'](data)
    except Exception:
        return None


def version_info(module):
    """Return version of a standard python module."""
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
    """Return either git info or standard module version if not a git repo.

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
    """Return information about a git repo."""
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
    """Make a json-izable actually safe for insertion into Mongo."""
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
    """Return version of arg that can be trivally serialized to json format."""
    if memo is None:
        memo = {}
    if id(arg) in memo:
        rval = memo[id(arg)]

    if isinstance(arg, ObjectId):
        rval = arg
    elif isinstance(arg, datetime.datetime):
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
    """Return version of x that can be serialized trivally to json format."""
    try:
        json.dumps(x)
    except TypeError:
        return sonify(x)
    else:
        return x


def load_modules_from_path(path):
    """Import all modules from the given directory."""
    # Check and fix the path
    if path[-1:] != '/':
        path += '/'

    # Get a list of files in the directory, if the directory exists
    if not os.path.exists(path):
        raise OSError('Directory does not exist: {}'.format(path))

    # Add path to the system path
    sys.path.append(path)
    # Load all the files in path
    for f in os.listdir(path):
        # Ignore anything that isn't a .py file
        if len(f) > 3 and f[-3:] == '.py':
            modname = f[:-3]
            # Import the module
            __import__(modname, globals(), locals(), ['*'])


def load_class_from_name(fqcn):
    """Break apart fully qualified name (fqcn) to get module and classname."""
    paths = fqcn.split('.')
    modulename = '.'.join(paths[:-1])
    classname = paths[-1]
    # Import the module
    __import__(modulename, globals(), locals(), ['*'])
    # Get the class
    cls = getattr(sys.modules[modulename], classname)
    # Check cls
    if not inspect.isclass(cls):
        raise TypeError('{} is not a class'.format(fqcn))
    # Return class
    return cls


def import_string(import_name, silent=False):
    """Imports an object based on a string.  This is useful if you want to
    use import paths as endpoints or something similar.  An import path can
    be specified either in dotted notation (``xml.sax.saxutils.escape``)
    or with a colon as object delimiter (``xml.sax.saxutils:escape``).
    If `silent` is True the return value will be `None` if the import fails.
    :param import_name: the dotted name for the object to import.
    :param silent: if set to `True` import errors are ignored and
                   `None` is returned instead.
    :return: imported object
    """
    # force the import name to automatically convert to strings
    # __import__ is not able to handle unicode strings in the fromlist
    # if the module is a package
    import_name = str(import_name).replace(':', '.')
    try:
        try:
            __import__(import_name)
        except ImportError:
            if '.' not in import_name:
                raise
        else:
            return sys.modules[import_name]

        module_name, obj_name = import_name.rsplit('.', 1)
        try:
            module = __import__(module_name, None, None, [obj_name])
        except ImportError:
            # support importing modules not yet set up by the parent module
            # (or package for that matter)
            module = import_string(module_name)

        try:
            return getattr(module, obj_name)
        except AttributeError as e:
            raise ImportError(e)

    except ImportError as e:
        if not silent:
            raise(
                ImportStringError,
                ImportStringError(import_name, e),
                sys.exc_info()[2])


def find_modules(import_path, include_packages=False, recursive=False):
    """Finds all the modules below a package.

    This can be useful to automatically import all views / controllers so
    that their metaclasses / function decorators have a chance to register
    themselves on the application.

    Packages are not returned unless `include_packages` is `True`.  This can
    also recursively list modules but in that case it will import all the
    packages to get the correct load path of that module.

    Args:
        import_path: the dotted name for the package to find child modules.
        include_packages: set to `True` if packages should be returned, too.
        recursive: set to `True` if recursion should happen.
    Returns:
        generator
    """
    module = import_string(import_path)
    path = getattr(module, '__path__', None)
    if path is None:
        raise ValueError('%r is not a package' % import_path)
    basename = module.__name__ + '.'
    for importer, modname, ispkg in pkgutil.iter_modules(path):
        modname = basename + modname
        if ispkg:
            if include_packages:
                yield modname
            if recursive:
                for item in find_modules(modname, include_packages, True):
                    yield item
        else:
            yield modname


class ImportStringError(ImportError):

    """Provides information about a failed :func:`import_string` attempt."""

    #: String in dotted notation that failed to be imported.
    import_name = None
    #: Wrapped exception.
    exception = None

    def __init__(self, import_name, exception):
        self.import_name = import_name
        self.exception = exception

        msg = (
            'import_string() failed for %r. Possible reasons are:\n\n'
            '- missing __init__.py in a package;\n'
            '- package or module path not included in sys.path;\n'
            '- duplicated package or module name taking precedence in '
            'sys.path;\n'
            '- missing module, class, function or variable;\n\n'
            'Debugged import:\n\n%s\n\n'
            'Original exception:\n\n%s: %s')

        name = ''
        tracked = []
        for part in import_name.replace(':', '.').split('.'):
            name += (name and '.') + part
            imported = import_string(name, silent=True)
            if imported:
                tracked.append((name, getattr(imported, '__file__', None)))
            else:
                track = ['- %r found in %r.' % (n, i) for n, i in tracked]
                track.append('- %r not found.' % name)
                msg = msg % (import_name, '\n'.join(track),
                             exception.__class__.__name__, str(exception))
                break

        ImportError.__init__(self, msg)

    def __repr__(self):
        return '<%s(%r, %r)>' % (self.__class__.__name__, self.import_name,
                                 self.exception)
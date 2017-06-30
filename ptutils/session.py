import copy
import time
import logging
import warnings
from collections import defaultdict

import torch
from torch.autograd import Variable

from .base import *
from .data import DataProvider
from .database import DBInterface
from .model import Model, Criterion, Optimizer


logging.basicConfig()
log = logging.getLogger('ptutils')
log.setLevel('DEBUG')
logging.captureWarnings(True)


class Session(Module):
    """Coordinates an NN experiment.

    Base class for all neural network experiments that specifically serves to
    leverage and extend PyTorch's dynamic nature. The `Session` class
    coordinates interactions between the ptutil objects it contains, such as
    a Model`, `DBInterface` and `DataProvider`, throughout an experiment.

    A Session can also contain other Session objects as regular
    attributes, allowing users to nest them in a tree structure.
    This property gives users a method for managing a large number
    of related experiments simultaneously.

    The 'define-by-run' paradigm established by Chainer, DyNet, PyTorch offers a
    powerful new way to structure neural network computations: the execution of
    the model/graph is conditioned on the state of the model, forming a dynamic
    graph. The Session attempts to accommodate this flexible nature by giving
    researchers a dynamic session whereby execution of the session is
    conditioned on the state of the session and any quantity contained within.
"""

    __name__ = 'sess'
    _CORE_MODULES = {
        Model.__name__: Model,
        DBInterface.__name__: DBInterface,
        DataProvider.__name__: DataProvider,
    }

    def __init__(self, *args, **kwargs):
        super(Session, self).__init__(*args, **kwargs)

        self._DEFAULT = defaultdict(list)
        self._MISSING = copy.deepcopy(self._CORE_MODULES)

        for arg in args:
            if isinstance(arg, Configuration):
                log.info('Configuration detected ...')
                log.info('Configuration verified ...')
                log.info('Configuring {}'.format(self.__class__.__name__))
                for key, value in self.Configuration.configure().items():
                    self[key] = value

    @property
    def status(self):
        self._status = self._determine_status()
        return self._status

    @status.setter
    def status(self, status):
        current_status = self._determine_status()
        if status != current_status:
            warnings.warn('Status mismatch')
        self._status = current_status

    @property
    def config(self):
        print('Getting config')
        # self._config = self._determine_config()
        return self._config

    @config.setter
    def config(self, config):
        current_config = self._determine_config()
        if config != current_status:
            print('Config mismatch')
        self._config = config

    def __setattr__(self, name, value):
        Module.__setattr__(self, name, value)

        if isinstance(value, Configuration):
            print('Configuration is being assigned!')
            Module.__setattr__(self, 'config', value)
            for modname, mod in value.configure().items():
                Module.__setattr__(self, modname, mod)

    def _determine_config(self):
        pass

    def _determine_status(self):
        log.info('Determining session status ...')
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
            'num_run': 'number of attempted sess.runs',
            'run_id': {
                'init': '[new/resume/restart]',
                        'start_date': 'start date',
                        'state': '[pending/in-progress/complete]',
                        'progress': '% complete',
                        'eta': 'estimated time to completion',
                        'errors': '[error1, error2, error3, ...]',
                        'end_date': 'end date',
                        'outcome': 'outcome description'}}
        return status

    def _find_stored_sess(self):
        pass
        # self.status = {'inventory': inventory,
        #                'essentials': essentials}

        # valid = has_model and has_db_interface and has_data_provider
        # essentials = inventory.intersection({DBInterface, DataProvider, Model})
        """
        TODO: determine the standard procedure for creating a session:
        (1) Iterate over all modules an generate an module inventory by type!
        (2) If at least one DBInterface is present, generate state_dict() and
            and query db for previous/related/similar session.
        (3) If a previous session is found , parse status property.
        (2) Confirm status with checks below
            - check to see if children are also session or else
            - check to see that one child module has/is a dbinterface
                - if no DBInterface, warn that default session execution procedure cannot proceed
                    - computations cannot be saved
                    - previous session cannot be resumed or restarted
            - check to see that one child module has/is a dataprovider
                - if no data_provider, warn that default run procedure cannot proceed
                    - no data will be provided to the session automatically
            - check to see that one child module has/is a model
                - if no model, warn that default run procedure cannot proceed
                    - no model will be executed
                    -
            - check for a mix of sessions and non-sessions
            - check to see if there are module conflicts
        (2) Check to see if run method has been overridden
        (3) Use config and db to get status if not specified, else verify.
        (4) Use config and status to load correct model and data
            if not specified, else verify compatability with status.
        (5) *** REGISTER ALL SUBSESSIONS AND/OR ptutils.Module

        -----------------------------------------------------------------------
        OLD:
        TODO: determine the standard procedure for creating a session:
        Option 1: parse a config dict that either contains the session
                  objects as elements or specifies params neccessary
                  for the session to create them.
        Option 2: Users subclass a BaseConfig class whose methods return
                  all the neccessary session objects.
        Option 3: Users manual assign session objects as regular attributes.
        """

        """
        TODO:
        (1) Parse config via property assignment.
        (2) Use config to load db, if not specified.
        (3) Use config and db to get status if not specified, else verify.
        (4) Use config and status to load correct model and data
            if not specified, else verify compatability with status.
        (5) *** REGISTER ALL SUBSESSIONS AND/OR ptutils.Module

        Maybe everything shoud subclass ptutils.unit/component/element/module
        """
        # self.config = config
        # self.model = config['model']
        # self.criterion = config['criterion']
        # self.optimizer = config['optimizer']
        # self.data_provider = config['data_provider']
        # self.db = config['db_interface']

        # TODO: Error checking.
        # if db is None:
        #     self.db = self._load_db()

        # TODO: Determine Session.run() behavior.
        # Option 1:
        #   (1): Check to see if session.run() has been overidden.
        #       If it has:
        #           - run user registered pre_run_hook functions.
        #           - run user's run method.
        #           - run user registered post_run_hook functions.
        #   (2):
        #   (2): Load model parameters onto GPUs
        #   (3):
    # @property
    # def config(self):
    #     return self._config

    # @config.setter
    # def config(self, config):
    #     # TODO: Parse config to ensure validity
    #     self._config = config

    def default_run(self):
        """Run an session specified by its configuration and status."""

        # TODO: Use status to start/resume session if needed
        # TODO: If run() is not overridden, run default from config.
        # TODO: Log and save progress

        self.model = self._DEFAULT['Model']
        self.db_interface = self._DEFAULT['DBInterface']
        self.data_provider = self._DEFAULT['DataProvider']
        # self.dataset = 'CIFAR10'
        # self.model = torch.nn.DataParallel(self.model).cuda()
        # for epoch in range(self.config['run']['num_epochs']):
        for epoch in range(10):

            # train for one epoch
            self.epoch = epoch
            self._run(mode='train')

            # evaluate on validation set
            prec1 = self._run(mode='test')

            self.db_interface.save({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'prec1': prec1})
            # 'optimizer': self.optimizer.state_dict()})
            # if prec1 > 60:
                # print('-' * 80)
                # self.model.re_init_fc(num_classes=100)
                # self.dataset = 'CIFAR100'

    def _run(self, mode='train'):

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # Move model and criterion to GPU
        # self.model = torch.nn.DataParallel(self.model).cuda()

        use_cuda = torch.cuda.is_available()
        dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

        if use_cuda:
            self.model.cuda()
            self.criterion.cuda()

        # Select mode
        if mode == 'train':
            self.model.train()
            volatile = False
        else:
            self.model.eval()
            volatile = True

        # data_loader = self.data_provider.get_data_loader(dataset=self.dataset, mode=mode)
        data_loader = self.data_provider.provide(mode=mode)

        start = time.time()
        for step, (input, target) in enumerate(data_loader):

            # Measure data loading time
            data_time.update(time.time() - start)

            input_var = Variable(input, volatile=volatile)
            target_var = Variable(target, volatile=volatile)

            # target = target.cuda(async=True)
            # input_var = Variable(input, volatile=volatile).cuda()
            # target_var = Variable(target, volatile=volatile).cuda()

            # target = target.type(dtype, async=True)
            # input_var = Variable(input.type(dtype), volatile=volatile)
            # target_var = Variable(target.type(torch.LongTensor), volatile=volatile)

            # Compute output and loss
            output = self.model(input_var)
            loss = self.criterion(output, target_var)

            # Measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data[0], input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            if mode == 'train':
                # Compute gradient and do SGD step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Measure elapsed time
            batch_time.update(time.time() - start)
            start = time.time()

            if step % 10 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
                      .format(self.epoch + 1, step, len(data_loader),
                              batch_time=batch_time,
                              data_time=data_time,
                              loss=losses,
                              top1=top1,
                              top5=top5))
        return top1.avg if mode == 'test' else None

    def register_pre_run_hook(self, hook):
        """Registers a backward hook on the module.

        The hook will be called every time the gradients with respect to module
        inputs are computed. The hook should have the following signature::

            hook(module, grad_input, grad_output) -> Tensor or None

        The :attr:`grad_input` and :attr:`grad_output` may be tuples if the
        module has multiple inputs or outputs. The hook should not modify its
        arguments, but it can optionally return a new gradient with respect to
        input that will be used in place of :attr:`grad_input` in subsequent
        computations.

        This function returns a handle with a method ``handle.remove()``
        that removes the hook from the module.
        """
        handle = hooks.RemovableHandle(self._backward_hooks)
        self._backward_hooks[handle.id] = hook
        return handle

    def register_post_run_hook(self, hook):
        """Registers a forward hook on the module.

        The hook will be called every time :func:`forward` computes an output.
        It should have the following signature::

            hook(module, input, output) -> None

        The hook should not modify the input or output.
        This function returns a handle with a method ``handle.remove()``
        that removes the hook from the module.
        """
        handle = hooks.RemovableHandle(self._forward_hooks)
        self._forward_hooks[handle.id] = hook
        return handle

    # def __call__(self, *input, **kwargs):
    #     result = self.forward(*input, **kwargs)
    #     for hook in self._forward_hooks.values():
    #         hook_result = hook(self, input, result)
    #         if hook_result is not None:
    #             raise RuntimeError(
    #                 "forward hooks should never return any values, but '{}'"
    #                 "didn't return None".format(hook))
    #     if len(self._backward_hooks) > 0:
    #         var = result
    #         while not isinstance(var, Variable):
    #             var = var[0]
    #         grad_fn = var.grad_fn
    #         if grad_fn is not None:
    #             for hook in self._backward_hooks.values():
    #                 wrapper = functools.partial(hook, self)
    #                 functools.update_wrapper(wrapper, hook)
    #                 grad_fn.register_hook(wrapper)
    #     return result


def train_epoch(train_loader, model, criterion, optimizer, epoch, db):
    _run_epoch(train_loader, model, criterion, optimizer, epoch, db, mode='train')


def validate_epoch(val_loader, model, criterion, optimizer, epoch, db):
    return _run_epoch(val_loader, model, criterion, optimizer, epoch, mode='validate')


def _run_epoch(data_loader, model, criterion, optimizer, epoch, db, mode=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # Move model and criterion to GPU
    # model = torch.nn.DataParallel(model).cuda()
    model.cuda()
    criterion.cuda()

    # Select mode
    if mode == 'train':
        model.train()
        volatile = False
    else:
        model.eval()
        volatile = True

    start = time.time()
    for i, (input, target) in enumerate(data_loader):

        # Measure data loading time
        data_time.update(time.time() - start)

        target = target.cuda(async=True)
        input_var = Variable(input, volatile=volatile).cuda()
        target_var = Variable(target, volatile=volatile).cuda()

        # Compute output and loss
        output = model(input_var)
        loss = criterion(output, target_var)

        # Measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        if mode == 'train':
            # Compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - start)
        start = time.time()

        if i % 10 == 0:
            save_checkpoint({'epoch': epoch + 1,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict()},
                            db)

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
                  .format(epoch, i, len(data_loader),
                          batch_time=batch_time,
                          data_time=data_time,
                          loss=losses,
                          top1=top1,
                          top5=top5))
    return top1.avg


def accuracy(output, target, topk=(1,)):
    """Compute the precision@k for the specified values of k"""

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, database):
    database.save(state)


class Monitor(object):
    """Interface for monitoring an arbitrary object during a session."""

    def __init__(self):
        self.state = None
        self.reset()

    def reset(self):
        raise NotImplementedError()

    def update(self):
        raise NotImplementedError()

    def view(self):
        raise NotImplementedError()


class AverageMeter(Monitor):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.state = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += n * val
        self.count += n
        self.avg = self.sum / self.count
        self.state = self.avg

    def view(self):
        return self.state


# JUNK: TO DELETE
# inventory = set(self.module_names())
# cls_inventory = set(map(type, self.modules()))
# class Config(object):
#     def __init__(self, config_dict):
#         self.session = None
#         self.model = config['model']
#         self.criterion = config['criterion']
#         self.optimizer = config['optimizer']
#         self.data_provider = config['data_provider']
#         self.db = config['db_interface']

#     def get_model(self):
#         pass

#     def get_data_provider(self):
#         pass

#     def get_db_interface(self):
#         pass


"""

Potential config structure:

config.session = {'session_id': session id number,
                  'description'(optional): description of session,
                  'status': {'num_run': number of attempted sess.runs,
                             'run_id': {'init': [new/resume/restart],
                                        'start_date': start date,
                                        'state': '[pending/in-progress/complete]',
                                        'progress': % complete,
                                        'eta': estimated time to completion,
                                        'errors': [error1, error2, error3, ...]
                                        'end_date': end data,
                                        'outcome': 'outcome description'}}
                  'history' (optional): {instance_prop1: val, intance_prop2: val, ...}
                  'subsessions: [subsession1, subsession2, subsession3, ...]}
config.model = {'model': instance of model class,
                'name' (optional): model name+description}
config.criterion = {'criterion': }
config.optimizer = {'model': instance of model class,
                'name' (optional): model name+description}
"""

    # def _get_status(self):
    #     pass

    # def _load_model(self):
    #     """Return a valid torch Module in the desired state. """
    #     pass

    # def _load_data_provider(self):
    #     """Return a DataProvider with the desired dataset and dataloader."""
    #     pass

    # def _load_db(self):
    #     """Return an implemented DBInterface."""
    #     pass
# has_essentials = (len(_ESSENTIAL_MODULES) == len(DEFAULTS))
        # missing_mods = (
        #     set(DEFAULTS.keys()).difference(map(type, self._CORE_MODULES)))

        # core = defaultdict(list)
        # # Partition core session components by class.
        # for name, module in self.named_modules():
        #     if isinstance(module, Model):
        #         core['model'].append(name)
        #     if isinstance(module, Session):
        #         core['session'].append(name)
        #     if isinstance(module, Criterion):
        #         core['criterion'].append(name)
        #     if isinstance(module, Optimizer):
        #         core['optimizer'].append(name)
        #     if isinstance(module, Configuration):
        #         core['config'].append(name)
        #     if isinstance(module, DBInterface):
        #         core['db_interface'].append(name)
        #     if isinstance(module, DataProvider):
        #         core['data_provider'].append(name)

        # # Set first module in list as default.
        # for mod_type, mods_list in core.items():
        #     if mods_list:
        #         if mod_type == 'session':
        #             if len(mods_list) > 1:
        #                 self['_DEFAULT_' + mod_type + '_name'] = mods_list[1]
        #         else:
        #             self['_DEFAULT_' + mod_type + '_name'] = mods_list[0]

        # if not core['config']:
            # self._generate_config()
            # pass
        # valid_config self._c

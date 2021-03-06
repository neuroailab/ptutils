"""ptutils Runner."""

import copy
import logging

from ptutils.base import Base
from .error import StepError, ExpIDError, LoadError

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel('DEBUG')


class Runner(Base):
    """This is the primary PTUtils class for running an experiment.

    The runner is the top level class of PTUtils. It organizes all of the
    other core PTUtils objects.

    Attributes:
        exp_id (str): Experiment ID for saving experiment to the database.
        model (:class:`Model`): Model (e.g. AlexNet, VGG16).
        global_step (int): The number of batches seen by the model during training.
        dbinterface (:class:`DBInterface`): Object to communicate with the database.
        dataprovider (:class:`DataProvider`): Provides training and validation data for the model.
        train_params (dict): Dictionary of parameters for training.
        save_params (dict): Dictionary of parameters for saving experiments.
        load_params (dict): Dictionary of parameters for loading past experiments.
    """

    def __init__(self,
                 exp_id,
                 model=None,
                 dbinterface=None,
                 dataprovider=None,
                 train_params=None,
                 global_step=None,
                 save_params=None,
                 load_params=None,
                 **kwargs):
        """Initialize the :class:`Runner` class.

        Args:
            exp_id (str): Experiment ID for saving experiment to the database.
            model (:class:`Model`): Model (e.g. AlexNet, VGG16).
            dbinterface (:class:`DBInterface`): Object to communicate with the database.
            dataprovider (:class:`DataProvider`): Provides training and validation data for the model.
            train_params (dict): Dictionary of parameters for training.
            save_params (dict): Dictionary of parameters for saving experiments.
            load_params (dict): Dictionary of parameters for loading past experiments.

        """
        super(Runner, self).__init__(**kwargs)

        # Core bases.
        self.model = model
        self.dbinterface = dbinterface
        self.dataprovider = dataprovider

        # Params.
        self.save_params = save_params
        self.load_params = load_params
        self.train_params = train_params

        self.exp_id = exp_id
        # global_step (int): The number of batches seen by the model during training.
        self.global_step = global_step

# -- Runner Properties ---------------------------------------------------------

    @classmethod
    def init(cls, params):
        """Initialize the :class:`Runner`.

        This is the user's primary means of initializing the :class:`Runner`. The function is
        distinct from the :class:`Runner`'s :func:__init__, which is used in constructing the runner
        in :class:`Base`'s :func:`from_params`s.

        A simple use case for loading a previous experiment::

            params = {

            'func': ptutils.runner.Runner,
            'name': 'MNISTRunner',
            'exp_id': 'example_experiment',
            'load_params': {
                'restore': False,
                'dbinterface': {
                    'func': ptutils.database.MongoInterface,
                    'name': 'mongo',
                    'port': 27017,
                    'host': 'localhost',
                    'database_name': 'ptutils_test',
                    'collection_name': 'ptutils_test'},
                'exp_id': 'mnist',
                'restore_params': None,
                'restore_mapping': None}}

            runner = ptutils.runner.Runner.init(params)
            runner.train()
        """
        # runner = Base.from_params(**params)
        runner = Base.from_params(params)
        if runner.load_params['restore']:
            loaded_run = runner.load_run()
            loaded_params = loaded_run['params']
            loaded_state = loaded_run['state']
            if loaded_params:
                loaded_params = Runner._replace_params(runner.to_params(), loaded_params)
                runner = Runner.from_params(loaded_params)
            runner.from_state(loaded_state,
                              restore_params=runner.load_params.get('restore_params'),
                              restore_mapping=runner.load_params.get('restore_mapping'))

        if runner.exp_id is None:
            error_msg = 'Cannot run an experiment without an exp_id'
            log.critical(error_msg)
            raise ExpIDError(error_msg)

        if (not runner.exp_id == runner.load_params['query']['exp_id']) and len(runner.dbinterface.load({'exp_id': runner.exp_id})) > 0:
            # If not resuming training, exp_id must be unique
            error_msg = 'Cannot run a new experiment with same exp_id as an existing record'
            log.critical(error_msg)
            raise ExpIDError(error_msg)

        runner.assign_devices()
        if runner.global_step is None:
            runner.global_step = 0

        return runner

#    @property
#    def global_step(self):
#        """Get the optimizer."""
#        return self._global_step
#
#    @global_step.setter
#    def global_step(self, value):
#        if value is None:
#            self._global_step = 0
#        if value > (self._global_step + 1):
#            raise StepError('The global step can only be incremented by one.')
#        elif value < 0:
#            raise StepError('The global step cannot be negative.')
#        else:
#            self._global_step = value
#
# -- Runner Methods ------------------------------------------------------------

    def setup_eval(self):
        """Set up the model for evaluation.

        If anything needs to be changed about the experiment for inference
        (e.g. the model's Dropout or Batch Norm), that should be taken care of here.
        """
        self.model.eval()

    def setup_train(self):
        """Set up the model for training.

        If anything needs to be changed about the experiment for inference
        (e.g. the model's Dropout or Batch Norm), that should be taken care of here.
        """
        self.model.train()

    def step(self, prev_output):
        """Define a single step of an experiment.

        A common use case will be to have the step correspond to a
        batch update (e.g. calling `self.model.step`).

        *This function must increment the `global_step`.*

        """
        data = self.dataprovider.provide(prev_output, mode='train')
        output = self.model.step(data)
        self.global_step += 1

        log.info('step: {}; loss: {}'.format(self.global_step,
                                             output['loss'].data[0]))
        return output

    def train(self):
        """Define the primary training loop.

        The default behavior is to step the trainer and
        save intermediate results.

        """
        model_output = None
        self.setup_train()
        for step in range(self.global_step, self.train_params['num_steps']):
            model_output = self.step(model_output)
            if self.global_step % self.save_params['metric_freq'] == 0:
                # Save desired results.
                record = {'exp_id': self.exp_id,
                          'step': self.global_step,
                          'loss': model_output['loss'].data[0],
                          'state': self.to_state(),
                          'params': self.to_params(),
                          }
                self.dbinterface.save(record)
                log.info("Saving step {}".format(self.global_step))

            if self.validation_params and self.global_step % self.save_params['val_freq'] == 0:
                # validation
                self.test()

        self.dbinterface.sync_with_host()

    def predict(self, prev_output=None):
        """Perform a single inference pass.

        Args:
            prev_output (Object, optional): Experiment ID for saving experiment to the database.

        Formally, this will call model.forward().
        """
        data = self.dataprovider.provide(prev_output, mode='test')[0]
        output = self.model.forward(data)
        return output

    def test(self):
        """Perform inference for several batches of data and save the result."""
        self.setup_eval()
        model_output = None
        all_model_outputs = []
        for step in range(self.validation_params['num_steps']):
            model_output = self.predict(model_output)
            all_model_outputs.append(model_output)

        # Save desired results.
        record = {'exp_id': self.exp_id,
                  'test_output': all_model_outputs,
                  'params': self.to_params(),
                  }
        self.dbinterface.save(record)
        self.dbinterface.sync_with_host()

    def load_run(self):
        """Load previous experiment from database.

        Uses the parameters in `self.load_params` to load a previous
        experiment. If multiple entries match the given query, the most
        recent entry in the database is returned.

        """
        # load_dbinterface = self.load_params['dbinterface']['func'](**self.load_params['dbinterface'])
        load_dbinterface = self.load_params['dbinterface']

        # filter for results that have the state saved
        # so that the state can be restored
        if 'state' not in self.load_params['query'].keys():  # make sure to not override 'state' query
            query = copy.copy(self.load_params['query'])
            query['state'] = {'$exists': True}


        all_results = load_dbinterface.load(query, from_load_run=True)

        try:
            # Load most recent run.
            return all_results[0]
        except IndexError:
            error_msg = 'No results in the database matched the load_query'
            log.critical(error_msg)
            raise LoadError(error_msg)

    @staticmethod
    def _replace_params(replacement, to_replace, parent_device=False):
        """Replace entries in :param:to_replace with :param:replacement key/val pairs.

        This function recurses through the :param:to_replace dictionary and replaces
        the key/val pairs with any keys specified in :param:replacement. If any values
        are dictionaries, only the keys specified within that dictionary are replaced.

        Additionally, if any devices are specified in `replacement`, all children
        of that dictionary will have their devices set to `None' such that
        `base.base_cuda` will override the child device with its parent.

        Args:
            replacement (dict): Dictionary with key/val pairs to use for replacement
            to_replace (dict): Dictionary whose key/val pairs will be replaced
            parent_device (boolean): Specifies whether the dictionary's parent
            had their 'device' replaced

        """
        for (key, value) in replacement.items():
            if isinstance(value, dict) and (key in to_replace.keys()):
                if not isinstance(to_replace[key], dict):
                    # this is necessary for the parameter remapping
                    # because if runner['load_params']['restore_params'] = None,
                    # or runner['load_params']['restore_mapping'] = None,
                    # this subsequent assignments won't be possible
                    to_replace[key] = {}
                if 'devices' in replacement.keys():
                    to_replace[key] = Runner._replace_params(value, to_replace[key], True)
                else:
                    to_replace[key] = Runner._replace_params(value, to_replace[key], parent_device or False)
            else:
                if value is not None:
                    to_replace[key] = value
                if parent_device and ('devices' in to_replace.keys()):
                    to_replace['devices'] = None
        return to_replace


class HyperParameterStatisticRunner(Runner):
    def __init__(self,
                 exp_id,
                 param_major=True,
                 param_dict_list=None,
                 num_iterations_per_param=1,
                 global_step=None,
                 dbinterface=None,
                 load_params=None,
                 **kwargs):
        self.param_dict_list = param_dict_list
        self.num_iterations_per_param = num_iterations_per_param
        # If outer loop of train is over params or statistics
        self.param_major = param_major
        # expand out the list of experiments to run
        self.total_param_dict_list = []
        if self.param_major is True:
            for d in self.param_dict_list:
                for itt in range(self.num_iterations_per_param):
                    # change the experiment id to take into account statistic run
                    d_copy = copy.deepcopy(d)
                    d_copy['exp_id'] = d['exp_id'] + '_statistic_run_' + str(itt)
                    d_copy['load_params']['query']['exp_id'] = d['exp_id'] + '_statistic_run_' + str(itt)
                    self.total_param_dict_list.append(d_copy)
        else:
            for itt in range(self.num_iterations_per_param):
                for d in self.param_dict_list:
                    d_copy = copy.deepcopy(d)
                    d_copy['exp_id'] = d['exp_id'] + '_statistic_run_' + str(itt)
                    d_copy['load_params']['query']['exp_id'] = d['exp_id'] + '_statistic_run_' + str(itt)
                    self.total_param_dict_list.append(d_copy)

        super(HyperParameterStatisticRunner, self).__init__(exp_id,
                                                            global_step=global_step,
                                                            dbinterface=dbinterface,
                                                            load_params=load_params,
                                                            **kwargs)


    def train(self):
        print(self.global_step)
        runner_list = self.total_param_dict_list[self.global_step:]
        for param_dict in runner_list:
            print(param_dict['exp_id'])
            print(param_dict['load_params']['query']['exp_id'])

            if len(self.dbinterface.load({'exp_id': param_dict['exp_id']})) > 0 and self.load_params['restore'] is True:
                print(param_dict['exp_id'])
                print(param_dict['load_params']['query']['exp_id'])
                param_dict['load_params']['restore'] = True

            runner = param_dict['func'].init(**param_dict)
            runner.train()
            self.global_step += 1

            record = {'exp_id': self.exp_id,
                      'linked_exp_id': param_dict['exp_id'],
                      'step': self.global_step,
                      'state': self.to_state(),
                      'params': self.to_params()}
            self.dbinterface.save(record)
            print('saved', record['exp_id'])

"""ptutils Runner."""

import logging

from ptutils.base import Base
from .error import StepError, ExpIDError, LoadError

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel('DEBUG')


class Runner(Base):
    """Summary.

    Attributes:
        dataprovider (TYPE): Description.
        dbinterface (TYPE): Description.
        exp_id (TYPE): Description.
        global_step (int): Description.
        load_params (TYPE): Description.
        model (TYPE): Description.
        save_params (TYPE): Description.
        train_params (TYPE): Description.

    """

    def __init__(self,
                 exp_id,
                 model=None,
                 global_step=None,
                 dbinterface=None,
                 dataprovider=None,
                 train_params=None,
                 save_params=None,
                 load_params=None,
                 **kwargs):
        """Initialize the :class:`Runner` class.

        Args:
            exp_id (str, optional): Description.
            model (Model, optional): Description.
            dbinterface (DBInterface, optional): Description.
            dataprovider (DataProvider, optional): Description.
            train_params (dict, optional): Description.
            save_params (dict, optional): Description.
            load_params (dict, optional): Description.
            **kwargs: Additional attr required by runner.

        """
        super(Runner, self).__init__(**kwargs)
        # Restore previous run.
        # no loaded params -- load from user specified params

        # Core bases.

        self.model = model
        self.dbinterface = dbinterface
        self.dataprovider = dataprovider

        # Params.
        self.save_params = save_params
        self.load_params = load_params
        self.train_params = train_params

        self.exp_id = exp_id
        self.global_step = global_step

# -- Runner Properties ---------------------------------------------------------

    @classmethod
    def init(cls, **params):
        runner = Base.from_params(**params)
        if runner.load_params['restore']:
            loaded_run = runner.load_run()
            loaded_params = loaded_run['params']
            loaded_state = loaded_run['state']

            if loaded_params:
                # loaded_params.update(runner.to_params())
                loaded_params = Runner._replace_params(runner.to_params(), loaded_params)
                runner = Runner.from_params(**loaded_params)
                runner.from_state(loaded_state)

        if runner.exp_id is None:
            error_msg = 'Cannot run an experiment without an exp_id'
            log.critical(error_msg)
            raise ExpIDError(error_msg)
        return runner

    @property
    def exp_id(self):
        return self._params['exp_id']

    @exp_id.setter
    def exp_id(self, value):
        self._params['exp_id'] = value

    @property
    def global_step(self):
        if self._params['global_step'] is None:
            self._params['global_step'] = 0
        return self._params['global_step']

    @global_step.setter
    def global_step(self, value):
        # if value <= self._params['global_step']:
            # raise StepError('The global step should have been incremented.')
        if value is None:
            self._params['global_step'] = value
        elif value > (self._params['global_step'] + 1):
            raise StepError('The global step can only be incremented by one.')
        elif value < 0:
            raise StepError('The global step cannot be negative.')
        else:
            self._params['global_step'] = value

    @property
    def model(self):
        """Get the model."""
        return self._bases['model']

    @model.setter
    def model(self, value):
        self._bases['model'] = value

    @property
    def dbinterface(self):
        return self._bases['dbinterface']

    @dbinterface.setter
    def dbinterface(self, value):
        self._bases['dbinterface'] = value

    @property
    def dataprovider(self):
        return self._bases['dataprovider']

    @dataprovider.setter
    def dataprovider(self, value):
        self._bases['dataprovider'] = value

    @property
    def save_params(self):
        """Get the save parameters."""
        return self._params['save_params']

    @save_params.setter
    def save_params(self, value):
        self._params['save_params'] = value

    @property
    def load_params(self):
        """Get the load parameters."""
        return self._params['load_params']

    @load_params.setter
    def load_params(self, value):
        self._params['load_params'] = value

    @property
    def train_params(self):
        """Get the train parameters."""
        return self._params['train_params']

    @train_params.setter
    def train_params(self, value):
        self._params['train_params'] = value

# -- Runner Methods ------------------------------------------------------------

    def setup_eval(self, prev_output):
        """Set up the model for evaluation."""
        self.model.eval()
        return prev_output

    def step(self, prev_output):
        """Define a single step of an experiment.

        This must increment the global step. A common use case
        will be to simply make a forward pass update the model.

        Formally, this will call model.forward(), whose output should
        be used by the dataprovider to provide the next batch of data.

        """
        data = self.dataprovider.provide(prev_output, mode='train')
        output = self.model.step(data)

        log.info('step: {}; loss: {}'.format(self.global_step,
                                             output['loss'].data[0]))
        return output

    def inference(self, prev_output):
        """Define a single step of an experiment.

        This must increment the global step. A common use case
        will be to simply make a forward pass update the model.

        Formally, this will call model.forward(), whose output should
        be used by the dataprovider to provide the next batch of data.

        """
        data = self.dataprovider.provide(prev_output, mode='test')[0]
        output = self.model.inference(data)
        return output

    def train(self):
        """Define the primary training loop.

        The default behavior is to step the trainer and
        save intermediate results.

        """
        model_output = None
        for step in range(self.global_step, self.train_params['num_steps']):
            model_output = self.step(model_output)

            if self.global_step % self.save_params['metric_freq'] == 0:

                # Save desired results.
                record = {'exp_id': self.exp_id,
                          'step': self.global_step,
                          'loss': model_output['loss'],
                          'state': self.to_state(),
                          'params': self.to_params(),
                          }
                log.info("Saving step {}".format(self.global_step))
                self.dbinterface.save(record)

            # if val_freq % 0:
                # val_model_output = None
                # for val_step in self.validation_params['num_steps']
                # val_model_output = self.validation_step(val_model_output)
            # You may want to do additional computation
            # in between steps.
            self.global_step += 1

        record = {'exp_id': self.exp_id,
                  'step': self.global_step,
                  'state': self.to_state(),
                  'params': self.to_params(),
                  }
        log.info("Saving step {}".format(self.global_step))
        self.dbinterface.save(record)

    def train_from_params(self, **params):
        """Run the execution of an experiment.

        This is the primary entrance to the Runner class.

        """
        log.info('Beginning experiment: {}'.format(self.exp_id))

        if self.exp_id is None:
            error_msg = 'Cannot run an experiment without an exp_id'
            log.critical(error_msg)
            raise ExpIDError(error_msg)

        self.base_cuda()

        # Start the main training loop, if desired.
        if self.train_params['train']:
            self.train()
        else:
            log.info('Skipping training.')

        log.info('Training complete!')

    def predict(self):
        # TODO
        raise NotImplementedError

    def test(self):
        """Perform inference.

        """
        model_output = None
        all_model_outputs = []
        for step in range(self.validation_params['num_steps']):
            model_output = self.inference(model_output)
            all_model_outputs.append(model_output)

        # Save desired results.
        record = {'exp_id': self.exp_id,
                  'output': all_model_outputs,
                  'params': self.to_params(),
                  }
        self.dbinterface.save(record)

    def test_from_params(self):

        log.info('Beginning experiment: {}'.format(self.exp_id))
        # if self.load_params['restore']:
        #     loaded_run = self.load_run()
        #     loaded_params = loaded_run['params']
        #     loaded_state = loaded_run['state']

        #     if loaded_params:
        #         loaded_params.update(self.to_params())
        #         self = self.from_params(**loaded_params)
        #         self.from_state(loaded_state)

        # if self.exp_id is None:
        #     error_msg = 'Cannot run an experiment without an exp_id'
        #     log.critical(error_msg)
        #     raise ExpIDError(error_msg)

        self.base_cuda()

        # Start the main training loop, if desired.

        self.test()

        log.info('Validation complete!')

    def load_run(self):

        # if ['exp_id',] not in self.load_params['load_query'].keys():
        #     error_msg = 'Cannot load an experiment without an exp_id'
        #     log.critical(error_msg)
        #     ExpIDError(error_msg)

        # if (not self.exp_id == self.load_params['load_query']['exp_id']) and len(self.dbinterface.load({'exp_id': self.exp_id})) > 0:
        #     # if not resuming training, exp_id must be unique
        #     error_msg = 'Cannot run a new experiment with same exp_id as an existing record'
        #     log.critical(error_msg)
        #     raise ExpIDError(error_msg)

        load_dbinterface = self.load_params['dbinterface']['func'](**self.load_params['dbinterface'])
        all_results = load_dbinterface.load({'exp_id': self.load_params['exp_id']})
        # all_results = self.dbinterface.load({'exp_id': self.load_params['exp_id']})

        try:
            # Load most recent run.
            return all_results[0]
        except IndexError:
            error_msg = 'No results in the database matched the load_query'
            log.critical(error_msg)
            raise LoadError(error_msg)

    @staticmethod
    def _replace_params(replacement, to_replace):
        for (key, value) in replacement.items():
            if isinstance(value, dict) and (key in to_replace.keys()):
                to_replace[key] = Runner._replace_params(value, to_replace[key])
            else:
                # if value is not None and not isinstance(value, type(None)):
                if value is not type(None):
                    to_replace[key] = value
        return to_replace

"""ptutils Runner."""

from ptutils.base import Base
from error import StepError, ExpIdError


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
                 dbinterface=None,
                 dataprovider=None,
                 train_params=None,
                 save_params=None,
                 load_params=None,
                 **kwargs):
        """Initialize the :class:`Runner` class.

        Args:
            exp_id (str): Description.
            model (Model, optional): Description.
            dbinterface (DBInterface, optional): Description.
            dataprovider (DataProvider, optional): Description.
            train_params (dict, optional): Description.
            save_params (dict, optional): Description.
            load_params (dict, optional): Description.
            **kwargs: Additional attr required by runner.

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
        self.global_step = 0

# -- Runner Properties ---------------------------------------------------------

    @property
    def exp_id(self):
        return self._params['exp_id']

    @exp_id.setter
    def exp_id(self, value):
        self._params['exp_id'] = value

    @property
    def global_step(self):
        return self._params['global_step']

    @global_step.setter
    def global_step(self, value):
        # if value <= self._params['global_step']:
            # raise StepError('The global step should have been incremented.')
        if value > (self._params['global_step'] + 1):
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

    def step(self, prev_output):
        """Define a single step of an experiment.

        This must increment the global step. A common use case
        will be to simply make a forward pass update the model.

        Formally, this will call model.forward(), whose output should
        be used by the dataprovider to provide the next batch of data.

        """
        prev_output = None
        data = self.dataprovider.provide(prev_output)
        output = self.model.step(data)

        print('step: {}; loss: {}'.format(self.global_step,
                                          output['loss'].data[0]))
        return output

    def train(self):
        """Define the primary training loop.

        The default behavior is to step the trainer and
        save intermediate results.

        """
        model_output = None
        for step in range(self.train_params['num_steps']):
            model_output = self.step(model_output)

            if self.global_step % self.save_params['metric_freq'] == 0:
                # Save desired results.
                record = {'exp_id': self.exp_id,
                          'step': self.global_step,
                          'loss': model_output['loss'].data[0],
                          'params': self.to_params(),
                          'state': self.to_state()}
                # self.dbinterface.save(record)
            # if val_freq % 0:
                # val_model_output = None
                # for val_step in self.validation_params['num_steps']
                    # val_model_output = self.validation_step(val_model_output)
            # You may want to do additional computation
            # in between steps.

            self.global_step += 1

    def train_from_params(self):
        """Run the execution of an experiment.

        This is the primary entrance to the Trainer class.

        """
        # Enforce that all Runners have an exp_id.
        if self.exp_id is None:
                raise ExpIdError('Cannot run an experiment without an exp_id')

        # Restore previous run.
        if self.load_params['restore']:
            self.load_run()

        # Prepare devices.
        self.cuda(devices=[0, 1])

        # Start the main training loop.
        self.train()

    def predict(self):
        # TODO
        pass

    def test(self):
        # TODO
        pass

    def test_from_params(self):
        # TODO
        pass

    def load_run(self):
        params = self.dbinterface.load({'exp_id': self.exp_id})
        # TODO: Raise exc if not found of exp_id collisions.
        if params is not None:
            return self.from_params(**params)
        else:
            return self
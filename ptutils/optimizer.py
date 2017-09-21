import torch
from ptutils.base import Base


class Optimizer(Base):
    def __init__(self, algorithm=None, params=None, defaults=None, **kwargs):
        """Initialize an optimizer for training.

        Args:
            algorithm (str or callable): Name of the optimizer when str, handle to the optimizer class when callable.
                If a name is provided, this optimizer looks for the optimizer in `torch.optim`
            params (dict or list of dict): Specifies the parameter group.
                Defaults to model.parameters() if None.
            **kwargs: Keyword Arguments.

        Raises:
            NotImplementedError: Description.

        """
        # super(Optimizer, self).__init__(algorithm=algorithm,
        Base.__init__(self,
                      algorithm=algorithm,
                      params=params,
                      defaults=defaults,
                      **kwargs)
        if isinstance(algorithm, str):
            optimizer_class = getattr(torch.optim, algorithm, None)
            if optimizer_class is None:
                # Look for algorithm in extensions
                optimizer_class = getattr(optimizers, algorithm, None)
            assert optimizer_class is not None, "Optimizer {} not found.".format(
                algorithm)
        elif callable(algorithm):
            optimizer_class = algorithm
        else:
            raise NotImplementedError

        self.optimizer_class = optimizer_class
        # if self.params is not None:
            # optimizer_class.__init__(self, self.params, self.defaults)

    def step(self, closure=None):
        return self.optimizer.step(closure=closure)

    def zero_grad(self):
        return self.optimizer.zero_grad()

    def compute_gradients(self, loss):
        loss.backward()

    def apply_gradients(self):
        self.step()

    def optimize(self, loss):
        self.compute_gradients(loss)
        self.apply_gradients()

    def __repr__(self):
        return Base.__repr__(self)

# class Optimizer(torch.optim.Optimizer):
#     __name__ = 'optimizer'

#     def __init__(self, optimizer):
#         base.Optimizer.__init__(self)
#         self.state = defaultdict(dict)
#         self.params = []
#         self.optimizer_cls = optimizer

#     def step(self, closure=None):
#         return self.optimizer(closure=closure)

#     def zero_grads(self):
#         return self.optimizer.zero_grads()

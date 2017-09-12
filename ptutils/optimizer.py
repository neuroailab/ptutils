import torch

from ptutils.base import Base

class Optimizer(torch.optim.Optimizer):
    def __init__(self, optimizer, param_groups=None, **kwargs):
        """Initialize an optimizer for training.

        Args:
            optimizer (str or callable): Name of the optimizer when str, handle to the optimizer class when callable.
                If a name is provided, this optimizer looks for the optimizer in `torch.optim`
            param_groups (list of dict): Specifies the parameter group. Defaults to model.parameters() if None.
            **kwargs: Keyword Arguments.

        Raises:
            NotImplementedError: Description.

        """
        if isinstance(optimizer, str):
            optimizer_class = getattr(torch.optim, optimizer, None)
            if optimizer_class is None:
                # Look for optimizer in extensions
                optimizer_class = getattr(optimizers, optimizer, None)
            assert optimizer_class is not None, "Optimizer {} not found.".format(
                optimizer)
        elif callable(optimizer):
            optimizer_class = optimizer
        else:
            raise NotImplementedError
        # param_groups = self.model.parameters() if param_groups is None else param_groups
        # self._optimizer = optimizer_class(param_groups, **kwargs)

    def step(self, closure=None):
        return self._optimizer(closure=closure)

    def zero_grad(self):
        return self._optimizer.zero_grad()

    def compute_gradients(self, loss):
        loss.backward()

    def apply_gradients(self):
        self.step()

    def optimize(self, loss):
        self.compute_gradients(loss)
        self.apply_gradients()


class Optimizer(torch.optim.Optimizer):
    __name__ = 'optimizer'

    def __init__(self, optimizer):
        base.Optimizer.__init__(self)
        self.state = defaultdict(dict)
        self.param_groups = []
        self.optimizer_cls = optimizer

    def step(self, closure=None):
        return self.optimizer(closure=closure)

    def zero_grads(self):
        return self.optimizer.zero_grads()

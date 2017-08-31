"""PTutils exeptions.

Global PTutils exception and warning classes.

"""


class StepError(Exception):
    """Exception class to raise if step is not incremented."""

    pass


class LoadError(Exception):
    """Exception class to raise if there is an error during loading."""

    pass


class ExpIdError(Exception):
    """Exception class to raise if there is en error with the experiment id."""

    pass

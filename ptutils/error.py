"""PTutils exeptions.

Global PTutils exception and warning classes.

"""


class StepError(Exception):
    """Exception class to raise if step is not incremented."""

    pass


class LoadError(Exception):
    """Exception class to raise if there is an error during loading."""

    pass

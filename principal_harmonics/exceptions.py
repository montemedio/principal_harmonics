
class PhException(RuntimeError):
    """
    principal_harmonics base exception class. 
    All subtypes must define a `message` field.
    """


class ParameterException(PhException):
    message = "Got invalid parameter"


class StrategyException(PhException):
    message = "Unknown strategy"
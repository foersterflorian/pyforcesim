"""This module contains custom exceptions that are raised within the pyforcesim package."""


class AssociationError(Exception):
    """error that describes cases in which systems are not associated with each other"""


# TODO check deletion
# class NoAllocationAgentAssignedError(Exception):
#     """error that describes that a system has no assigned allocation agent"""


class ViolationStartingConditionError(Exception):
    """error occurring if a starting condition of a ConditionSetter is not met"""


class CommonSQLError(Exception):
    """error that describes a common SQL error"""


class SQLNotFoundError(Exception):
    """error if DB lookups do not contain any values"""


class SQLTooManyValuesFoundError(Exception):
    """error if DB lookups return more values than expected"""

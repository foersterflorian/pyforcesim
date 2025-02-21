"""This module contains custom exceptions that are raised within the pyforcesim package."""


class SystemNotInitialisedError(Exception):
    """error which is raised if a system is not initialised, but the
    initial state is required"""


class AssociationError(Exception):
    """error that describes cases in which systems are not associated with each other"""


class SequencingAgentAssignmentError(Exception):
    """error that describes cases in which a sequencing agent is assigned to a
    non-supported system type"""


class ViolationStartingConditionError(Exception):
    """error occurring if a starting condition of a ConditionSetter is not met"""


class InvalidPolicyError(Exception):
    """error raised if chosen policy is unknown"""


class CommonSQLError(Exception):
    """error that describes a common SQL error"""


class SQLNotFoundError(Exception):
    """error if DB lookups do not contain any values"""


class SQLTooManyValuesFoundError(Exception):
    """error if DB lookups return more values than expected"""

# user-defined errors
class AssociationError(Exception):
    """error that describes cases in which systems are 
    created but not added to supersystems"""
    pass

class NoAllocationAgentAssignedError(Exception):
    """error that describes that a system has no assigned allocation agent"""
    pass

class ViolateStartingCondition(Exception):
    """error occuring if a starting condition of a ConditionSetter is not met"""
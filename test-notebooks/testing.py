from pyforcesim.constants import SimSystemTypes
from pyforcesim.common import enum_str_values_as_frzset

print(SimSystemTypes.__members__)

ret = enum_str_values_as_frzset(SimSystemTypes)
print(ret)

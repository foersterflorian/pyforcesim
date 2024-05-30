import enum
import logging

from pyforcesim.constants import SimStates


# for i in LoggingLevels:
#     print(i)
#     print(i.name, i.value)

print('--------------')

for i in SimStates:
    print(i)
    print(i.name, i.value)

print(SimStates.__members__)

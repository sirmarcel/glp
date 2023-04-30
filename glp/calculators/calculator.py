from collections import namedtuple

# expect potential to have signature Graph -> energies[ndarray]
# .calculate has the signature (system, state) -> (result, state) and is jittable
# .update has the signature System -> Calculator and is not jittable
# state will always have the property `.overflow`, if it is `True` the result is wrong and
# you need to call .update
Calculator = namedtuple("Calculator", ("calculate", "update"))
# todo: define the interface more completely: what are the signatures for calculator/calculate?
# for external use we need unified signatures / dealing with unwanted kwargs

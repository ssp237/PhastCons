""" Basic test cases for the files """
from tree import Node

s1 = "(((One:0.200000,Two:0.300000):0.300000,(Three:0.500000," \
     "Four:0.300000):0.200000):0.300000,Five:0.700000):0.000000;"
t1 = Node.from_str(s1)
print(str(t1))
assert(str(t1).replace(" ", "") == s1)

""" Basic test cases for the files """
from tree import Node, read_data
import numpy as np

s1 = "(((One:0.200000,Two:0.300000):0.300000,(Three:0.500000," \
     "Four:0.300000):0.200000):0.300000,Five:0.700000):0.000000;"
t1 = Node.from_str(s1)
assert(str(t1).replace(" ", "") == s1)


top1 = "((human:0.075170, dog:0.117610):0.000000, (mouse:0.030590, rat:0.031610):0.142890):0.000000;"
t1 = Node.from_str(top1)
assert(str(t1) == top1)
data, l = read_data("fel test.fa")
t1.setData(data, l)
assert(round(t1.tot_prob, 2) == -292.79)
assert(round((t1 * 1).tot_prob, 2) == -292.79)

top2 = "((human:0.208430, mouse:0.033970):0.000000, (rat:0.034970, dog:0.249520):0.000000):0.000000;"
t2 = Node.from_str(top2)
assert(str(t2) == top2)
t2.setData(data, l)
assert(round(t2.tot_prob, 2) == -283.47)
assert(round((t2 * 1).tot_prob, 2) == -283.47)

top3 = "((mouse:0.033970, dog:0.249520):0.000000, (human:0.208430, rat:0.034970):0.000000):0.000000;"
t3 = Node.from_str(top3)
assert(str(t3) == top3)
t3.setData(data, l)
assert(round(t3.tot_prob, 2) == -283.47)
assert(round((t3 * 1).tot_prob, 2) == -283.47)

chimp_corner = "(bonobo:0.007840, (human:0.006550, chimp:0.006840):0.001220):0.00500;"
primates = "((gorilla:0.086940, (orangutan:0.018940, (gibbon:0.022270, (green_monkey:0.027000, " \
           "(baboon:0.008042, (rhesus:0.004991, " \
           "macaque:0.004991):0.003000):0.019610):0.022040):0.003471):0.009693):0.000500, " \
           "(bonobo:0.007840, (human:0.006550, chimp:0.006840):0.001220):0.000500):0.000000;"
t_prim = Node.from_str(primates)
d_prim, l_prim = read_data("Data/HI1-Compiled.txt")
t_prim.setData(d_prim, l_prim)
# branch_lengths = np.array(
#     [[0.07517, 0.03059, 0.03161, 0.11761, 0.14289],
#      [0.20843, 0.03397, 0.03497, 0.24952, 0.00000],
#      [0.20843, 0.03397, 0.03497, 0.24952, 0.00000]], dtype=float)
#
# index = 2
# names = ['human', 'mouse', 'rat', 'dog']
# branches = [0, 1, 2, 3]
# leaves = [Node(s, None, None, bl, i) for (s, i, bl) in
#           zip(names, branches, branch_lengths[index, :])]
# branch_probs = [np.zeros((4, 4), dtype=float) for _ in range(6)]
# # Note that branch 5 (or 6 in 1-index) is the branch of 0-length
# if index == 0:
#         hum_dog = Node(None, leaves[0], leaves[3], 0, 5)
#         mouse_rat = Node(None, leaves[1], leaves[2], branch_lengths[index,4], 4)
#         root = Node('root', hum_dog, mouse_rat, None, None)
#         ordering = [leaves[0], leaves[3], hum_dog, leaves[1], leaves[2], \
#                     mouse_rat, root]
# elif index == 1:
#         hum_mouse = Node(None, leaves[0], leaves[1], 0, 5)
#         rat_dog = Node(None, leaves[2], leaves[3], branch_lengths[index, 4], 4)
#         root = Node('root', hum_mouse, rat_dog, None, None)
#         ordering = [leaves[0], leaves[1], hum_mouse, leaves[2], leaves[3], \
#                     rat_dog, root]
# else:
#         mouse_dog = Node(None, leaves[1], leaves[3], 0, 5)
#         hum_rat = Node(None, leaves[0], leaves[2], branch_lengths[index, 4], 4)
#         root = Node('root', mouse_dog, hum_rat, None, None)
#         ordering = [leaves[1], leaves[3], mouse_dog, leaves[0], leaves[2], \
#                     hum_rat, root]


""" Basic test cases for the files """
from tree import Node, read_data, read_dist
import numpy as np
from scipy.optimize import minimize

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
primates = "((Gorilla:0.086940, (Orangutan:0.018940, (Gibbon:0.022270, (Green_monkey:0.027000, " \
           "(Baboon:0.008042, (Rhesus:0.004991, " \
           "Crab_eating_macaque:0.004991):0.003000):0.019610):0.022040):0.003471):0.009693):0.000500, " \
           "(Bonobo:0.007840, (Human:0.006550, Chimp:0.006840):0.001220):0.000500):0.000000;"

primates2 = "((Gorilla:0.086940, (Orangutan:0.018940, (Gibbon:0.022270, (Green_monkey:0.027000, " \
            "(Baboon:0.008042, (Rhesus:0.004991, " \
            "Crab_eating_macaque:0.004991):0.003000):0.019610):0.022040):0.003471):0.009693):0.000500, " \
            "(Human:0.006550, Chimp:0.006840):0.000500):0.000000;"
d_prim, l_prim = read_data("Data/H1-1/Data_total.txt")
t_prim = Node.from_str(primates2)
# d_prim, l_prim = read_data("Data/HI1-Compiled.txt")
t_prim.setData(d_prim, l_prim)
print(t_prim.tot_prob)
t_prim.E(l_prim)
# W = t_prim.findW()
# g = t_prim.kruskal(W)
# g.setChildren()
# g.print_deg()
# print(" ------- ")
# g.bifurcate_step_1()
# g.print_deg()
# print(g.size())
# print(g.children[0].children[0].name)
# print(" ------- ")
# g = g.bifurcate_step_2()
# g.print_deg()
# t_new = g.to_node()
# print(t_new)
# print(list(map(lambda x: x.id, t_new.preorder())))
prim_dist, mapping = read_dist("dist10.txt")
t_primnj = Node.neighbor_join(prim_dist)
t_primnj.swap_names(mapping)
# print(t_primnj)
t_primnj.setData(d_prim, l_prim)
print(t_primnj.tot_prob)
t = Node.EM(t_prim, d_prim, l_prim)

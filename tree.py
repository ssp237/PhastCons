"""
File to store data on phylogenetic trees
"""
import re
import numpy as np


def read_data(filename):
    """ Reads data from ```filename``` in fasta format.
hum_dog = Node(None, leaves[0], leaves[3], 0, 5)
mouse_rat = Node(None, leaves[1], leaves[2], branch_lengths[index,4], 4)
root = Node('root', hum_dog, mouse_rat, None, None)
ordering = [leaves[0], leaves[3], hum_dog, leaves[1], leaves[2], mouse_rat, root]
    Arguments:
        filename: name of fasta file to read
    Returns:
        sequences: dictionary of outputs (string (sequence id) -> sequence (string))
        size: length of each sequence
    """
    with open(filename, "r") as f:
        lines = f.readlines()
        sequences = {}
        output = ""
        size = 0
        curr = ""
        for l in lines:
            if l[0] == ">":
                if len(output) != 0:
                    sequences[curr] = output
                    size = len(output)
                    output = ""
                curr = l[2:].strip()
            else:
                output += l.strip().upper()
        sequences[curr] = output
    return sequences, size


class Node:
    def __init__(self, name, left, right, branch_len=0.0, this_id=0, parent_id=0):
        """ Initializes a node with given parameters.
            Arguments:
                name: name of node (only relevant for leaves)
                left: left child (Node)
                right: right child (Node)
                branch_len: length to parent
        """
        bases = 'ACGT'
        self.name = name
        self.left = left
        self.right = right
        self.branch_length = branch_len if branch_len else 0.0
        self.id = this_id
        self.parent_id = parent_id
        self.probs = [None for _ in bases]
        self.bp = np.array([[Node.jcm(b, a, self.branch_length) for b in bases] for a in bases])
        self.data = {}
        self.seqlen = 0
        self.fel_probs = np.array([])
        self.tot_prob = 0

    def is_leaf(self):
        return (self.left is None) and (self.right is None)

    def __getitem__(self, item):
        """
        Return cached felsenstein prob at index
        :param item: index of felsenstein probability to return
        :return: cached felsenstein prob at index item
        """
        return self.fel_probs[item]

    def swap_names(self, mapping):
        """
        Swap names out for those given in the mapping (if applicable)
        """
        if self.name in mapping:
            self.name = mapping[self.name]
        if self.left is not None:
            self.left.swap_names(mapping)
        if self.right is not None:
            self.right.swap_names(mapping)

    def __str__(self):
        """
        :return: Newick representation of this tree
        """
        if self.is_leaf():
            out = f"{self.name}:{self.branch_length:.6f}"
        elif self.right is None:
            out = f"({self.left}){self.name}:{self.branch_length:.6f}"
        elif self.left is None:
            out = f"({self.right}){self.name}:{self.branch_length:.6f}"
        else:
            out = f"({self.left}, {self.right}):{self.branch_length:.6f}"
        if self.parent_id == -1:  # Root node
            out += ";"
        return out

    @classmethod
    def from_str(cls, string):
        """
        Return a node formed from a tree in newick form
        :param string: Newick form of a phylogenetic tree
        :return: Tree representing string
        """
        tokens = re.findall(r"([^:;,()\s]*)(?:\s*:\s*([\d.]+)\s*)?([,);])|(\S)", string)

        def helper(nextid=0, parentid=-1):  # one node
            thisid = nextid
            children = []

            name, length, delim, ch = tokens.pop(0)
            if ch == "(":
                while ch in "(,":
                    node, ch, nextid = helper(nextid + 1, thisid)
                    children.append(node)
                name, length, delim, ch = tokens.pop(0)
            length = float(length) if length else 0.0
            if not children:
                out = Node(name, None, None, length, thisid, parentid)
            elif len(children) == 1:
                out = Node(name, children[0], None, length, thisid, parentid)
            elif len(children) == 2:
                out = Node(name, children[0], children[1], length, thisid, parentid)
            else:
                raise TypeError("Passed a non-binary tree to from_str!")
            return out, delim, nextid

        return helper()[0]

    @classmethod
    def jcm(cls, b, a, t, u=1.0):
        """ Evaluates P(b|a, t) under the Jukes-Cantor model

        Arguments:
            b: descendant base (string)
            a: ancestral base (string)
            t: branch length (float)
            u: mutation rate (float, defaults to 1)
        Returns:
            prob: float probability P(b|a, t)
        """
        e_4ut = np.exp(-4 * u * t / 3)
        if a == b:
            return 0.25 * (1 + 3 * e_4ut)
        else:
            return 0.25 * (1 - e_4ut)

    def fel_at_ind(self, ind, data, fel_probs):
        bases = 'ACGT'
        if self.is_leaf():
            c = data[self.name][ind]
            self.probs = [int(c == a) for a in bases]
            fel_probs[ind] = np.log(0.25 * np.sum(self.probs))
            return

        if self.left:
            self.left.fel_at_ind(ind, data, fel_probs)
        if self.right:
            self.right.fel_at_ind(ind, data, fel_probs)

        for i_a, a in enumerate(bases):
            p_i, p_j = 0, 0
            for i_b, b in enumerate(bases):
                p_i += (self.left.probs[i_b] * self.left.bp[i_a, i_b])
            for i_c, c in enumerate(bases):
                p_j += (self.right.probs[i_c] * self.right.bp[i_a, i_c])
            self.probs[i_a] = p_i * p_j

        fel_probs[ind] = np.log(0.25 * np.sum(self.probs))

    def felsenstein(self):
        """
        Calculate and cache felsenstein probability of this tree
        :return:
        """
        for i in range(self.seqlen):
            self.fel_at_ind(i, self.data, self.fel_probs)
        self.tot_prob = np.sum(self.fel_probs)

    def setData(self, data, seqlen):
        """
        Set data and cache felsenstein probabilities
        :param seqlen: length of data
        :param data: Sequences for leaves of the phylogeny
        :return: None
        """
        self.data = data
        self.seqlen = seqlen
        self.fel_probs = np.zeros(seqlen)
        self.felsenstein()

    def __mul__(self, other):
        """
        Multiply branch lengths by constant
        :param other: (float) constant to scale branch lengths
        :return: Tree with new branch lengths. Cached data _is_ transferred.
        """
        left = self.left * other if self.left else None
        right = self.right * other if self.left else None
        out = Node(self.name, left, right, self.branch_length, self.id, self.parent_id)

        if self.data:
            out.setData(self.data, self.seqlen)

        return out

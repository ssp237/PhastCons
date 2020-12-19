"""
File to store data on phylogenetic trees
"""
import re


class Node:
    def __init__(self, name, left, right, parent_len=0.0, this_id=0, parent_id=0):
        """ Initializes a node with given parameters.
            Arguments:
                name: name of node (only relevant for leaves)
                left: left child (Node)
                right: right child (Node)
                parent_len: length to parent
        """
        self.name = name
        self.left = left
        self.right = right
        self.branch_length = parent_len
        self.id = this_id
        self.parent_id = parent_id

    def is_leaf(self):
        return (self.left is None) and (self.right is None)

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
        if self.left is None or self.right is None:
            return f"{self.name}:{self.branch_length:.6f}"
        elif self.branch_length == 0:  # Root node
            return f"({self.left}, {self.right}):{self.branch_length:.6f};"
        return f"({self.left}, {self.right}):{self.branch_length:.6f}"

    @classmethod
    def from_str(cls, string):
        """
        Return a node formed from a tree in newick form
        :param string: Newick form of a phylogenetic tree
        :return: Tree representing string
        """
        tokens = re.findall(r"([^:;,()\s]*)(?:\s*:\s*([\d.]+)\s*)?([,);])|(\S)", string)

        def help(nextid=0, parentid=-1):  # one node
            thisid = nextid
            children = []

            name, length, delim, ch = tokens.pop(0)
            if ch == "(":
                while ch in "(,":
                    node, ch, nextid = help(nextid + 1, thisid)
                    children.append(node)
                name, length, delim, ch = tokens.pop(0)
            length = float(length)
            if not children:
                out = Node(name, None, None, length, thisid, parentid)
            elif len(children) == 1:
                out = Node(name, children[0], None, length, thisid, parentid)
            elif len(children) == 2:
                out = Node(name, children[0], children[1], length, thisid, parentid)
            else:
                raise TypeError("Passed a non-binary tree to from_str!")
            return out, delim, nextid

        return help()[0]

    def felsenstein(self, seq):
        """
        Calculate felsenstein probability of this tree
        :param seq: Sequence? maybe?
        :return:
        """
        pass

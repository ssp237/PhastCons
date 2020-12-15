"""
File to store data on phylogenetic trees
"""

class Node:
    def __init__(self, name, left, right):
        """ Initializes a node with given parameters.
            Arguments:
                name: name of node (only relevant for leaves)
                left: left child (Node)
                right: right child (Node)
        """
        self.name = name
        self.left = left
        self.right = right
        self.branch_length = 0

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
            return "{}:{:.6f}".format(self.name, self.branch_length)
        elif self.branch_length == 0:  # Root node
            return "({}, {});".format(str(self.left), str(self.right))
        return "({}, {}):{:.6f}".format(str(self.left), str(self.right), self.branch_length)

    @classmethod
    def from_str(cls, string):
        """
        Return a node formed from a tree in newick form
        :param string: Newick form of a phylogenetic tree
        :return: Tree representing string
        """
        # Note to Audrey - @classmethod makes this into a static method
        # so we can call, for ex, Node.from_str(<string>)
        pass

"""
File to store data on phylogenetic trees
"""
import re
import numpy as np
from scipy.optimize import minimize
from collections import deque


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
        size = min([len(v) for k, v in sequences.items()])
        new_sequences = {k: "" for k, _ in sequences.items()}
        for i in range(size):
            if np.all([v[i] == '-' for _, v in sequences.items()]):
                for k, v in new_sequences.items():
                    new_sequences[k] = v + sequences[k][i]
                size -= 1

    return sequences, size


def read_dist(distances_file):
    """ Reads the input file of distances between the sequences

    Arguments:
        distances_file: file name of distances between sequences
    Returns:
        D: matrix of distances (map of maps)
        mapping: index to name mapping (dictionary)
    """
    with open(distances_file, "r") as f:
        lines = [l.strip().split() for l in f.readlines()]
        mapping = {i: s for i, s in enumerate(lines[0])}
        lines = [l[1:] for l in lines[1:]]
        D = {i: {} for i in range(len(lines))}
        for i, l in enumerate(lines):
            for j, sval in enumerate(l):
                D[i][j] = float(sval)
    return D, mapping


# Classes for Kruskal's algorithm on generic graphs


class Edge:
    def __init__(self, length, left, right):
        self.length = length
        self.left = left
        self.right = right


class Vertex:
    def __init__(self, this_id, name):
        self.id = this_id
        self.neighbors = []
        self.children = []
        self.parent_len = 0.0
        self.name = name
        self.visited = False
        self.parent = None

    def is_leaf(self):
        return not self.children

    def d(self):
        return len(self.neighbors)

    def setChildren(self):
        self.visited = True
        for w, n in self.neighbors:
            if not n.visited:
                n.visited = True
                self.children.append(n)
                n.parent = self
                n.parent_len = w
                n.setChildren()

    def unvisit(self):
        self.visited = False
        for c in self.children:
            c.unvisit()

    def bifurcate_step_1(self):
        to_visit = deque([self])
        cur = self
        while to_visit:
            cur = to_visit.pop()
            children = deque(cur.children)
            while children:
                c = children.pop()
                if c.d() == 1 and not c.name:
                    cur.children.remove(c)
                    cur.neighbors.remove((c.parent_len, c))
                    c.neighbors.remove((c.parent_len, cur))
                elif c.d() == 2 and not c.name:
                    cur.children.remove(c)
                    cur.neighbors.remove((c.parent_len, c))
                    c.neighbors.remove((c.parent_len, cur))
                    assert(len(c.children) == 1)
                    new_child = c.children[0]
                    new_child.parent = cur
                    new_child.neighbors.remove((new_child.parent_len, c))
                    new_child.parent_len += c.parent_len
                    cur.children.append(new_child)
                    cur.neighbors.append((new_child.parent_len, new_child))
                    children.append(new_child)
                    new_child.neighbors.append((new_child.parent_len, cur))
                else:
                    to_visit.appendleft(c)

    def bifurcate_step_2(self):
        to_visit = deque([self])
        root = self
        while to_visit:
            cur = to_visit.pop()
            if cur.name:
                if cur.d() > 1:
                    new_parent = Vertex(-1, "")
                    new_parent.neighbors.append((0, cur))
                    cur.neighbors.append((0, new_parent))
                    new_parent.children.append(cur)
                    new_parent.parent, cur.parent = cur.parent, new_parent
                    new_parent.parent_len, cur.parent_len = cur.parent_len, 0
                    daddy = new_parent.parent
                    if daddy:
                        new_parent.neighbors.append((new_parent.parent_len, daddy))
                        daddy.children.append(new_parent)
                        daddy.children.remove(cur)
                        daddy.neighbors.append((new_parent.parent_len, new_parent))
                        daddy.neighbors.remove((new_parent.parent_len, cur))
                        cur.neighbors.remove((new_parent.parent_len, daddy))
                    else:
                        root = new_parent
                        to_visit.appendleft(root)
                    for c in cur.children:
                        new_parent.neighbors.append((c.parent_len, c))
                        new_parent.children.append(c)
                        cur.children.remove(c)
                        cur.neighbors.remove((c.parent_len, c))
                        c.parent = new_parent
                        c.neighbors.remove((c.parent_len, cur))
                        c.neighbors.append((c.parent_len, new_parent))
                    to_visit.appendleft(cur)
                else:
                    for c in cur.children:
                        to_visit.append(c)
            else:
                if cur.d() > 3 or (cur.d() == 3 and not cur.parent):
                    new_parent = Vertex(-1, "")
                    new_parent.neighbors.append((0, cur))
                    cur.neighbors.append((0, new_parent))
                    new_parent.children.append(cur)
                    new_parent.parent, cur.parent = cur.parent, new_parent
                    new_parent.parent_len, cur.parent_len = cur.parent_len, 0
                    daddy = new_parent.parent
                    if daddy:
                        new_parent.neighbors.append((new_parent.parent_len, daddy))
                        daddy.children.append(new_parent)
                        daddy.children.remove(cur)
                        daddy.neighbors.append((new_parent.parent_len, new_parent))
                        daddy.neighbors.remove((new_parent.parent_len, cur))
                        cur.neighbors.remove((new_parent.parent_len, daddy))
                    else:
                        root = new_parent
                        to_visit.appendleft(root)
                    for c in cur.children[2:]:
                        new_parent.neighbors.append((c.parent_len, c))
                        new_parent.children.append(c)
                        cur.children.remove(c)
                        cur.neighbors.remove((c.parent_len, c))
                        c.parent = new_parent
                        c.neighbors.remove((c.parent_len, cur))
                        c.neighbors.append((c.parent_len, new_parent))
                    to_visit.appendleft(cur)
                    for c in cur.children:
                        to_visit.append(c)
                else:
                    for c in cur.children:
                        to_visit.append(c)
        return root

    def print_deg(self):
        def printer(node, tabs):
            print(("\t" * tabs) + str(len(node.children)), node.name)
            for c in node.children:
                printer(c, tabs + 1)
        printer(self, 0)

    def size(self):
        return 1 + sum(map(lambda x: x.size(), self.children))

    def node_of_vertex(self):
        if len(self.children) == 0:
            return Node(self.name, None, None, self.parent_len, self.id)
        if len(self.children) == 1:
            return Node(self.name, self.children[0].node_of_vertex(), None,
                        self.parent_len, self.id)
        if len(self.children) == 2:
            return Node(self.name, self.children[0].node_of_vertex(),
                        self.children[1].node_of_vertex(), self.parent_len, self.id)
        raise TypeError("Attempted to convert non binary tree to phylogeny!")

    def to_node(self):
        out = self.node_of_vertex()
        out.assignIDs()
        return out


class Graph:
    def __init__(self, vertices):
        self.vertices = sorted(vertices, key=lambda v: v.id)
        self.V = len(vertices)
        self.graph = []

    # function to add an edge to graph
    def addEdge(self, u, v, w):
        self.graph.append([u, v, w])

    # A utility function to find set of an element i
    # (uses path compression technique)
    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])

    # A function that does union of two sets of x and y (uses union by rank)
    def union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)

        # Attach smaller rank tree under root of high rank tree (Union by Rank)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot

        # If ranks are same, then make one as root and increment its rank by one
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    # The main function to construct MST using Kruskal's algorithm
    def KruskalMST(self):
        result = []  # This will store the resultant MST
        # An index variable, used for sorted edges
        i = 0
        # An index variable, used for result[]
        e = 0

        # Step 1:  Sort all the edges in non-decreasing order of their weight.
        self.graph = list(reversed(sorted(self.graph, key=lambda item: item[2])))

        parent = []
        rank = []

        # Create V subsets with single elements
        for node in self.vertices:
            parent.append(node.id)
            rank.append(0)

        # Number of edges to be taken is equal to V-1
        while e < self.V - 1:

            # Step 2: Pick the smallest edge and increment the index for next iteration
            u, v, w = self.graph[i]
            i = i + 1
            x = self.find(parent, u)
            y = self.find(parent, v)

            # If including this edge does't cause cycle, include it in result
            #  and increment the indexof result for next edge
            if x != y:
                e = e + 1
                result.append([u, v, w])
                self.vertices[u].neighbors.append((w, self.vertices[v]))
                self.vertices[v].neighbors.append((w, self.vertices[u]))

                self.union(parent, rank, x, y)
            # Else discard the edge

        minimumCost = 0
        print("Edges in the constructed MST")
        for u, v, weight in result:
            minimumCost += weight
            print("%d -- %d == %d" % (u, v, weight))
        print("Minimum Spanning Tree", minimumCost)


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
        self.E_abij = None

    def setParentLen(self, l):
        bases = 'ACGT'
        self.branch_length = l
        self.bp = np.array([[Node.jcm(b, a, self.branch_length) for b in bases] for a in bases])

    def is_leaf(self):
        return (self.left is None) and (self.right is None)

    def assignIDs(self):
        for i, n in enumerate(self.preorder()):
            n.id = i

        self.parent_id = -1

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
            if c in bases:
                self.probs = [int(c == a) for a in bases]
            else:
                self.probs = [1 for _ in bases]
            fel_probs[ind] = np.log(0.25 * np.sum(self.probs))
            return

        if self.left:
            self.left.fel_at_ind(ind, data, fel_probs)
        if self.right:
            self.right.fel_at_ind(ind, data, fel_probs)

        for i_a, a in enumerate(bases):
            p_i, p_j = 0, 0
            if self.left:
                for i_b, b in enumerate(bases):
                    p_i += (self.left.probs[i_b] * self.left.bp[i_a, i_b])
            else:
                p_i = 0
            if self.right:
                for i_c, c in enumerate(bases):
                    p_j += (self.right.probs[i_c] * self.right.bp[i_a, i_c])
            else:
                p_j = 0
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
        right = self.right * other if self.right else None
        out = Node(self.name, left, right, self.branch_length * other, self.id, self.parent_id)

        if self.data:
            out.setData(self.data, self.seqlen)

        return out

    def tot_branch_len(self):
        """
        Sum of branch lengths in this tree
        :return:
        """
        return self.branch_length + \
               (self.left.tot_branch_len() if self.left else 0.0) + \
               (self.right.tot_branch_len() if self.right else 0.0)

    def size(self):
        """
        Number of nodes in this tree
        """
        return 1 + \
               (self.left.size() if self.left else 0) + \
               (self.right.size() if self.right else 0)

    def E_at_ind(self, ind, visited, data, E_abij):
        """
        Preform E from paper at single index and cache results in E_abij
        :param data: data from root node
        :param E_abij: array to cache results
        :param ind: index to preform step at
        :param visited: list of nodes visited on the way to this one
        :return:
        """
        bases = 'ACGT'
        n = len(bases)
        if self.id != 0:
            last = visited[-1]
            tot = np.sum(last.probs)
            for a in range(n):
                for b in range(n):
                    E_abij[a, b, self.id, last.id] = \
                        (0.25 * self.probs[a] * self.bp[a, b] * last.probs[a]) / tot
            for node in list(reversed(visited))[1:]:
                for a in range(n):
                    for b in range(n):
                        E_abij[a, b, self.id, node.id] = \
                            E_abij[a, b, self.id, last.id] * E_abij[a, b, last.id, node.id]

        # if self.is_leaf():
        #     c = data[self.name][ind]
        #     self.probs = [int(c == a) for a in bases]
        #     # fel_probs[ind] = np.log(0.25 * np.sum(self.probs))
        #     return

        visited.append(self)
        if self.left:
            self.left.E_at_ind(ind, visited, data, E_abij)
        if self.right:
            self.right.E_at_ind(ind, visited, data, E_abij)
        visited.pop()
        # fel_probs[ind] = np.log(0.25 * np.sum(self.probs))

    def E(self, seqlen):
        """
        Preform E step from paper and store in self.E_abij
        """
        n = self.size()
        bases = 'ACGT'
        m = len(bases)
        self.E_abij = np.zeros((m, m, n, n))
        for i in range(seqlen):
            # Make sure that self.probs are right for entire tree
            self.fel_at_ind(i, self.data, self.fel_probs)
            temp = np.zeros((m, m, n, n))
            self.E_at_ind(i, [], self.data, temp)
            self.E_abij += temp

    def L_local(self, t, i, j):
        """
        Calculate l local of (i, j, t) in T using precomputed E matrix
        :return: L local(i, j, t) according to formula in paper
        """
        bases = 'ACGT'
        num_b = len(bases)
        out = 0
        for a in range(num_b):
            for b in range(num_b):
                out += self.E_abij[a, b, i, j] * (np.log(self.jcm(bases[a], bases[b], t) - np.log(0.25)))
        return -out

    def findW(self):
        n = self.size()
        out = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    break
                res = minimize(self.L_local, np.array([0.0]),
                               args=(i, j),
                               method="L-BFGS-B",
                               options={"maxiter": 250, "disp": False},
                               bounds=([(0, 1)]))
                out[i, j] = -self.L_local(res.x[0], i, j)
        return out

    def preorder(self):
        return [self] + \
               (self.left.preorder() if self.left else []) + \
               (self.right.preorder() if self.right else [])

    def kruskal(self, adj_mat):
        pre_nodes = self.preorder()
        print(len(pre_nodes))
        krusk_nodes = list(map(lambda x: Vertex(x.id, x.name), pre_nodes))
        krusk_graph = Graph(krusk_nodes)
        n = self.size()
        for i in range(n):
            for j in range(n):
                if i == j:
                    break
                krusk_graph.addEdge(i, j, adj_mat[i, j])
        krusk_graph.KruskalMST()
        assert(krusk_graph.V == len(pre_nodes))
        return krusk_nodes[0]

    @classmethod
    def neighbor_join(cls, D):
        """ Performs the neighbor joining algorithm on a given set of sequences.

        Arguments:
            D: map of maps, defining distances between the sequences
               (initially n x n, symmetric, 0 on the diagonal)
               (index -> index -> distance)
        Returns:
            Root of the phylogenic tree resulting from neighbor joining algorithm
                on D
        """
        nodes = {}
        for k in D:
            nodes[k] = Node(k, None, None)
        while len(D) > 2:
            # Find minimum distance
            totalDistance = {k: sum(v.values()) for (k, v) in D.items()}
            n = len(D)
            njM = {i: {j: 0 if i == j else ((n - 2) * Dij - totalDistance[i] - totalDistance[j])
                       for (j, Dij) in d.items()} for (i, d) in D.items()}
            min_keys = {(k, min(v, key=v.get)): min(v.values()) for (k, v) in njM.items()}
            i, j = min(min_keys, key=min_keys.get)
            name = "".join(sorted((str(i), str(j))))

            # Handle nodes
            l, r = nodes[i], nodes[j]
            l.setParentLen(0.5 * (D[i][j] + ((totalDistance[i] - totalDistance[j]) / (n - 2))))
            r.setParentLen(0.5 * (D[i][j] + ((totalDistance[j] - totalDistance[i]) / (n - 2))))
            m = Node(None, l, r)
            nodes[name] = m
            del nodes[i]
            del nodes[j]

            # Adjust D
            for k in D:
                if k != name and k != i and k != j:
                    D[k][name] = 0.5 * (D[i][k] + D[j][k] - D[i][j])
                    del D[k][i]
                    del D[k][j]

            del D[i]
            del D[j]

            D[name] = {k: v[name] for k, v in D.items()}
            D[name][name] = 0.0

        # Handle last 2 nodes
        a, b = D

        dab = sum(D[a].values()) / 2
        l, r = nodes[a], nodes[b]
        l.setParentLen(dab)
        r.setParentLen(dab)

        out = Node(None, l, r)
        out.assignIDs()
        return out

    @classmethod
    def EM(cls, filename, data, seqlen):
        """
        Run em on the distances in file filename to find the optimal phylogeny
        :param seqlen: sequence length
        :param data: Sequence data
        :param filename: File containing distances
        :return: Optimal phylogeny
        """
        D, mapping = read_dist(filename)
        t = Node.neighbor_join(D)
        t.swap_names(mapping)
        print(t)
        t.setData(data, seqlen)
        p = t.tot_prob
        print(p)
        for i in range(10):
            t.E(seqlen)
            adj = t.findW()
            g = t.kruskal(adj)
            g.setChildren()
            g.bifurcate_step_1()
            g = g.bifurcate_step_2()
            t_new = g.to_node()
            t_new.setData(data, seqlen)
            # t_new = t_new * (1 / t_new.tot_branch_len())
            p_new = t_new.tot_prob
            print(p_new)
            t = t_new
            print(t)

        return t

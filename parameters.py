#!/usr/bin/env python3

"""Script for computing posterior probabilities of hidden states at each
   position of a given sequence.
Arguments:
    -f: file containing the sequence (fasta file)
    -mu: the probability of switching states

Outputs:
    posteriors.csv - a KxN matrix outputted as a CSV with the posterior
                     probability of each state at each position

Example Usage:
    python 1b.py -f hmm-sequence.fa -mu 0.01
"""

import argparse
import numpy as np
from numpy import log1p
from numpy import exp
from scipy.special import logsumexp
from scipy.optimize import minimize
from tree import read_data, Node
import matplotlib.pyplot as plt


def sumLogProb(a, b):
    """
    Approximate log(a' + b') given log(a') and log(b') as given in lecture.
    :param a: log(a') for some a'
    :param b: log(b') for some b'
    :return: log(a' + b')
    """
    if a > b:
        return a + log1p(exp(b - a))
    else:
        return b + log1p(exp(a - b))


def read_fasta(filename):
    """Reads the fasta file and outputs the sequence to analyze.
    Arguments:
        filename: name of the fasta file
    Returns:
        s: string with relevant sequence
    """
    with open(filename, "r") as f:
        s = ""
        for l in f.readlines()[1:]:
            s += l.strip()
    return s


def unfold_params(params, nstates=2):
    """
    Unfold 1d array of params into init and trans probs
    :param params: 1d array of init and trans probs
    :param nstates: number of states in the HMM
    :return: init and trans probs
    """
    init = params[:nstates]
    # emiss = params[nstates:nstates * 5].reshape((nstates, 4))
    trans = params[nstates:].reshape((nstates, nstates))
    return init, trans


def forward(obs, trans_probs, init_probs, trees):
    """ Outputs the forward and backward probabilities of a given observation.
    Arguments:
        obs: observed sequence of emitted states (list of emissions)
        trans_probs: transition log-probabilities (dictionary of dictionaries)
        init_probs: initial log-probabilities for each hidden state (dictionary)
        trees: numpy array of trees
    Returns:
        F: matrix of forward probabilities
        likelihood_f: P(obs) calculated using the forward algorithm
    """
    ind_bases = {b: i for i, b in enumerate('ACGT')}
    n, m = init_probs.shape[0], obs
    F = np.zeros((n, m))

    # forwards
    for i, p in enumerate(init_probs):
        F[i, 0] = p + trees[i][0]

    for i in range(obs)[1:]:
        for j in range(n):
            e = trees[j][i]
            probs = [F[k, i - 1] + trans_probs[k][j] for k in range(n)]
            F[j, i] = e + logsumexp(probs)

    likelihood_f = logsumexp(F[:, -1])

    return likelihood_f


def forward_backward(obs, trans_probs, init_probs, trees):
    """ Outputs the forward and backward probabilities of a given observation.
    Arguments:
        obs: observed sequence of emitted states (list of emissions)
        trans_probs: transition log-probabilities (dictionary of dictionaries)
        trees: numpy array of trees
        init_probs: initial log-probabilities for each hidden state (dictionary)
    Returns:
        F: matrix of forward probabilities
        likelihood_f: P(obs) calculated using the forward algorithm
        B: matrix of backward probabilities
        likelihood_b: P(obs) calculated using the backward algorithm
        R: matrix of posterior probabilities
    """
    n, m = len(init_probs), obs
    F = np.zeros((n, m))
    B = np.zeros((n, m))
    R = np.zeros((n, m))

    # forwards
    for i, p in enumerate(init_probs):
        F[i, 0] = p + trees[i][0]

    for i in range(obs)[1:]:
        for j in range(n):
            e = trees[j][i]
            probs = [F[k, i - 1] + trans_probs[k][j] for k in range(n)]
            F[j, i] = e + logsumexp(probs)

    likelihood_f = logsumexp(F[:, -1])

    # backwards (note that log(1) = 0, so last col is already set)
    for i in reversed(list(range(obs))[1:]):
        for j in range(n):
            probs = [trans_probs[j][k] + trees[k][i] + B[k, i]
                     for k in range(n)]
            B[j, i - 1] = logsumexp(probs)

    likelihood_b = logsumexp([p + i + e[0] for p, i, e in zip(B[:, 0], init_probs, trees)])

    # Calculate posterior probabilities
    pzi_x = np.array([logsumexp([F[j, i] + B[j, i] for j in range(n)]) for i in range(m)])
    for j in range(m):
        R[:, j] = (F[:, j] + B[:, j]) - pzi_x[j]

    return F, likelihood_f, B, likelihood_b, np.exp(R)


def normalize(arr):
    """
    return normalized array
    """
    if arr.ndim <= 1:
        return arr - logsumexp(arr)
    return arr - (logsumexp(arr, axis=1)[:, None])


def getProb(params, nstates, seqlen, trees):
    """
    Compute the posterior probability of seq given an HMM with
    nstates states and parameters params
    :param params: 1d array of parameters
    :param trees: numpy array of trees
    :param nstates: number of states in HMM
    :param seqlen: length of sequence
    :return: prob of seq given params
    """
    init, trans = unfold_params(params, nstates=nstates)
    p = 0
    return -forward(seqlen, normalize(trans), normalize(init), trees)


def optimize(seqlen, trees, nstates=2):
    """
    Find the parameters of a hidden markov model with nstates
    states given observation seq
    :param trees: numpy array of trees
    :param seqlen: length of
    :param nstates: number of states
    :return: tuple of initial, emission, and transition probabilities
    that define the HMM
    """
    num_params = nstates * (nstates + 1)
    guess = np.log(np.random.rand(num_params))
    # guess = np.log(np.array([0.5, 0.5, 0.13, 0.37, 0.37, 0.13,
    #                          0.32, 0.18, 0.18, 0.32, 0.95, 0.05, 0.05, 0.95]))
    i, t = unfold_params(guess, nstates=nstates)
    i = np.log(np.ones(nstates) / nstates)
    i, t = normalize(i), normalize(t)

    i = np.log(np.array([0.5, 0.5]))
    # t = np.log(np.array([
    #     [1 - 0.01, 0.01],
    #     [0.01, 1 - 0.01]
    # ]))
    t = np.log(np.array([
        [0.5, 0.5],
        [0.5, 0.5]
    ]))

    guess = np.concatenate((i, np.ndarray.flatten(t)))
    res = minimize(getProb, guess, args=(nstates, seqlen, trees), method="BFGS",
                   options={"maxiter": 250, "disp": True})
    # bounds=([(np.NINF, 0)] * num_params))
    i, t = unfold_params(res.x, nstates=nstates)
    i, t = normalize(i), normalize(t)
    print(forward(seqlen, t, i, trees))
    return np.exp(i), np.exp(t)


def saveplot(probs, factor):
    """ Helper function to save plot of log likelihoods over iterations to file for
        visualization.
    Arguments:
        probs: probability of each state at each iteration
        factor: The scaling factor for branch lengths
    Outputs:
        plot of log likelihoods to file
    """
    plt.title("Probability vs ind")
    plt.xlabel("Index")
    plt.ylabel("Probability of state")
    n, m = probs.shape
    plt.plot(range(m), probs[0, :], 'r-')
    plt.plot(range(m), probs[1, :], 'b-')
    plt.savefig("phylohmm%.2f.png" % factor)


def main():
    parser = argparse.ArgumentParser(
        description='Compute posterior probabilities at each position of a given sequence.')
    parser.add_argument('-f', action="store", dest="f", type=str, default='apoe.fa')
    # parser.add_argument('-mu', action="store", dest="mu", type=float, required=True)
    parser.add_argument('-c', action='store', dest='c', type=str, required=True)
    parser.add_argument('-mul', action='store', dest='mul', type=float, default=1.5)
    parser.add_argument('-nc', action='store', dest='nc', type=str, default=None)

    args = parser.parse_args()
    data, data_len = read_data(args.f)
    cons = Node.from_str(args.c)
    cons.setData(data, data_len)
    if not args.nc:
        non_cons = cons * args.mul
    else:
        non_cons = Node.from_str(args.nc)
        non_cons.setData(data, data_len)

    seqs = list(map(read_fasta, [
        "hmm-sequence.fa",
        "test.fa"
    ]))
    trees = np.array([cons, non_cons])

    init, trans = optimize(data_len, trees)
    print(init)
    print(trans)
    F, l_f, B, l_b, R = forward_backward(data_len, trans, init, trees)
    saveplot(R, args.mul)


if __name__ == "__main__":
    main()

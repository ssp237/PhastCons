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
from scipy.interpolate import make_interp_spline, BSpline
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
    :return: init and trans probs as well as scale factor
    """
    init = params[:nstates]
    # emiss = params[nstates:nstates * 5].reshape((nstates, 4))
    trans = params[nstates:(nstates * (nstates + 1))].reshape((nstates, nstates))
    scale = params[-1]
    return init, trans, scale


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
    init, trans, scale = unfold_params(params, nstates=nstates)
    p = 0
    if scale == 0:
        return np.NINF
    new_trees = trees * scale
    return -forward(seqlen, normalize(trans), normalize(init), new_trees)


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
    i, t, s = unfold_params(guess, nstates=nstates)
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
    s = 1.0 / trees[0].tot_branch_len()

    guess = np.concatenate((i, np.ndarray.flatten(t), [1]))
    res = minimize(getProb, guess, args=(nstates, seqlen, trees), method="L-BFGS-B",
                   options={"maxiter": 250, "disp": True},
                   bounds=(([(np.NINF, 0)] * num_params) + [(0, np.Inf)]))
    i, t, s = unfold_params(res.x, nstates=nstates)
    i, t = normalize(i), normalize(t)
    print(forward(seqlen, t, i, (trees * s)))
    return np.exp(i), np.exp(t), s


def saveplot(probs, factor, smooth=False):
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
    if smooth:

        xnew = np.linspace(0, m, 300)
        spl_cons = make_interp_spline(range(m), probs[0, :], k=3)
        cons_smooth = spl_cons(xnew)
        spl_ncons = make_interp_spline(range(m), probs[1, :], k=3)
        ncons_smooth = spl_ncons(xnew)

        plt.plot(xnew, cons_smooth, 'r-')
        plt.plot(xnew, ncons_smooth, 'b-')
    else:
        plt.plot(range(m), probs[0, :], 'r-')
        plt.plot(range(m), probs[1, :], 'b-')
    plt.savefig("phylohmm%.2f.png" % factor)


def main():
    primates = "((Gorilla:0.086940, (Orangutan:0.018940, (Gibbon:0.022270, (Green_monkey:0.027000, " \
                "(Baboon:0.008042, (Rhesus:0.004991, " \
                "Crab_eating_macaque:0.004991):0.003000):0.019610):0.022040):0.003471):0.009693):0.000500, " \
                "(Human:0.006550, Chimp:0.006840):0.000500):0.000000;"
    parser = argparse.ArgumentParser(
        description='Compute posterior probabilities at each position of a given sequence.')
    parser.add_argument('-f', action="store", dest="f", type=str, default='apoe.fa')
    # parser.add_argument('-mu', action="store", dest="mu", type=float, required=True)
    parser.add_argument('-c', action='store', dest='c', type=str, default=primates)
    parser.add_argument('-mul', action='store', dest='mul', type=float, default=4.0)
    parser.add_argument('-nc', action='store', dest='nc', type=str, default=None)
    parser.add_argument('-d', action='store', dest='df', type=str, default=None)

    args = parser.parse_args()
    data, data_len = read_data(args.f)
    if args.df:
        t = Node.EM(args.df, data, data_len)
        print(t)
        return

    cons = Node.from_str(args.c)
    # cons = cons * (1 / cons.tot_branch_len())
    print(cons.tot_branch_len())
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

    init, trans, s = optimize(data_len, trees)
    print(init)
    print(trans)
    print(s)
    F, l_f, B, l_b, R = forward_backward(data_len, trans, init, trees * s)
    saveplot(R, args.mul)


if __name__ == "__main__":
    main()

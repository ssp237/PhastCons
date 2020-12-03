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


def forward_backward(obs, trans_probs, emiss_probs, init_probs):
    """ Outputs the forward and backward probabilities of a given observation.
    Arguments:
        obs: observed sequence of emitted states (list of emissions)
        trans_probs: transition log-probabilities (dictionary of dictionaries)
        emiss_probs: emission log-probabilities (dictionary of dictionaries)
        init_probs: initial log-probabilities for each hidden state (dictionary)
    Returns:
        F: matrix of forward probabilities
        likelihood_f: P(obs) calculated using the forward algorithm
        B: matrix of backward probabilities
        likelihood_b: P(obs) calculated using the backward algorithm
        R: matrix of posterior probabilities
    """
    ind_bases = {b: i for i, b in enumerate('ACGT')}
    n, m = init_probs.shape[0], len(obs)
    F = np.zeros((n, m))
    B = np.zeros((n, m))
    R = np.zeros((n, m))

    # forwards
    for i, p in enumerate(init_probs):
        F[i, 0] = p + emiss_probs[i][ind_bases[obs[0]]]

    for i, c in list(enumerate(obs))[1:]:
        ic = ind_bases[c]
        for j in range(n):
            e = emiss_probs[j][ic]
            probs = [F[k, i - 1] + trans_probs[k][j] for k in range(n)]
            F[j, i] = e + logsumexp(probs)

    likelihood_f = logsumexp(F[:, -1])

    # # backwards (note that log(1) = 0, so last col is already set)
    # for i, c in reversed(list(enumerate(obs))[1:]):
    #     for j, s in enumerate(init_probs):
    #         probs = [trans_probs[s][s1] + emiss_probs[s1][c] + B[k, i]
    #                  for k, s1 in enumerate(init_probs)]
    #         B[j, i - 1] = sumLogProbList(probs)
    #
    # likelihood_b = logsumexp([p + v + emiss_probs[k][obs[0]]
    #                           for p, (k, v) in zip(B[:, 0], init_probs.items())])
    #
    # # Calculate posterior probabilities
    # pzi_x = []
    # for i in range(m):
    #     pzi_x.append(logsumexp([F[j, i] + B[j, i] for j in range(n)]))
    #
    # for j in range(m):
    #     for i in range(n):
    #         R[i, j] = (F[i, j] + B[i, j]) - pzi_x[j]

    return F, likelihood_f
    # B, likelihood_b, np.exp(R)


def main():
    parser = argparse.ArgumentParser(
        description='Compute posterior probabilities at each position of a given sequence.')
    parser.add_argument('-f', action="store", dest="f", type=str, required=True)
    parser.add_argument('-mu', action="store", dest="mu", type=float, required=True)

    args = parser.parse_args()
    fasta_file = args.f
    mu = args.mu

    obs_sequence = read_fasta(fasta_file)
    trans_probs = np.log(np.array([
        [1 - mu, mu],
        [mu, 1 - mu]
    ]))
    em_probs = np.log(np.array([
        [0.13, 0.37, 0.37, 0.13],
        [0.32, 0.18, 0.18, 0.32]
    ]))
    init_probs = np.log(np.array([0.5, 0.5]))
    F, likelihood_f = \
        forward_backward(obs_sequence, trans_probs, em_probs, init_probs)
    # np.savetxt("posteriors.csv", R, delimiter=",", fmt='%.4e')
    print("Forward likelihood: {:.8f}".format(likelihood_f))
    # print("Backward likelihood: {:.8f}".format(likelihood_b))


if __name__ == "__main__":
    main()

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


def forward(obs, trans_probs, emiss_probs, init_probs):
    """ Outputs the forward and backward probabilities of a given observation.
    Arguments:
        obs: observed sequence of emitted states (list of emissions)
        trans_probs: transition log-probabilities (dictionary of dictionaries)
        emiss_probs: emission log-probabilities (dictionary of dictionaries)
        init_probs: initial log-probabilities for each hidden state (dictionary)
    Returns:
        F: matrix of forward probabilities
        likelihood_f: P(obs) calculated using the forward algorithm
    """
    ind_bases = {b: i for i, b in enumerate('ACGT')}
    n, m = init_probs.shape[0], len(obs)
    F = np.zeros((n, m))

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

    return likelihood_f


def normalize(arr):
    """
    return normalized array
    """
    if arr.ndim <= 1:
        return arr - logsumexp(arr)
    return arr - (logsumexp(arr, axis=1)[:, None])


def getProb(params, nstates, seq, emiss):
    """
    Compute the posterior probability of seq given an HMM with
    nstates states and parameters params
    :param params: 1d array of parameters
    :param emiss: emission probabilities
    :param nstates: number of states in HMM
    :param seq: Sequence to test
    :return: prob of seq given params
    """
    init, trans = unfold_params(params, nstates=nstates)
    p = 0
    for s in seq:
        p += -forward(s, normalize(trans), normalize(emiss), normalize(init))
    return p


def optimize(seq, nstates=2):
    """
    Find the parameters of a hidden markov model with nstates
    states given observation seq
    :param seq: observation data
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
    e = np.log(np.array([
        [0.13, 0.37, 0.37, 0.13],
        [0.32, 0.18, 0.18, 0.32]
    ]))
    t = np.log(np.array([
        [1 - 0.01, 0.01],
        [0.01, 1 - 0.01]
    ]))

    guess = np.concatenate((i, np.ndarray.flatten(t)))
    res = minimize(getProb, guess, args=(nstates, seq, e), method="BFGS",
                   options={"maxiter": 250, "disp": True})
                   # bounds=([(np.NINF, 0)] * num_params))
    i, t = unfold_params(res.x, nstates=nstates)
    i, t = normalize(i), normalize(t)
    for s in seq:
        print(forward(s, t, e, i))
    return np.exp(i), np.exp(e), np.exp(t)


def main():
    parser = argparse.ArgumentParser(
        description='Compute posterior probabilities at each position of a given sequence.')
    # parser.add_argument('-f', action="store", dest="f", type=str, required=True)
    # parser.add_argument('-mu', action="store", dest="mu", type=float, required=True)

    args = parser.parse_args()
    # fasta_file = args.f
    # mu = args.mu
    mu = 0.05

    # obs_sequence = read_fasta(fasta_file)
    seqs = list(map(read_fasta, [
        "hmm-sequence.fa",
        "test.fa"
    ]))
    trans_probs = np.log(np.array([
        [1 - mu, mu],
        [mu, 1 - mu]
    ]))
    em_probs = np.log(np.array([
        [0.13, 0.37, 0.37, 0.13],
        [0.32, 0.18, 0.18, 0.32]
    ]))
    init_probs = np.log(np.array([0.5, 0.5]))

    init, emiss, trans = optimize(seqs)
    print(init)
    print(emiss)
    print(trans)


if __name__ == "__main__":
    main()

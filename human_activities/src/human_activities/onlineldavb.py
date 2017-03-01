# onlineldavb.py: Package of functions for fitting Latent Dirichlet
# Allocation (LDA) with online variational Bayes (VB).
#
# Copyright (C) 2010  Matthew D. Hoffman
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import sys, re, time, string
import numpy as np
from scipy.special import gammaln, psi
import pdb

np.random.seed(100000001)
meanchangethresh = 0.001

def dirichlet_expectation(alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
        return(psi(alpha) - psi(np.sum(alpha)))
    return(psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis])


class OnlineLDA:
    """
    Implements online VB for LDA as described in (Hoffman et al. 2010).
    """

    def __init__(self, vocab, K, D, alpha, eta, tau0, kappa, updatect, prev_lam=[]):
        """
        Arguments:
        K: Number of topics
        vocab: A set of words to recognize. When analyzing documents, any word
           not in this set will be ignored.
        D: Total number of documents in the populationp. For a fixed corpus,
           this is the size of the corpus. In the truly online setting, this
           can be an estimate of the maximum number of documents that
           could ever be seenp.
        alpha: Hyperparameter for prior on weight vectors theta
        eta: Hyperparameter for prior on topics beta
        tau0: A (positive) learning parameter that downweights early iterations
        kappa: Learning rate: exponential decay rate---should be between
             (0.5, 1.0] to guarantee asymptotic convergence.

        Note that if you pass the same set of D documents in every time and
        set kappa=0 this class can also be used to do batch VB.
        """
        self._vocab = dict()
        for word in vocab:
            #word = word.lower()
            #word = re.sub(r'[^a-z]', '', word)
            self._vocab[word] = len(self._vocab)

        self._K = K
        self._W = len(self._vocab)
        self._D = D
        self._alpha = alpha
        self._eta = eta
        self._tau0 = tau0 + 1
        self._kappa = kappa
        self._updatect = updatect

        #print "lambda: %s, %s" %(type(self._lambda), self._lambda.shape)
        if prev_lam==[]:
            # Initialize the variational distribution q(beta|lambda)
            self._lambda = 1*np.random.gamma(100., 1./100., (self._K, self._W))
        else:
            #print "lam", prev_lam
            self._lambda = np.vstack([l.data for l in prev_lam])
            #print "self.lam", self._lambda

        # print "lambda: %s, %s" %(type(self._lambda), self._lambda.shape)
        # print "rand: ", self._lambda.sum(axis=1)
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = np.exp(self._Elogbeta)
        print "self._K: %s. self._W: %s. _lambda.shape: %s." % (self._K, self._W, self._lambda.shape)

    def add_new_features(self, new_length):
        """If the codebook increases between each run, update lamda for new features
        %todo: Make this 1/w for each column and normalised instead of 0.
        """
        # template:
        # new_lambda = np.ones((self._lambda.shape[0], new_length))/self._lambda.sum(axis=1)
        new_lambda = np.ones((self._lambda.shape[0], new_length))/self._lambda.shape[1]
        diff = new_lambda.shape[1] - self._lambda.shape[1]
        # print "diff: ", diff, self._lambda.sum(axis=1), "\n",    new_lambda

        if diff>0:
            new_lambda[:,:-diff] = self._lambda
            self._lambda = new_lambda
        # else:
            # Don't update lambda at all.
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = np.exp(self._Elogbeta)


    def do_e_step(self, wordids, wordcts):
        batchD = len(wordids)  #num of docs in batch

        # Initialize the variational distribution q(theta|gamma) for the mini-batch
        gamma = 1*np.random.gamma(100., 1./100., (batchD, self._K))
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = np.exp(Elogtheta)

        sstats = np.zeros(self._lambda.shape)
        # Now, for each document d update that document's gamma and phi
        it = 0
        meanchange = 0
        for d in range(0, batchD):
            # print sum(wordcts[d])
            # These are mostly just shorthand (but might help cache locality)
            ids = wordids[d]#[0]
            cts = wordcts[d]#[0]
            gammad = gamma[d, :]
            Elogthetad = Elogtheta[d, :]

            expElogthetad = expElogtheta[d, :]

            expElogbetad = self._expElogbeta[:, ids]
            # The optimal phi_{dwk} is proportional to
            # expElogthetad_k * expElogbetad_w. phinorm is the normalizer.

            phinorm = np.dot(expElogthetad, expElogbetad) + 1e-100
            # Iterate between gamma and phi until convergence
            for it in range(0, 100):
                lastgamma = gammad
                # We represent phi implicitly to save memory and time.
                # Substituting the value of the optimal phi back into
                # the update for gamma gives this update. Cf. Lee&Seung 2001.
                gammad = self._alpha + expElogthetad * \
                    np.dot(cts / phinorm, expElogbetad.T)
                # print gammad[:, np.newaxis]
                Elogthetad = dirichlet_expectation(gammad)
                expElogthetad = np.exp(Elogthetad)
                phinorm = np.dot(expElogthetad, expElogbetad) + 1e-100
                # If gamma hasn't changed much, we're done.
                meanchange = np.mean(abs(gammad - lastgamma))
                if (meanchange < meanchangethresh):
                    break
            gamma[d, :] = gammad
            # Contribution of document d to the expected sufficient
            # statistics for the M step.
            sstats[:, ids] += np.outer(expElogthetad.T, cts/phinorm)

        # This step finishes computing the sufficient statistics for the
        # M step, so that
        # sstats[k, w] = \sum_d n_{dw} * phi_{dwk}
        # = \sum_d n_{dw} * exp{Elogtheta_{dk} + Elogbeta_{kw}} / phinorm_{dw}.
        sstats = sstats * self._expElogbeta

        return((gamma, sstats))

    def update_lambda(self, wordids, wordcts):
        """
        First does an E step on the mini-batch given in wordids and
        wordcts, then uses the result of that E step to update the
        variational parameter matrix lambda.

        Arguments:
        docs:  List of D documents. Each document must be represented
               as a string. (Word order is unimportant.) Any
               words not in the vocabulary will be ignored.

        Returns gamma, the parameters to the variational distribution
        over the topic weights theta for the documents analyzed in this
        update.

        Also returns an estimate of the variational bound for the
        entire corpus for the OLD setting of lambda based on the
        documents passed inp. This can be used as a (possibly very
        noisy) estimate of held-out likelihood.
        """

        # rhot will be between 0 and 1, and says how much to weight
        # the information we got from this mini-batch.
        rhot = pow(self._tau0 + self._updatect, -self._kappa)
        self._rhot = rhot
        # Do an E step to update gamma, phi | lambda for this
        # mini-batch. This also returns the information about phi that
        # we need to update lambda.
        (gamma, sstats) = self.do_e_step(wordids, wordcts)
        # Estimate held-out likelihood for current values of lambda.
        bound = self.approx_bound(wordids, wordcts, gamma)
        # Update lambda based on documents.
        self._lambda = self._lambda * (1-rhot) + rhot * (self._eta + self._D * sstats / len(wordids))
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = np.exp(self._Elogbeta)
        self._updatect += 1

        return(gamma, bound)

    def approx_bound(self, wordids, wordcts, gamma):
        """
        Estimates the variational bound over *all documents* using only
        the documents passed in as "docs." gamma is the set of parameters
        to the variational distribution q(theta) corresponding to the
        set of documents passed inp.

        The output of this function is going to be noisy, but can be
        useful for assessing convergence.
        """

        # This is to handle the case where someone just hands us a single
        # document, not in a list.
        batchD = len(wordids)

        score = 0
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = np.exp(Elogtheta)

        # E[log p(docs | theta, beta)]
        for d in range(0, batchD):
            gammad = gamma[d, :]
            ids = wordids[d]#[0]
            cts = np.array(wordcts[d])
            phinorm = np.zeros(len(ids))
            for i in range(0, len(ids)):

                temp = Elogtheta[d, :] + self._Elogbeta[:, ids[i]]
                tmax = max(temp)
                phinorm[i] = np.log(sum(np.exp(temp - tmax))) + tmax
            score += np.sum(cts * phinorm)
#             oldphinorm = phinorm
#             phinorm = np.dot(expElogtheta[d, :], self._expElogbeta[:, ids])
#             print oldphinorm
#             print np.log(phinorm)
#             score += np.sum(cts * np.log(phinorm))

        # E[log p(theta | alpha) - log q(theta | gamma)]
        score += np.sum((self._alpha - gamma)*Elogtheta)
        score += np.sum(gammaln(gamma) - gammaln(self._alpha))
        score += sum(gammaln(self._alpha*self._K) - gammaln(np.sum(gamma, 1)))

        # Compensate for the subsampling of the population of documents
        score = score * self._D / len(wordids)

        # E[log p(beta | eta) - log q (beta | lambda)]
        score = score + np.sum((self._eta-self._lambda)*self._Elogbeta)
        score = score + np.sum(gammaln(self._lambda) - gammaln(self._eta))
        score = score + np.sum(gammaln(self._eta*self._W) -
                              gammaln(np.sum(self._lambda, 1)))

        return(score)

# def main():
#     infile = sys.argv[1]
#     K = int(sys.argv[2])
#     alpha = float(sys.argv[3])
#     eta = float(sys.argv[4])
#     kappa = float(sys.argv[5])
#     S = int(sys.argv[6])
#
#     docs = corpus.corpus()
#     docs.read_data(infile)
#
#     vocab = open(sys.argv[7]).readlines()
#     model = OnlineLDA(vocab, K, 100000,
#                       0.1, 0.01, 1, 0.75)
#     for i in range(1000):
#         print i
#         wordids = [d.words for d in docs.docs[(i*S):((i+1)*S)]]
#         wordcts = [d.counts for d in docs.docs[(i*S):((i+1)*S)]]
#         model.update_lambda(wordids, wordcts)
#         np.savetxt('/tmp/lambda%d' % i, model._lambda.T)


if __name__ == '__main__':
    main()

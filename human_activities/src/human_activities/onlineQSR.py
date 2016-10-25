import cPickle, string, numpy, getopt, sys, random, time, re, pprint

import onlineldavb
import wikirandom

def main():
    """
    Test function for Online LDA using Variational Bayes
    """

    # The number of documents to analyze each iteration
    batchsize = 4

    # The total number of documents (or an estimate of all docs)
    D = 16

    # The number of topics
    K = 3

    # How many documents to look at
    if (len(sys.argv) < 2):
        num_iters = int(D/batchsize)
    else:
        num_iters = int(sys.argv[1])

    # Our vocabulary
    #vocab = file('./dictnostops.txt').readlines()
    #W = len(vocab)

    print "num_iters: %s " %num_iters

    QSR_vectors=cPickle.load(open("Data/feature_space.p","rb"))
    QSR_codebook=cPickle.load(open("Data/code_book.p","rb"))
    codebook_len=len(QSR_codebook)

    # Initialize the algorithm with alpha=1/K, eta=1/K, tau_0=1024, kappa=0.7
    olda = onlineldavb.OnlineLDA(QSR_codebook, K, D, 1./K, 1./K, 1., 0.7)

    # Run until we've seen D documents.
    for iteration in range(0, num_iters):
        print "it: %s. start: %s. end: %s" % (iteration, iteration*batchsize, (iteration+1)*batchsize)
        # Download some articles
        #(docset, articlenames) = wikirandom.get_random_wikipedia_articles(batchsize)
        # Give them to online LDA
        docset=QSR_vectors[iteration*batchsize:(iteration+1)*batchsize]

        print "size of docset: %s" %len(docset)

        wordids=[]
        wordcts=[]

        for cnt, v in enumerate(docset):
            print "\n cnt: ", cnt

            nonzeros=numpy.nonzero(v)
            available_features=nonzeros

            wordids.append(available_features)
            feature_counts=v[nonzeros]
            wordcts.append(feature_counts)

            print "v ", v
            print "avail features %s, feature_cnts: %s" %(available_features, feature_counts)

        print "wordids %s, wordcts: %s" %(wordids, wordcts)

        (gamma, bound) = olda.update_lambda(wordids,wordcts)
        # Compute an estimate of held-out perplexity

        perwordbound = bound * len(docset) / (D * sum(map(sum, wordcts)))
        print '%d:  rho_t = %f,  held-out perplexity estimate = %f' % \
            (iteration, olda._rhot, numpy.exp(-perwordbound))

        # Save lambda, the parameters to the variational distributions
        # over topics, and gamma, the parameters to the variational
        # distributions over topic weights for the articles analyzed in
        # the last iteration.
        if (iteration % 1 == 0):
            numpy.savetxt('Data/lambda-%d.dat' % iteration, olda._lambda)
            numpy.savetxt('Data/gamma-%d.dat' % iteration, gamma)

if __name__ == '__main__':
    main()

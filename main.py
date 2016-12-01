#!/usr/bin/python
import sys, getopt
from time import time
from optparse import OptionParser

def main(argv):
    # parse commandline arguments
    op = OptionParser()
    op.add_option("-r", "--trainfile", dest="trainfile",
                  default="data/train.json",
                  help="Training file "
                       "[default: %default]")
    op.add_option("-s", "--testfile", dest="testfile",
                  default="data/test.json",
                  help="Testing file "
                       "[default: %default]")
    op.add_option("-o", "--outputfile", dest="outputfile",
                  default="submission.csv",
                  help="Output file "
                       "[default: %default]")
    op.add_option("--use_hashing",
                  action="store_true",
                  help="Use a hashing vectorizer instead of Tf-idf vectorizer. "
                       "[defualt: False]")
    op.add_option("--n_features",
                  action="store", type=int, default=5000,
                  help="n_features when using the hashing vectorizer. "
                       "[default: 5000]")
    op.add_option("-b", "--bag_of_words",
                  action="store_true", dest="bag_of_words",
                  help="Use the bag of words model to vectorize features. "
                       "[default: False]")
    (opts, args) = op.parse_args()
    if len(args) > 0:
        op.error("this script takes no arguments.")
        sys.exit(1)

    print(__doc__)
    op.print_help()

    # usage = 'usage: classification.py [-b] [-i| <trainfile>] [-t <testfile>] [-o <outputfile>]'
    trainfile = opts.trainfile
    testfile = opts.testfile
    outputfile = opts.outputfile
    useBagOfWords = opts.bag_of_words
    n_features = opts.n_features

    # Plug in algorithm here
#    from sklearn.naive_bayes import GaussianNB # 34.443%
#    clf = GaussianNB()

#    from sklearn.naive_bayes import MultinomialNB # 73.914%
#    clf = MultinomialNB()

#    from sklearn.tree import DecisionTreeClassifier
#    clf = DecisionTreeClassifier() # using gini 61.9%
#    clf = DecisionTreeClassifier(criterion='entropy') # 56.4%

    from sklearn.linear_model import LogisticRegression # 78.329%
    clf = LogisticRegression()

#    from sklearn.neighbors import KNeighborsClassifier
#    clf = KNeighborsClassifier()

    if useBagOfWords:
        from classification_bow import ClassificationBagOfWords
        executor = ClassificationBagOfWords(trainfile, testfile, outputfile, clf)
    else:
        from classification_plain import ClassificationPlain
        executor = ClassificationPlain(trainfile, testfile, outputfile, clf)
    print 'Starting preprocessing'
    startTime = time()
    executor.preprocess()
    print 'Preprocessing finished in {:f}s'.format(time() - startTime)

    print 'Starting training'
    startTime = time()
    executor.train()
    print 'Training finished in {:f}s'.format(time() - startTime)

    print 'Starting predicting'
    startTime = time()
    executor.predict()
    print 'Prediction finished in {:f}s'.format(time() - startTime)

    executor.output()
    print 'Output to ' + outputfile

if __name__ == '__main__':
    main(sys.argv[1:])

#!/usr/bin/python
import sys
from time import time
from optparse import OptionParser

def main(argv):
    # parse commandline arguments
    op = OptionParser()
    op.add_option("-i", "--trainfile", dest="trainfile",
                  default="data/train.json",
                  help="Training file "
                       "[default: %default]")
    op.add_option("-t", "--testfile", dest="testfile",
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
                       "[default: False]")
    op.add_option("--n_features",
                  action="store", type=int, default=5000,
                  help="n_features when using the hashing vectorizer. "
                       "[default: 5000]")
    op.add_option("-b", "--bag_of_words",
                  action="store_true", dest="bag_of_words",
                  help="Use the bag of words model to vectorize features. "
                       "[default: False]")
    opts, args = op.parse_args()

    trainfile = opts.trainfile
    testfile = opts.testfile
    outputfile = opts.outputfile
    use_bag_of_words = opts.bag_of_words
    
    if len(args) > 1:
        print 'Too many arguments, only 1 needed\n'
        op.print_help()
        sys.exit(2)
    if len(args) == 0 or args[0] == 'MNB':
        from sklearn.naive_bayes import MultinomialNB # 73.914%
        clf = MultinomialNB()
    elif args[0] == 'GNB':
        from sklearn.naive_bayes import GaussianNB # 34.443%
        clf = GaussianNB()
    elif args[0] == 'GiniDT':
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier() # using gini 61.9%
    elif args[0] == 'InfoDT':
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(criterion='entropy') # 56.4%
    elif args[0] == 'LR':
        from sklearn.linear_model import LogisticRegression # 78.329%
        clf = LogisticRegression()
    else:
        print 'Train model {} not recognized'.format(args[0])
        sys.exit(2)

#    from sklearn.neighbors import KNeighborsClassifier
#    clf = KNeighborsClassifier()

    if use_bag_of_words:
        from classification_bow import ClassificationBagOfWords
        executor = ClassificationBagOfWords(trainfile, testfile, outputfile, clf, opts.use_hashing, opts.n_features)
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

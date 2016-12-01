#!/usr/bin/python
import sys
from time import time
import argparse

def main(argv):
    # parse commandline arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', '--trainfile', dest='trainfile',
                        default='data/train.json',
                        help='training file\n(default: %(default)s)')
    parser.add_argument('-t', '--testfile', dest='testfile',
                        default='data/test.json',
                        help='testing file\n(default: %(default)s)')
    parser.add_argument('-o', '--outputfile', dest='outputfile',
                        default='submission.csv',
                        help='output file\n(default: %(default)s)')
    parser.add_argument('--use_hashing', action='store_true',
                        help='use a hashing vectorizer instead of Tf-idf vectorizer\n(default: %(default)s)')
    parser.add_argument('--n_features', action='store',
                        type=int, default=5000,
                        help='n_features when using the hashing vectorizer\n(default: %(default)d)')
    parser.add_argument('-b', '--bag_of_words', action='store_true',
                        help='use the bag of words model to vectorize features\n(default: %(default)s)')
    parser.add_argument('model', nargs='?',
                        default='MNB',
                        help='model for training and predicting\n'
                        'GNB: Gaussian Naive Bayes\n'
                        'MNB: Multinomial Naive Bayes\n'
                        'GiniDT: Decision Tree with Gini Index\n'
                        'InfoDT: Decision Tree with Information Gain\n'
                        'LR: Logistic Regression\n(default: %(default)s)')
    args = parser.parse_args()
    
    trainfile = args.trainfile
    testfile = args.testfile
    outputfile = args.outputfile
    model = args.model
    
    if model == 'MNB':
        from sklearn.naive_bayes import MultinomialNB # 73.914%
        clf = MultinomialNB()
    elif model == 'GNB':
        from sklearn.naive_bayes import GaussianNB # 34.443%
        clf = GaussianNB()
    elif model == 'GiniDT':
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier() # using gini 61.9%
    elif model == 'InfoDT':
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(criterion='entropy') # 56.4%
    elif model == 'LR':
        from sklearn.linear_model import LogisticRegression # 78.329%
        clf = LogisticRegression()
    else:
        print 'Train model {} not recognized'.format(model)
        sys.exit(2)

#    from sklearn.neighbors import KNeighborsClassifier
#    clf = KNeighborsClassifier()

    if args.bag_of_words:
        from classification_bow import ClassificationBagOfWords
        executor = ClassificationBagOfWords(trainfile, testfile, outputfile, clf, args.use_hashing, args.n_features)
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

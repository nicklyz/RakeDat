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
    parser.add_argument('--max_iter', action='store',
                        type=int, default=100,
                        help='max number of iterations when using SVMs')
    parser.add_argument('model', nargs='?',
                        default='MNB',
                        help='model for training and predicting\n'
                        ' GNB: Gaussian Naive Bayes\n'
                        ' MNB: Multinomial Naive Bayes\n'
                        ' GiniDT: Decision Tree with Gini Index\n'
                        ' InfoDT: Decision Tree with Information Gain\n'
                        ' SVM: C Support Vector Machine\n'
                        ' LinearSVM: Linear Support Vector Machine\n'
                        ' LR: Logistic Regression\n(default: %(default)s)')
    args = parser.parse_args()
    
    trainfile = args.trainfile
    testfile = args.testfile
    outputfile = args.outputfile
    model = args.model
    
    if model == 'MNB':
        from sklearn.naive_bayes import MultinomialNB # 0.73914
        clf = MultinomialNB()
    elif model == 'GNB':
        from sklearn.naive_bayes import GaussianNB # 0.34443
        clf = GaussianNB()
    elif model == 'GiniDT':
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier() # using gini 0.61947; 0.62611 -b
    elif model == 'InfoDT':
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(criterion='entropy') # 56.4%
    elif model == 'SVM':
        from sklearn.svm import SVC # 0.50975 -b max_iter=20
        clf = SVC(max_iter=args.max_iter)
    elif model == 'LinearSVM':
        from sklearn.svm import LinearSVC # 0.77293; 0.78681 -b
        clf = LinearSVC(max_iter=args.max_iter)
    elif model == 'LR':
        from sklearn.linear_model import LogisticRegression # 0.78329
        clf = LogisticRegression()
    else:
        print 'Train model {} not recognized'.format(model)
        sys.exit(2)

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
    
    print 'Calculating accuracy on training dataset'
    print 'Accuracy on training dataset: {:f}'.format(executor.accuracyOnTrain())

    print 'Starting predicting'
    startTime = time()
    executor.predict()
    print 'Prediction finished in {:f}s'.format(time() - startTime)

    executor.output()
    print 'Output to ' + outputfile

if __name__ == '__main__':
    main(sys.argv[1:])

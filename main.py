#!/usr/bin/python
import sys, getopt
from time import time

def main(argv):
    usage = 'usage: classification.py [-b] [-i| <trainfile>] [-t <testfile>] [-o <outputfile>]'
    trainfile = 'data/train.json'
    testfile = 'data/test.json'
    outputfile = 'submission.csv'
    useBagOfWords = False
    try:
        opts, args = getopt.getopt(argv, 'hbito', ['help', 'train=', 'test=', 'output='])
    except getopt.GetoptError:
        print usage
        sys.exit(2)
    for o, a in opts:
        if o in ('-h', '--help'):
            print usage
            sys.exit()
        elif o == '-b':
            useBagOfWords = True
        elif o in ('-i', '--train'):
            trainfile = a
        elif o in ('-t', '--test'):
            testfile = a
        elif o in ('-o', '--output'):
            outputfile = a

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
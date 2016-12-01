class Classification(object):
    def __init__(self, trainFile, testFile, outputFile, clf):
        self.trainFile = trainFile
        self.testFile = testFile
        self.outputFile = outputFile
        self.clf = clf
    def preprocess(self):
        raise NotImplementedError
    def train(self):
        raise NotImplementedError
    def predict(self):
        raise NotImplementedError
    def output(self):
        raise NotImplementedError

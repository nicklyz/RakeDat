class Classification:
    def __init__(self, trainFile, testFile, outputFile, clf, use_hashing, n_features):
        self.trainFile = trainFile
        self.testFile = testFile
        self.outputFile = outputFile
        self.clf = clf
        self.use_hashing = use_hashing
        self.n_features = n_features
    def preprocess(self):
        raise NotImplementedError
    def train(self):
        raise NotImplementedError
    def predict(self):
        raise NotImplementedError
    def output(self):
        raise NotImplementedError

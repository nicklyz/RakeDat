from classification import Classification
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

class ClassificationBagOfWords(Classification):
    def __init__(self, trainFile, testFile, outputFile, clf, use_hashing, n_features):
        super(Classification, self).__init__(self, trainFile, testFile, outputFile, clf)
        self.use_hashing = use_hashing
        self.n_features = n_features

    def preprocess(self):
        # pandas to load data
        self.traindf = pd.read_json(self.trainFile)
        self.testdf = pd.read_json(self.testFile)

        # use scikit-learn's bag of words tool
        if self.use_hashing:
            vectorizer = HashingVectorizer(analyzer='word', stop_words=None, n_features=self.n_features)
        else:
            vectorizer = TfidfVectorizer(sublinear_tf=False, max_df=0.5,
                binary = False, analyzer='word', stop_words='english')

        print "Creating the bag of words features for training data..."
        self.traindf['ingredients_clean_string'] = [' '.join(ingred).strip().lower() for ingred in self.traindf['ingredients']]
        self.train_data_features = vectorizer.fit_transform(self.traindf['ingredients_clean_string']).todense()
        print("n_samples: %d, n_features: %d " % self.train_data_features.shape)

        print "Creating the bag of words features for test data..."
        self.testdf['ingredients_clean_string'] = [' '.join(ingred).strip().lower() for ingred in self.testdf['ingredients']]
        self.test_data_features = vectorizer.transform(self.testdf['ingredients_clean_string']).todense()
        print("n_samples: %d, n_features: %d " % self.test_data_features.shape)

    def train(self):
        self.clf.fit(self.train_data_features, self.traindf['cuisine'])

    def predict(self):
        self.result = self.clf.predict(self.test_data_features)

    def output(self):
        output = pd.DataFrame(data={'id': self.testdf['id'], 'cuisine': self.result})
        output.to_csv(self.outputFile, index=False, quoting=3)

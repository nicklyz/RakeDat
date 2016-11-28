import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from time import time

"""
An example of a dish
{
"id": 47028,
"cuisine": "japanese",
"ingredients": [
  "melted butter",
  "matcha green tea powder",
  "white sugar",
  "milk",
  "all-purpose flour",
  "eggs",
  "salt",
  "baking powder",
  "chopped walnuts"
  ]
}
"""
# pandas to load data
traindf = pd.read_json('data/train.json')
testdf = pd.read_json('data/test.json')

# use scikit-learn's bag of words tool
print("Extracting features from the training data using a sparse vectorizer")
vectorizer = HashingVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             n_features = 5000)

print "Creating the bag of words features for training data..."
t0 = time()
traindf['ingredients_clean_string'] = [' '.join(ingred).strip().lower() for ingred in traindf['ingredients']]
train_data_features = vectorizer.fit_transform(traindf['ingredients_clean_string']).todense()
duration = time() - t0
print("done in %fs" % duration)
print("n_samples: %d, n_features: %d " % train_data_features.shape)

print "Creating the bag of words features for test data..."
t0 = time()
testdf['ingredients_clean_string'] = [' '.join(ingred).strip().lower() for ingred in testdf['ingredients']]
test_data_features = vectorizer.transform(testdf['ingredients_clean_string']).todense()
duration = time() - t0
print("done in %fs" % duration)
print("n_samples: %d, n_features: %d " % test_data_features.shape)

# Plug in algorithm here
#from sklearn.naive_bayes import GaussianNB # 34.4%
#clf = GaussianNB()
# from sklearn.tree import DecisionTreeClassifier # 63.053%
# clf = DecisionTreeClassifier()
# from sklearn.svm import SVC
# clf = SVC()
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
print 'Starting training'
t0 = time()
clf.fit(train_data_features, traindf['cuisine'])
duration = time() - t0
print("done in %fs" % duration)

print 'Starting predicting'
t0 = time()
result = clf.predict(test_data_features)
duration = time() - t0
print("done in %fs" % duration)

print 'Outputing result'
output = pd.DataFrame( data={"id":testdf["id"], "cuisine":result} )
output.to_csv( "data/submission.csv", index=False, quoting=3 )

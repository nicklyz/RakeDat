import json
import pandas as pd
import numpy as np

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
print "Creating the bag of words features...\n"
traindf['ingredients_clean_string'] = [' '.join(ingred).strip().lower() for ingred in traindf['ingredients']]

from sklearn.feature_extraction.text import CountVectorizer
# use scikit-learn's bag of words tool
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)
# this create a sparse matrix with shape(39774, 3010)
train_data_features = vectorizer.fit_transform(traindf['ingredients_clean_string'])
train_data_features = train_data_features.toarray()

# preprocess data
with open('data/train.json') as f:
    trainData = json.load(f)
    allCuisines = {} # map from cuisine name to number
    allCuisinesList = []
    allIngredients = {} # map from ingredient name to number
    cntIngr = 0 # counter for ingredients
    for dish in trainData:
        # processing cuisine
        cuisine = dish['cuisine']
        if cuisine not in allCuisines:
            allCuisines[cuisine] = len(allCuisinesList)
            allCuisinesList.append(cuisine)
        dish['cuisine'] = allCuisines[cuisine]
        # processing ingredients
        ingredients = dish['ingredients']
        for idx, ingr in enumerate(ingredients):
            if ingr not in allIngredients:
                allIngredients[ingr] = cntIngr
                cntIngr += 1
            ingredients[idx] = allIngredients[ingr]
    trainDataMatrix = []
    for dish in trainData:
        row = [0] * cntIngr
        for ingr in dish['ingredients']:
            row[ingr] = 1
        trainDataMatrix.append(row)

with open('data/test.json') as f:
    testData = json.load(f)
    testDataMatrix = []
    for dish in testData:
        ingredients = [ingr for ingr in dish['ingredients'] if ingr in allIngredients]
        row = [0] * cntIngr
        for ingr in ingredients:
            row[allIngredients[ingr]] = 1
        testDataMatrix.append(row)

# Plug in algorithm here
#from sklearn.naive_bayes import GaussianNB # 34.4%
#clf = GaussianNB()
#from sklearn.tree import DecisionTreeClassifier # 61.9%
#clf = DecisionTreeClassifier()
from sklearn.svm import SVC
clf = SVC()
print 'Starting training'
clf.fit(train_data_features, traindf['cuisine'])
print 'Starting predicting'
result = map(lambda i: allCuisinesList[i], clf.predict(testDataMatrix))

# Output in csv for submission on Kaggle
import csv
with open('submission.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(('id', 'cuisine'))
    for i, ingr in zip([dish['id'] for dish in testData], result):
        writer.writerow((i, ingr))

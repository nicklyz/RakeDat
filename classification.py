import json

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
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(trainDataMatrix, [dish['cuisine'] for dish in trainData])
result = map(lambda i: allCuisinesList[i], clf.predict(testDataMatrix))

# Output in csv for submission on Kaggle
import csv
with open('submission.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(('id', 'cuisine'))
    for i, ingr in zip([dish['id'] for dish in testData], result):
        writer.writerow((i, ingr))
    
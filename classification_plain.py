from classification import Classification
import json
import csv

class ClassificationPlain(Classification):
    allCuisinesList = []
    trainDataCuisines = [] # all mapped cuisines in train data set
    trainDataMatrix = []
    testDataIds = [] # all IDs in test data set
    testDataMatrix = []
    
    def preprocess(self):
        with open(self.trainFile) as f:
            trainData = json.load(f)
            allCuisines = {} # map from cuisine name to number
            allIngredients = {} # map from ingredient name to number
            cntIngr = 0 # counter for ingredients
            for dish in trainData:
                # processing cuisine
                cuisine = dish['cuisine']
                if cuisine not in allCuisines:
                    allCuisines[cuisine] = len(self.allCuisinesList)
                    self.allCuisinesList.append(cuisine)
                self.trainDataCuisines.append(allCuisines[cuisine])
                # processing ingredients
                ingredients = dish['ingredients']
                for idx, ingr in enumerate(ingredients):
                    if ingr not in allIngredients:
                        allIngredients[ingr] = cntIngr
                        cntIngr += 1
                    ingredients[idx] = allIngredients[ingr]
            for dish in trainData:
                row = [0] * cntIngr
                for ingr in dish['ingredients']:
                    row[ingr] = 1
                self.trainDataMatrix.append(row)
        with open(self.testFile) as f:
            testData = json.load(f)
            for dish in testData:
                self.testDataIds.append(dish['id'])
                ingredients = [ingr for ingr in dish['ingredients'] if ingr in allIngredients]
                row = [0] * cntIngr
                for ingr in ingredients:
                    row[allIngredients[ingr]] = 1
                self.testDataMatrix.append(row)
    
    def train(self):
        self.clf.fit(self.trainDataMatrix, self.trainDataCuisines)
    
    def predict(self):
        self.result = map(lambda i: self.allCuisinesList[i], self.clf.predict(self.testDataMatrix))
    
    def output(self):
        with open(self.outputFile, 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(('id', 'cuisine'))
            for i, ingr in zip(self.testDataIds, self.result):
                writer.writerow((i, ingr))
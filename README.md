# RakeDat
Team RakeDat for CS145 Data Mining

Instruction for running:
You need to have Python 2.7, Numpy, Pandas, and Scikit-learn package installed on your computer before proceeding. It can be run either with command line or directly in Python 2.7.
You should go under the directory  RakeDat , and type in the following: $chmod +x main.py
$./main.py
Or alternatively,
$python main.py
By default, typing the above command will run the simple preprocessing (see section “Data Preprocessing” option 1) for the datasets, and apply the Multinomial Naive Bayesian model. The default output file will be  submission.csv  under the same directory.
You can choose the other model available for data processing - bag of words - by using option  -b . All models available for training are (argument for command line in brackets): 1)
RakeDat 2
Gaussian Naive Bayes [ GNB ], 2) Multinomial Naive Bayes [ MNB ] which is the default if argument left empty, 3) Decision Tree with Gini Index [ GiniDT ], 4) Decision Tree with Information Gain [ InfoDT ], and 5) Logistic Regression [ LR ]. So if you wish to run with the bag-of-words model and Logistic Regression, you can type:
$./main.py -b LR
The training dataset is default to  data/train.json  and test dataset to
data/test.json . You can always choose the training and test datasets as well as the output file. For details, see the help doc:
$./main.py -h

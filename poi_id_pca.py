#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import pprint
pp = pprint.PrettyPrinter(indent=4)
from time import time

from sklearn import linear_model
from feature_format import featureFormat, targetFeatureSplit

from sklearn.cross_validation import train_test_split
from sklearn.decomposition import RandomizedPCA
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from tester import dump_classifier_and_data

### Load the dictionary containing the original dataset
""" Data was enhanced beyond what was done in class.  
Added addtioanal data points representing the count of weasele words in users
sent e-mail.
Added additional people - namely those who sent email but were not in the orignal 
data list.
This was done via 3 python routines run before this and could be used in future similar circumstances:
    recreateMissing.pkl - searched all e-mails pulling out and sorting e-mails by sender
    vectorizeWeaselWords2.py - bag of words approach but limited to a list of 
        chosen typical business "weasel words".  this was done namely to speed the process.
    mergeUserdata.py - match users and weasel counts into the original data and get rid of
        users with no Weaselword i.e. Sent data.
    scaleData - scales the data to make fitting a model using routines that compare
        data points to ech other more effective.
"""
data_dict = pickle.load(open("final_project_dataset_woWeasel_scaled.pkl", "r") )
#pp.pprint(data_dict.keys())
#pp.pprint(data_dict['neal'])
#pp.pprint(data_dict['LAY KENNETH L'])


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',        'bonus',
                              'deferral_payments',
                              'deferred_income',
                              'director_fees',
                              'exercised_stock_options',
                              'expenses',
                              'from_messages',
                              'from_poi_to_this_person',
                              'from_this_person_to_poi',
                              'loan_advances',
                              'long_term_incentive',
                              'other',
                              'restricted_stock',
                              'restricted_stock_deferred',
                              'salary',
                              'shared_receipt_with_poi',
                              'to_messages',
                              'total_payments',
                              'total_stock_value'
                   ]
#also append the new weasel keys
import csv
with open('weaselWords.csv', 'rb') as f:
    reader = csv.reader(f)
    weaselList = list(reader)
for weaselWord in weaselList:
    features_list.append(str(weaselWord[0]))

### Task 2: Remove outliers
# remove haedicke as his weasel data is among highest as an internal counsel and yet he is not a POI, noise to our algorithm
data_dict.pop('haedicke')
#remove weasel key "kill" as it occurs in all e-mails sent to Skilling, not a generalized indicator fro this training set of Enron
features_list.remove('kill')
#remove any POI.. feature as it uses POI and that won't exist in future scenrios and is in fact the label we are trying to predict
features_list.remove('from_poi_to_this_person')
features_list.remove('shared_receipt_with_poi')
#remove Total Payments and Load Advances which in the Enron data are 
#skewed to 1 individual and do not seem generalizable to other companies.
#features_list.remove('total_payments')
features_list.remove('loan_advances')

### Task 3: Create new feature(s)
#note comments at the start of the code listing additional features added as well as scaling which in essence added new features 
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
#data = featureFormat(my_dataset, features_list, sort_keys = True)
data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)

# the label to predict is the id of the person
#y = labels
#X = features

target_names = ['not a POI','this is a POI']
n_classes = 2
n_features = len(features[0])
n_samples = len(features) 

print "Total dataset size:"
print "n_samples: %d" % n_samples
print "n_features: %d" % n_features
print "n_classes: %d" % n_classes

###############################################################################
# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=42)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
###############################################################################
# Compute a PCA on the POI dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
# searching for the optimal # of PCA components to reduce to
# original trial on line below and discovered 2 and 3 components were tied for
# precision and recall  
#n_components = [1, 2, 3, 5, 10]
n_components = [3]
#n_components = [1,5, 10]

for nComp in n_components:
    print "Extracting the top %d PCA component(s) from %d people's data (training data)" % (nComp, len(X_train))
    t0 = time()
    pca = RandomizedPCA(n_components=nComp, whiten=True).fit(X_train)
    print "done in %0.3fs" % (time() - t0)

    #eigenfaces = pca.components_.reshape((nComp, h, w))

    print "Projecting the input data to principle components for both train and test sets"
    t0 = time()
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print "done in %0.3fs" % (time() - t0)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. 
###############################################################################
# Train a SVM classification model using grid search to indentify the best 
# Kernal, C and gamma

    print "Fitting the classifier to the training set"
    t0 = time()
    param_grid = {
         'C': [1e3, 5e3, 1e4, 5e4, 1e5],
          'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
          'kernel': ['linear','rbf','poly']
          }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto'), param_grid)
    clf = clf.fit(X_train_pca, y_train)
    print "done in %0.3fs" % (time() - t0)
    print "Best estimator found by grid search:"
    print clf.best_estimator_


###############################################################################
# Quantitative evaluation of the model quality on the test set

    print "Predicting the people names on the testing set"
    t0 = time()
    y_pred = clf.predict(X_test_pca)
    print "done in %0.3fs" % (time() - t0)

    print classification_report(y_test, y_pred, target_names=target_names)
    ### calculate and return the accuracy on the test data
    from sklearn.metrics import accuracy_score
    t0 = time()
    accuracy = accuracy_score(y_pred, y_test)
    print "calc accuracy time:", round(time()-t0, 3), "s"
    print "accuraccy was:{}".format (accuracy)

# setup the Classifier based on the findings above as a pipeline to include both
# the PCA and SVC steps so that it can be appropriately saved and re-used:
from sklearn.pipeline import Pipeline
print "Using Pipeline -- Predicting the people names on the testing set"
estimators = [('reduce_dim', RandomizedPCA(n_components=3, whiten=True)), 
                     ('svm', SVC(C=1000.0, gamma=0.01, kernel='poly'))]
clf = Pipeline(estimators)
t0 = time()

clf = clf.fit(X_test_pca, y_test)

y_pred = clf.predict(X_test_pca)
print "done in %0.3fs" % (time() - t0)
print classification_report(y_test, y_pred, target_names=target_names)
### calculate and return the accuracy on the test data
from sklearn.metrics import accuracy_score
t0 = time()
accuracy = accuracy_score(y_pred, y_test)
print "calc accuracy time:", round(time()-t0, 3), "s"
print "accuraccy was:{}".format (accuracy)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)





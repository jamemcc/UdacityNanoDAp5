#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import pprint
pp = pprint.PrettyPrinter(indent=4)
from time import time

from sklearn import linear_model
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

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

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
#data = featureFormat(my_dataset, features_list, sort_keys = True)
data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=.20, random_state=42)
print labels_test

from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split=2,max_depth=3)

t0 = time()
clf = clf.fit(features_train, labels_train)
print "calc fit time:", round(time()-t0, 3), "s"

#use the feature_importances_ attribute to get a list of the relative importance of all the features being used. We suggest iterating through this list 
print "Type of object for feature import: {}".format(type(clf.feature_importances_))
print "Shape of the training object: {}".format(clf.feature_importances_.shape)
for i in range(len(clf.feature_importances_)):
    if clf.feature_importances_[i] > .005:
        print "index {} was {} for {}".format(i,clf.feature_importances_[i],features_list[i+1] )

###############################################################################
# Quantitative evaluation of the model quality on the test set

print "Predicting the people names on the testing set"
t0 = time()
labels_pred = clf.predict(features_test)
print "done in %0.3fs" % (time() - t0)
target_names = ['not a POI','this is a POI']
print classification_report(labels_test, labels_pred, target_names=target_names)
#print confusion_matrix(y_test, y_pred, labels=range(n_classes))
### calculate and return the accuracy on the test data
from sklearn.metrics import accuracy_score
t0 = time()
accuracy = accuracy_score(labels_pred, labels_test)
print "calc accuracy time:", round(time()-t0, 3), "s"
print "accuraccy was:{}, score was:{}".format (accuracy,clf.score(features_test, labels_test))

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

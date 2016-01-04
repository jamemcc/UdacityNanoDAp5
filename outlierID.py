#!/usr/bin/python

import random
import numpy
import matplotlib.pyplot as plt
import pickle
import sys

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
#from outlier_cleaner import outlierCleaner

"""
    Routine to examine pickle file for outliers.  Prompts for the feature to examine, prints out a listing of users andthat feature's value,  prints a graph and regression of that feature versus their POI result and then prompts for any user for which details of their record will be shown

"""
### load up data with outliers in it and reshape desired attributes into numpy arrays

#fileIn = "../final_project/final_project_dataset_woWeasel_scaled.pkl"
fileIn = "../final_project/final_project_dataset_woWeasel_scaled.pkl"
print fileIn
data_dict = pickle.load(open(fileIn, "r") )

print data_dict['SKILLING JEFFREY K']

#features_list = ['poi','totalEmails','total_stock_value','kill','limit'] # You will need to use more 
feature = raw_input (" What feature? ")
features_list = ['poi',feature] # You will need to use more features

data = featureFormat(data_dict, features_list)
for user in data_dict: print "{} {} {} {}".format (user,
    data_dict[user]['poi'],
    data_dict[user]['email_address'][:4],
    data_dict[user][feature])



#print type(data)

#print data
labels, features = targetFeatureSplit(data)



### labels and features need to be reshaped into 2D numpy arrays
### second argument of reshape command is a tuple of integers: (n_rows, n_columns)
### by convention, n_rows is the number of data points
### and n_columns is the number of features
labels   = numpy.reshape( numpy.array(labels),   (len(labels), 1))
features = numpy.reshape( numpy.array(features), (len(features), 1))
from sklearn.cross_validation import train_test_split
labels_train, labels_test, features_train, features_test = train_test_split(labels, features, test_size=0.1, random_state=42)

### fill in a regression here!  Name the regression object reg so that
### the plotting code below works, and you can see what your regression looks like

from sklearn import linear_model
reg = linear_model.LinearRegression()
reg = reg.fit (labels_train, features_train)
### get the slope ###
slope = reg.coef_[0]
### get the intercept
intercept = reg.intercept_
### get the score on test data
test_score = reg.score(labels_test, features_test)
### get the score on the training data
training_score = reg.score(labels_train, features_train)
print "slope={},intercept={},test score={}, training score={}".format(
    slope,intercept,test_score,training_score)

try:
    plt.plot(labels, reg.predict(labels), color="blue")
except NameError:
    print NameError
plt.scatter(labels, features)
plt.show()

# prompt to explore user data /limits that may be revealed
user = raw_input (" What user? ")
while input is not "":
    print data_dict[user]
    user = raw_input (" What user? ")

### identify and remove the most outlier-y points
cleaned_data = []
try:
    predictions = reg.predict(labels_train)
    cleaned_data = outlierCleaner( predictions, labels_train, features_train )
except NameError:
    print "your regression object doesn't exist, or isn't name reg"
    print "can't make predictions to use in identifying outliers"


### only run this code if cleaned_data is returning data
if len(cleaned_data) > 0:
    labels, features, errors = zip(*cleaned_data)
    labels       = numpy.reshape( numpy.array(labels), (len(labels), 1))
    features = numpy.reshape( numpy.array(features), (len(features), 1))

    ### refit your cleaned data!
    try:
        reg.fit(labels, features)

        ### get the slope ###
        slope = reg.coef_[0]
        ### get the intercept
        intercept = reg.intercept_
        ### get the score on test data
        test_score = reg.score(labels_test, features_test)

        ### get the score on the training data
        training_score = reg.score(labels, features)
        print "slope={},intercept={},test score={}, training score={}".format(
    slope,intercept,test_score,training_score)



        plt.plot(labels, reg.predict(labels), color="blue")
    except NameError:
        print "you don't seem to have regression imported/created,"
        print "   or else your regression object isn't named reg"
        print "   either way, only draw the scatter plot of the cleaned data"
    plt.scatter(labels, features)
    plt.xlabel("labels")
    plt.ylabel("net worths")
    plt.show()


else:
    print "outlierCleaner() is returning an empty list, no refitting to be done"


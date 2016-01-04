#!/usr/bin/python
from __future__ import division
import sys
import pickle
import numpy
sys.path.append("../tools/")
import pprint
pp = pprint.PrettyPrinter(indent=4)
from numbers import Number


from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

from sklearn import preprocessing
"""
    scale the data in final_project_dataset.pkl using min max scaler routine

"""


#list of keys to scale
finKeystoScale = [          'bonus',
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

weaselKeystoScale = []
for weaselWord in weaselList:
    weaselKeystoScale.append(str(weaselWord[0]))

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset_woWeasel.pkl", "r") )
#pp.pprint(data_dict.keys())
pp.pprint(data_dict["SKILLING JEFFREY K"])

### Loop through Financial keys and scale min max based on all other values
for feature in  finKeystoScale: 
    #print "==starting with feature:{}".format(feature)
    # extract feature to be scaled
    features_list = ['poi',feature] 
    data = featureFormat(data_dict, features_list,remove_NaN=True, remove_all_zeroes=False)
    #print data
    ### transform data to numpy array for SCIKIT processing
    labels, features = targetFeatureSplit(data)
    features = numpy.reshape( numpy.array(features), (len(features), 1))    
    # use Sci Kit Learn min max scaler to scale the feature
    #print features [0:5]
    min_max_scaler = preprocessing.MinMaxScaler()
    features = min_max_scaler.fit_transform(features)
    #print features [0:5]
    # put scaled features back in the data dictionary
    n=0
    for user, userdata in data_dict.iteritems():
        data_dict[user][feature] = features[n]
        n+=1

### Loop through user and then each Weasel keys and scale as a % of e-mails sent
for user, userdata in data_dict.iteritems():
    emailSent = data_dict[user]["totalEmails"]
    # handle when the value is not given i.e. 0
    if not isinstance(emailSent,Number): continue
    for feature in  weaselKeystoScale: 
        # handle when the value is not given i.e. 0
        if not isinstance(data_dict[user][feature],Number): continue
        data_dict[user][feature] = data_dict[user][feature] /emailSent

pp.pprint(data_dict["SKILLING JEFFREY K"])

### Store to my_dataset  and export below.
pickle.dump( data_dict, open("final_project_dataset_woWeasel_scaled.pkl", "w") )


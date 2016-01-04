#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import pprint
pp = pprint.PrettyPrinter(indent=4)

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
"""
    merge 2 pickle files, 
        - 1st "final_project_dataset.pkl" was earlier worked up and given as part of Machine Learning course. 
        - 2nd "usersWeasel3.pkl" was generated from custom routines whick separated a set of e-mailboxes by sender and then scanned for count of weasel words.

    the merge matches the keys in the 2 files so that when data exists in both files the new weasel infomation can be merged and if no match adds the new user infomation as a new key.  finally it clears out user keys where there is no weasel word information (i.e. we don't have any sent mail by that user.

"""


#use list of previous keys to ensure new users have all keys
prevExpectedKeys = [          'bonus',
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
#use list of new keys to ensure all users have all the keys
import csv
with open('weaselWords.csv', 'rb') as f:
    reader = csv.reader(f)
    weaselList = list(reader)
print weaselList
weaselKeys = []
for weaselWord in weaselList:
    weaselKeys.append(str(weaselWord[0]))

### Load the dictionary containing the original dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )
#pp.pprint(data_dict.keys())
#pp.pprint(data_dict)
    
### Load the dictionary containing the addtional user counts data
add_dict = pickle.load(open("usersWeasel3.pkl", "r") )
#pp.pprint(add_dict.keys())
#pp.pprint(add_dict)
#exit()

### Map of new data keys to original data keys
keyMap = {
'allen,':'ALLEN PHILLIP K',
'beck,':'BECK SALLY W',
'belden':'BELDEN TIMOTHY N',
'buy,':'BUY RICHARD B',
'calger@enron': 'CALGER CHRISTOPHER F',
'colwell@enron': 'COLWELL WESLEY', 
'delainey': 'DELAINEY DAVID W',
'derrick,': 'DERRICK JR. JAMES V',
'hayslett': 'HAYSLETT RODERICK J',
'horton': 'HORTON STANLEY C',
'kaminski': 'KAMINSKI WINCENTY J',
'kean': 'KEAN STEVEN J',
'kitchen': 'KITCHEN LOUISE', 
'lavorato': 'LAVORATO JOHN J',
'lay,': 'LAY KENNETH L',
'lewis': 'LEWIS RICHARD', 
'martin,': 'MARTIN AMANDA K',
'mccarty': 'MCCARTY DANNY J',
'meyers': 'MEYER JEROME J',
'pereira': 'PEREIRA PAULO V. FERRAZ',
'rice@enron': 'RICE KENNETH D',
'shankman': 'SHANKMAN JEFFREY A',
'shapiro': 'SHAPIRO RICHARD S',
'skilling': 'SKILLING JEFFREY K',
'taylor,': 'TAYLOR MITCHELL S',
'whalley': 'WHALEY DAVID A',
'white,': 'WHITE JR THOMAS E'
}

### Loop through addtional user data
for user, userData in  add_dict.iteritems():  
    ### Task 2: if new user maps to existing user  
    if keyMap.has_key(user):
        ### merge it
        for key, keyValue in userData.iteritems():
            data_dict[keyMap[user]][key] = keyValue
        ### add all missing Weasel keys
        for key in weaselKeys:
            if data_dict[keyMap[user]].has_key(key):
                continue
            else:
                data_dict[keyMap[user]][key] = ""
    ### Task 1: if user data not found 
    else:
        ### add it 
        
        data_dict[user] = {}
        data_dict[user]['email_address'] = user
        data_dict[user]['poi'] = False
        for key, keyValue in userData.iteritems():
            data_dict[user][key] = keyValue
        ### add all other expected NaN keys
        for key in weaselKeys:
            if data_dict[user].has_key(key):
                continue
            else:         
                data_dict[user][key] = ""
        ### add all previous keys
        for key in prevExpectedKeys:
            data_dict[user][key] = ""
### Loop through main user data and remove data where there is no weasel data
keys = data_dict.keys()
for user in  keys:  
    ### Task 3: if users exist without Weasel keys add all as NaN
    if data_dict[user].has_key('totalEmails'):
        continue
    else:
        del data_dict[user]
""" this section commented out as it would prepare data as opposed to deleting it
        for key in weaselKeys:
            data_dict[user][key] = ""
        data_dict[user]['totalEmails'] = 0
"""
### Add POI = true for new POIs
data_dict['scott,']['poi'] = True
data_dict['LAVORATO JOHN J']['poi'] = True
data_dict['BUY RICHARD B']['poi'] = True
data_dict['causey@enron']['poi'] = True


### Store to my_dataset  and export below.
pickle.dump( data_dict, open("final_project_dataset_woWeasel.pkl", "w") )


# UdacityNanoDAp5
Udacity Nanodegree Data Analysis Project 5 - Identifying Fraud from Enron Data

Following python Routines developed as part of this project:

recreateMissing.py: process all e-mails (files) in a directory finding and creating new directories of the e-mails by senders..  These directories are symbolic links to the original files which are in directories by users mailbox.

vectorizeWeaselWords2.py :  process all e-mails (files) in a directory scanning and reporting on the occurence of weasel words as provided in the weaselwords file.

MergeUserdata.py:     merge 2 pickle files, the original class developed Features and observations (enron affiliates) with the Vectorized Weasel word and missing enron email data.  Note that to do this we needed to match e-mail data/observatiosn tot he original observations and in doing so both added additional observations and got rid of observations  (when we did not have any weasel word/sent e-mail data).

outlierID: Routine to examine pickle file for outliers.  Prompts for the feature to examine, prints out a listing of users andthat feature's value,  prints a graph and regression of that feature versus their POI result and then prompts for any user for which details of their record will be shown

scaleData:  scale all features using MinMax Scaling algorithm for the Fin data and %of emails for the Weaselword data.

poi_id_dt – Evaluates the various features (including the new “weasel words” counts created with routines above) to determine their feature importance using decision tree algorithm.  Also produces decision tree algorithm though ultimately the best algorithm was chose in the following routine and the decision tree was not submitted.

poi_id_pca.py – Evaluates various levels of compression into principle components and various Support vector parameters to determine the optimal PCA and support vector algorithm.

Tester.py – Uses StratifiedShuffleSplit to test the results. 

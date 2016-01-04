
#!/usr/bin/python

import os
import subprocess
import pickle
import re
import sys
from time import time
t0 = time()


sys.path.append( "../tools/" )
from parse_out_email_text import parseOutText

"""
    process all e-mails (files) in a directory scanning and reporting on the occurence of weasel words as provided in the weaselwords file.  the intent is to provide a general ability to look for weasel words often used by those trying to get away with someothing they know is not quite correct.  the output of this routine is a list of users and vectors of the amount of weasel words found in their e-mail. 
    the data is stored in lists and packed away in pickle files at the end for subsequent analysis. 

"""

#this is the path to the directory containing folders of people's e-mails
emailFolder = "/media/jamey/UbuntuFiles/ud120-projects/maildirsent2"

#bring in list of bag of weasel words to look for and build grep string
import csv
with open('weaselWords.csv', 'rb') as f:
    reader = csv.reader(f)
    weaselList = list(reader)
firstWord=True
for weaselWord in weaselList:
    wordToFind = str(weaselWord[0])
    if firstWord: searchWords="'"+wordToFind;firstWord=False
    else: searchWords = searchWords + "|"+wordToFind
searchWords = searchWords + "'"
print "will look for {}".format(searchWords)

t1 = time()
#generate list of directories found there
p = subprocess.Popen(
    ["echo "+emailFolder+"/*/"],
    shell=True,
    stdout=subprocess.PIPE, 
    stderr=subprocess.PIPE)
dirStream, err = p.communicate()
dirList = dirStream.split()
print "-found {} directories".format(len(dirList))
print "-time taken was:{} ".format(round(time()-t1))
from_data = []
word_data = []

# user helps add up the counts by user
lastUser = ""
itIsFirstUser = True
# foundCounter accumulates the # of times a word was found
foundCounter = 0
bagOfWeasel={} #  counts and other info for a users e-mail in a dictionary
usersWeasel={} # all users info accumulated in a dictionary

t2 = time()
#loop through all directories found
for directory in dirList:
  t3 = time()
  #first determine the user out of the directory path
  user = directory[53:53+directory[53:80].find("/")] # user name out of directory path
  print directory
  # handle case when this is the first directory path
  if itIsFirstUser: lastUser=user;itIsFirstUser = False
  # handle accumulating and storing counts found once you hit the next user
  if user is not lastUser:
      if user in ["fossum","gang"]: print "---Found {} Finished with user: {} and found {}".format(user,lastUser,bagOfWeasel)
      print "---time taken for {} was:{} ".format(lastUser,round(time()-t3))
      usersWeasel[lastUser] = bagOfWeasel
      bagOfWeasel={}
      lastUser=user

  # determine # of files we will be searching and save that in the user's dictionary
  lsOutput = subprocess.check_output(
      ["ls "+directory+" -LR |wc -l"],
      shell=True)
  emailCount = lsOutput.split(os.linesep)[0]
  if user == "fossum":
    print lsOutput
    print str(emailCount)
  #print " will search {} which has {} files".format(directory,emailCount)
  bagOfWeasel["totalEmails"] = int(str(emailCount)) -1 # -1 since there is an exrta ine in the output

  # set counters to accumulate the counts found
  foundCounter = 0
  #print "searching for {} in {} which is user {}".format(searchWords,directory, user)
  indiv = directory.split("/")[1]

  # using subprocess execute the system command to grep
  # which will search all files in the directory for the weasel word.
  # Note the use of -Eiho options which will count # times found    
  lines = subprocess.check_output(
      ["grep "+searchWords+" "+directory+"* -d recurse -EIho|sort|uniq -c "],
      shell=True)
  if user == "fossum":
    print lines
  # loop through the grep output accumulating the amount of finds in each file
  for line in lines.split(os.linesep):
      #print line
      if len(line.split()) >1:
            try: 
                foundCounter = int(str(line.split()[0]))
                foundWord = str(line.split()[1])
                
                bagOfWeasel[foundWord] = foundCounter
                if user == "fossum":
                    print "foundCounter {} foundWord {}".format(foundCounter,foundWord)

            except ValueError:  continue

# handle accumulating and storing counts found for final user
print "---Finished with user: {} and found {}".format(user,bagOfWeasel)
print "---time taken was:{} ".format(round(time()-t3))
usersWeasel[lastUser] = bagOfWeasel
print usersWeasel

# store off found Weasel word counts for all users for later use
pickle.dump( usersWeasel, open("usersWeasel3.pkl", "w") )
print "overall run time:{} ".format(round(time()-t0))




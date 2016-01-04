#!/usr/bin/python

import os
from subprocess import CalledProcessError, check_output
import pickle
import re
import sys
from time import time
t0 = time()


sys.path.append( "../tools/" )
from parse_out_email_text import parseOutText

"""
    process all e-mails (files) in a directory finding and creating directories of e-mails
    for senders.  These directories are symbolic links in essence remapping the 
    enron e-mail corpus by sender. Input file of e-mail addresses to map is given as
    input file. e-mails.txt

"""

#this is the path to the directory containing folders of people's e-mails
emailFolder = "/media/jamey/UbuntuFiles/ud120-projects/maildir"
# note this must exist and should be empty
emailFolderSent = "/media/jamey/UbuntuFiles/ud120-projects/maildirsent2/" 

#bring in list of e-mails to search and create new directories
f = open('newEmailUserList.txt')
for user in f.read().split('\n'):
    t1 = time()
    #generate list of files sent by the user on this input line
    command = 'egrep "From:.*?'+user+'" '+emailFolder+'  -d recurse -li'
    print command
    try:
        lines = check_output(
            [command],
            shell=True)
    except CalledProcessError as e:
        continue
    print "-found {} files -time taken was:{}".format(len(lines.split(os.linesep)),round(time()-t1))
    t1 = time()
    ### create a subdirectory for this user: 
    # check if user search string needs to be simplified
    if   ' ' in user: userDir = user[0:user.find(' ')]
    elif ',' in user: userDir = user[0:user.find(',')]
    else: userDir = user
    result = check_output(
            ["mkdir "+emailFolderSent+userDir],
            shell=True)

    # loop through the grep output and create a symbolic link for each file
    uniqueFile=1
    for line in lines.split(os.linesep):
                try: 
                    #print line
                    result = check_output(
                        ["ln -s "+line+" "+emailFolderSent+userDir+
                        "/"+str(uniqueFile)+"-"+os.path.basename(line)],
                        shell=True)
                    uniqueFile += 1
                except CalledProcessError as e:
                    continue
    print "-finished creating symbolic dirs time taken was:{} ".format(round(time()-t1))

print "overall run time:{} ".format(round(time()-t0))

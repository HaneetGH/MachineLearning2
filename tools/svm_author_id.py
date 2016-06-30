#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../")
from sklearn import svm
from email_preprocess import preprocess
from class_vis import prettyPicture, output_image
from sklearn.metrics import accuracy_score

# ## features_train and features_test are the features for the training
# ## and testing datasets, respectively
# ## labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
 
clf = svm.SVC(kernel="rbf", C=1500)
t0 = time()
clf.fit(features_train, labels_train)
print "Training time:", round(time() - t0, 3), "s"
# Training Time: 213 seconds
# Training Time w/ subset of data: 0.117 seconds
# Training Time w/ rbf kernel: 0.164 seconds
# Training Time w/ rbf kernel & C=10: 0.127 seconds
# Training Time w/ rbf kernel & C=100: 0.138 seconds
# Training Time w/ rbf kernel & C=1000: 0.127 seconds
# Training Time w/ rbf kernel & C=10000: 0.129 seconds
# Training Time (full) rbf kernel & C=10000: 137.442 seconds
t1 = time()
prediction = clf.predict(features_test)
print "Prediction time:", round(time() - t1, 3), "s"
# Prediction Time: 23 second
# Prediction Time w/ subset of data: 0.88 seconds
# Prediction Time w/ rbf kernel: 1.475 seconds
# Prediction Time w/ rbf kernel & C=10: 1.465 seconds
# Prediction Time w/ rbf kernel & C=100: 1.338 seconds
# Prediction Time w/ rbf kernel & C=1000: 1.279 seconds
# Prediction Time w/ rbf kernel & C=10000: 1.159 seconds
# Prediction Time (full) rbf kernel & C=10000: 14.656 seconds
print accuracy_score(prediction, labels_test)
#########################################################



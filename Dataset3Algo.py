import sys
import os
import numpy as np
from math import sqrt, pi, exp
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.cross_validation import cross_val_score


def getdata(filename):
   data_list = []
   sample_count = 0
   feature_count = 0
   with open(filename, "rb") as f:
       for line in f:
           feature_count += 1
           sample_values = line.split()
           if (sample_count == 0):
               sample_count = len(sample_values)
           feature = []
           for sample in sample_values:
               feature.append(float(sample))
           data_list.append(feature)
   temp_data = np.empty([feature_count, sample_count])
   data = np.empty([sample_count, feature_count])
   temp_data = np.vstack(data_list)
   data = np.transpose(temp_data)
   return data

def getlabeldata(filename):
   sample_count = 0
   train_labels = np.empty([sample_count, 1])
   label_list = []
   with open(filename, "rb") as f:
       for line in f:
           sample_count += 1
           label_list.append(float(line))
   print label_list
   #train_labels = np.append(train_labels, label_list, axis=0)
   #print train_labels
   return label_list

train_filename = os.path.join("D:/Pycharm Workspace/Data Mining/", "train.txt")
test_filename = os.path.join("D:/Pycharm Workspace/Data Mining/", "test.txt")
train_truth_filename = os.path.join("D:/Pycharm Workspace/Data Mining/","train_truth.txt")

train_data = getdata(train_filename)
test_data = getdata(test_filename)
print 'test data shape: ',test_data.shape
train_truths = getlabeldata(train_truth_filename)
train_truths_transpose = np.transpose(train_truths)
print "train_data:", train_data.shape#, " train_truths:", train_truths.shape

UnivLowVariance = VarianceThreshold(threshold=(.7 * (1 - .7)))
UnivLowVariance1 = UnivLowVariance.fit_transform(train_data)
print 'Univ Low Variance: ',UnivLowVariance1.shape


pca = PCA(n_components=cmp)# adjust yourself
pca.fit(train_data)
X_t_train = pca.transform(train_data)
X_t_test = pca.transform(test_data)
clf = SVC()
clf.fit(X_t_train, train_truths)
#print X_t_train.shape, X_t_test.shape
#print 'score', clf.score(X_t_test, clf.predict(X_t_test))
scores = cross_val_score(clf, X_t_train, train_truths)
#print 'score', scores.mean()
results.append(scores.mean())
#print 'pred label', clf.predict(X_t_test)

# pca = PCA(n_components=2)
# clf = ExtraTreesClassifier()
# print 'train truths size: ',train_truths
# clf = clf.fit(UnivLowVariance1, train_truths)
# print 'clf features importances: ',clf.feature_importances_
# count = 0
# for i in clf.feature_importances_:
#     if i != 0.0:
#         count += 1
#         print i
# print count

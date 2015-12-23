from sklearn.feature_selection import VarianceThreshold
import numpy as np
import os

def dataProcess(data):
     dataProcessList = np.empty([len(data), len(data[0])])
     print 'dataProcessList shape: ',dataProcessList.shape
     feature_list = []
     for feature in data:
         sample_list = []
         for indx, sample in enumerate(feature):
             if indx < len(feature)+1:
                 sample_list.append(sample)
             print 'sample list len: ', len(sample_list)
         feature_list = np.append(sample_list)
         print 'feature list len: ', len(feature_list)
     dataProcessList = np.hstack(feature_list)
     print dataProcessList.shape
     return dataProcessList

def getdata(filename):
    data = []
    with open(filename, "rb") as f:
        for line in f:
            features = line.split()
            point = []
            for ft in features:
                try:
                    point.append(float(ft))
                except ValueError:
                    point.append(float(ft))
            data.append(point)
    return data


filename = os.path.join("D:/Pycharm Workspace/Data Mining/", "train.txt");
print filename
data = getdata(filename)
print len(data)
dataProcessList = dataProcess(data)

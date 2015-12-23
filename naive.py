import sys
import os
import numpy as np
import math

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
                    point.append(ft)
            data.append(point)
    return data

def splitdata(data):
    train = []
    test = []
    valid = []
    train_index = int(0.7 * len(data)) - 1
    test_index = (len(data) - 1 - train_index)/2 + 1 + train_index
    #print "train index: ", train_index
    #print "test index: ", test_index
    for idx, val in enumerate(data):
        if (idx <= train_index):
            train.append(val)
        elif(idx > train_index and idx <= test_index):
            test.append(val)
        else:
            valid.append(val)
    splits = [train, test, valid]            
    return splits

def getclasspriors(data):
    class_priors = {}
    total_samples = float(len(data))
    for point in data:
        class_label = point[len(point)-1]
        if (class_priors.has_key(class_label)):
            class_label_value = class_priors.get(class_label)
            class_priors[class_label] = class_label_value + 1.0
        else:
            class_priors[class_label] = 1.0
    for k in class_priors.keys():
        v = class_priors.get(k)
        class_priors[k] = v / total_samples
    return class_priors

def getdescriptorposterior(x, class_label, data):
    desc_posterior = 1.0
    for idx, feature in enumerate(x):
        if (idx < len(x)-1):
            feature_posterior = 1.0
            if (isinstance(feature, float)):
                feature_posterior = getfloatposterior(idx, x, class_label, data)
            elif (isinstance(feature, str)):
                feature_posterior = getstrposterior(idx, x, class_label, data)
            desc_posterior *= feature_posterior
    return desc_posterior

def getstrposterior(feature_index, x, class_label, data):
    feature_values = getfeaturevalues(feature_index, data).get(class_label)
    numerator = 0.0
    denominator = float(len(feature_values))
    for fv in feature_values:
        if (fv == x[feature_index]):
            numerator += 1.0
    return numerator / denominator

def getfloatposterior(feature_index, x, class_label, data):
    feature_stats = getfeaturestats(feature_index, data).get(class_label)
    feature_mean = feature_stats[0]
    feature_var = feature_stats[1]
    x_value = x[feature_index]
    factor = 1.0 / (math.sqrt(2.0*math.pi*feature_var))
    exponent = -((x_value-feature_mean)**2.0)/(2.0*feature_var)
    posterior = factor * math.exp(exponent)
    return posterior

def getfeaturevalues(feature_index, data):
    feature_values = {}  # k:class_label, v:list of feature values
    for point in data:
        feature_value = point[feature_index]
        class_label = point[len(point)-1]
        if (feature_values.has_key(class_label)):
            values = feature_values.get(class_label)
            values.append(feature_value)
            feature_values[class_label] = values
        else:
            feature_values[class_label] = [feature_value]
    return feature_values

def getfeaturestats(feature_index, data):
    feature_values = getfeaturevalues(feature_index, data)
    feature_stats = {}
    for class_label in feature_values.keys():
        values = feature_values.get(class_label)
        mean = np.mean(values)
        var = np.var(values)
        feature_stats[class_label] = [mean, var]
    return feature_stats


filename = os.path.join(sys.argv[0], "project3_dataset1.txt");
data = getdata(filename)
train = []
test = []
valid = []
splits = splitdata(data)
train = splits[0]
test = splits[1]
valid = splits[2]
#print "TRAIN"
#for idx, val in enumerate(train):
#    print idx, val
print "# Training samples = ", len(train)
print "# Testing samples = ", len(test)
#print "# Validation samples = ", len(valid)
#print "# Total samples = ", len(data)

class_priors = getclasspriors(train)
class_label0 = 0.0
class_label1 = 1.0
class_prior0 = class_priors.get(0.0)
class_prior1 = class_priors.get(1.0)
predictions = []
for x in valid:
    p0 = getdescriptorposterior(x, class_label0, train)
    p1 = getdescriptorposterior(x, class_label1, train)
    if (p0 < p1):
        predictions.append(1.0)
    elif (p0 > p1):
        predictions.append(0.0)
    else:
        predictions.append(3.0)
total_inputs = 0
correct_inputs = 0
for idx, x in enumerate(valid):
    predicted = predictions[idx]
    true_value = x[len(x)-1]
    total_inputs += 1
    if (predicted == true_value):
        correct_inputs += 1
print "correct predictions   = ", correct_inputs
print "incorrect predictions = ", total_inputs-correct_inputs
print "total predictions     = ", total_inputs
print "Accuracy = ", float(correct_inputs)/float(total_inputs)

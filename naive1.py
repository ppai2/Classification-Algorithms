import sys
import os
import numpy as np
from math import sqrt, pi, exp
import matplotlib.pyplot as plt



#----------------------- NAIVE BAYES CODE ------------------------
def getnaivedata(filename):
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
    f.close()
    return data

def splitnaivedata(data):
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
    factor = 1.0 / (sqrt(2.0*pi*feature_var))
    exponent = -((x_value-feature_mean)**2.0)/(2.0*feature_var)
    posterior = factor * exp(exponent)
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

def naivebayes(test, train):
    class_priors = getclasspriors(train)
    class_label0 = 0.0
    class_label1 = 1.0
    class_prior0 = class_priors.get(0.0)
    class_prior1 = class_priors.get(1.0)
    predictions = []
    truths = []
    for x in test:
        p0 = class_prior0 * getdescriptorposterior(x, class_label0, train)
        p1 = class_prior1 * getdescriptorposterior(x, class_label1, train)
        if (p0 < p1):
            predictions.append(1.0)
        elif (p0 > p1):
            predictions.append(0.0)
        else:
            predictions.append(3.0)
    for idx, x in enumerate(test): 
        truths.append(x[len(x)-1])
    return predictions, truths
#----------------------- END OF NAIVE BAYES CODE --------------------------

#----------------------- PERFORMANCE MEASURES CODE ------------------------
def getconfusionmatrix(predictions, truths):
    positive0 = {"tp":0, "tn":0, "fp":0, "fn":0}
    positive1 = {"tp":0, "tn":0, "fp":0, "fn":0}
    for idx, pred in enumerate(predictions):
        truth = truths[idx]
        if (pred == truth and pred == 0.0):
            positive0["tp"] += 1
        elif (pred == truth and pred == 1.0):
            positive0["tn"] += 1
        elif (pred != truth and pred == 0.0):
            positive0["fp"] += 1
        elif (pred != truth and pred == 1.0):
            positive0["fn"] += 1
    for idx, pred in enumerate(predictions):
        truth = truths[idx]
        if (pred == truth and pred == 1.0):
            positive1["tp"] += 1
        elif (pred == truth and pred == 0.0):
            positive1["tn"] += 1
        elif (pred != truth and pred == 1.0):
            positive1["fp"] += 1
        elif (pred != truth and pred == 0.0):
            positive1["fn"] += 1
    return positive0, positive1

def getaccuracy(positive0, positive1):
    num = float(positive0["tp"] + positive0["tn"])
    denom = float(positive0["tp"] + positive0["tn"] + positive0["fp"] + positive0["fn"])
    return num / denom

def getprecision(positive0, positive1):
    num0 = float(positive0["tp"])
    num1 = float(positive1["tp"])
    denom0 = float(positive0["tp"] + positive0["fp"])
    denom1 = float(positive1["tp"] + positive1["fp"])
    return num0 / denom0, num1 / denom1
    
def getrecall(positive0, positive1):
    num0 = float(positive0["tp"])
    num1 = float(positive1["tp"])
    denom0 = float(positive0["tp"] + positive0["fn"])
    denom1 = float(positive1["tp"] + positive1["fn"])
    return num0 / denom0, num1 / denom1

def getfmeasure(precision0, precision1, recall0, recall1):
    f0 = (2.0 * recall0 * precision0) / (recall0 + precision0)
    f1 = (2.0 * recall1 * precision1) / (recall1 + precision1)
    return f0, f1

def getperformance(predictions, truths):
    positive0, positive1 = getconfusionmatrix(predictions, truths)
    #print positive0, positive1
    accuracy = getaccuracy(positive0, positive1)
    precision0, precision1 = getprecision(positive0, positive1)
    recall0, recall1 = getrecall(positive0, positive1)
    f0, f1 = getfmeasure(precision0, precision1, recall0, recall1)
    perf = {"ac":accuracy, "p0":precision0, "p1":precision1, "r0":recall0, "r1":recall1, "f0":f0, "f1":f1}
    #print "Accuracy: ", accuracy
    #print "Precision0: ", precision0
    #print "Precision1: ", precision1
    #print "Recall0: ", recall0
    #print "Recall1: ", recall1
    #print "F_Measure0: ", f0
    #print "F_Measure1: ", f1
    return perf
#----------------------- END OF PERFORMANCE MEASURES CODE ------------------------

#----------------------- CROSSVALIDATION CODE ------------------------------------
def mergekperfs(kperfs):
    mergedperfs = {"ac":0.0, "p0":0.0, "p1":0.0, "r0":0.0, "r1":0.0, "f0":0.0, "f1":0.0}
    k = len(kperfs)
    for perf in kperfs:
        mergedperfs["ac"] += perf["ac"]
        mergedperfs["p0"] += perf["p0"]
        mergedperfs["p1"] += perf["p1"]
        mergedperfs["r0"] += perf["r0"]
        mergedperfs["r1"] += perf["r1"]
        mergedperfs["f0"] += perf["f0"]
        mergedperfs["f1"] += perf["f1"]
    for key in mergedperfs.keys():
        mergedperfs[key] /= k
    return mergedperfs
    

def getkfolds(k, data):
    kfold = []
    fold_count = 0
    fold = []
    fold_block_size = float(len(data)) / float(k)
    remainder = 0
    if (round(fold_block_size) < fold_block_size):
        remainder = len(data) % k
    fold_limit = int(round(fold_block_size))
    for idx, point in enumerate(data):
        if (fold_count < fold_limit):
            fold.append(point)
            fold_count += 1
            #print idx, "->", fold_count
            if (fold_count == fold_limit and idx < len(data)-remainder-1):
                kfold.append(fold)
                fold = []
                fold_count = 0
            elif (fold_count == fold_limit and idx == len(data)-remainder-1):
                fold_count -= remainder
        if (idx == len(data)-1):
            kfold.append(fold)
    #print "# folds = ", len(kfold)
    #for f in kfold:
    #    print "# samples in fold = ", len(f)
    return kfold

def mergefolds(kfolds, exclude_idx):
    ktest = []
    ktrain = []
    for idx, fold in enumerate(kfolds):
        if (idx == exclude_idx):
            ktest = fold
        else:
            for sample in fold:
                ktrain.append(sample)
    #print "ktest size = ", len(ktest)
    #print "ktrain size = ", len(ktrain)
    return ktrain, ktest

def naivecrossvalidation(k, data):
    kfold_data = getkfolds(k, data)
    kperfs = []
    for idx, fold in enumerate(kfold_data):
        ktrain, ktest = mergefolds(kfold_data, idx)
        kpredictions, ktruths = naivebayes(ktest, ktrain)
        kperf = getperformance(kpredictions, ktruths)
        kperfs.append(kperf)
    #for perf in kperfs:
    #    print perf
    #    print "----------------------------------"
    #print "# perfs = ", len(kperfs)
    plotkperfs(kperfs)
    perfs = mergekperfs(kperfs)
    return perfs
#----------------------- END OF CROSSVALIDATION CODE ------------------------

def printperfmeasures(perfs):
    print "------------------------------------------------------"
    print "Accuracy", perfs["ac"]
    print "------------------------------------------------------"
    print "assumption: class_label 0 = positive, 1 = negative"
    print "Precision", perfs["p0"]
    print "Recall", perfs["r0"]
    print "FMeasure", perfs["f0"]
    print "------------------------------------------------------"
    print "assumption: class_label 1 = positive, 0 = negative"
    print "Precision", perfs["p1"]
    print "Recall", perfs["r1"]
    print "FMeasure", perfs["f1"]
    print "------------------------------------------------------"

def plotkperfs(kperfs):
    acs = []
    p0s = []
    p1s = []
    r0s = []
    r1s = []
    f0s = []
    f1s = []
    for kperf in kperfs:
        acs.append(kperf["ac"])
        p0s.append(kperf["p0"])
        p1s.append(kperf["p1"])
        r0s.append(kperf["r0"])
        r1s.append(kperf["r1"])
        f0s.append(kperf["f0"])
        f1s.append(kperf["f1"])
    print 'Accuracy:', acs
    print 'Precision0:', p0s
    print 'Precision1:', p1s
    print 'Recall0:', r0s
    print 'Recall1:', r1s
    print 'F-Measure0:', f0s
    print 'F-Measure1:', f1s
    plt.close('all')
    #showplot(acs, 'Fold', 'Accuracy', 'NAIVE BAYES - Dataset2 - Accuracy vs Fold')
    #showplot(p0s, 'Fold', 'Precision', 'NAIVE BAYES - Dataset2 - 0:positive, 1:negative')
    #showplot(r0s, 'Fold', 'Recall', 'NAIVE BAYES - Dataset2 - 0:positive, 1:negative')
    #showplot(f0s, 'Fold', 'F-Measure', 'NAIVE BAYES - Dataset2 - 0:positive, 1:negative')
    #showplot(p1s, 'Fold', 'Precision', 'NAIVE BAYES - Dataset2 - 1:positive, 0:negative')
    #showplot(r1s, 'Fold', 'Recall', 'NAIVE BAYES - Dataset2 - 1:positive, 0:negative')
    #showplot(f1s, 'Fold', 'F-Measure', 'NAIVE BAYES - Dataset2 - 1:positive, 0:negative')
    #plt.show()

def showplot(y_values, x_label, y_label, plot_title):
    size = len(y_values)
    x_values = []
    for i in range(size):
        x_values.append(i+1)
    plt.plot(x_values, y_values)
    plt.title(plot_title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.show()
    
    


#-------------------------------- MAIN SCRIPT -------------------------------

dataset = sys.argv[2]
cross_valid = sys.argv[3]
k = 10


filenames = os.path.join("D:/Pycharm Workspace/Data Mining", "project3_dataset1.txt"),
#filenames = os.path.join(sys.argv[1], "project3_dataset2.txt");
data = getnaivedata(filenames)

if (cross_valid):
    perfs = naivecrossvalidation(k, data)
    print "CrossValidation Naive Bayes performance measures"
    printperfmeasures(perfs)
else:
    splits = splitnaivedata(data)
    train = splits[0]
    test = splits[1]
    valid = splits[2]
    n_predictions, n_truths = naivebayes(test, train)
    n_perf = getperformance(n_predictions, n_truths)
    print "Naive Bayes performance measures"
    printperfmeasures(n_perf)

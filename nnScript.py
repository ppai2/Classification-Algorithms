from __future__ import division
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import matplotlib.pyplot as plt
import pickle
import timeit
import os
from sklearn import preprocessing

def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer

    # Output:
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W

def sigmoid(z):

    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    sig_val = 1/(1 + np.exp(-z))
    return  sig_val#your code here

def featureLabelClassify(data):
    features = []
    #num_columns = len(data[0])
    print len(data)
    final_data = np.empty([len(data), len(data[0])-1])
    for sample in data:
        sample_feat = []
        for idx, feat in enumerate(sample):
            if (idx < len(sample)-1):
                sample_feat.append(feat)
        features.append(sample_feat)
    final_data = np.vstack([features])
    return final_data

def classLabelClassify(data):
    sample_class = []
    final_class_data = np.empty((len(data),2))
    for sample in data:
        if sample[-1] == 1.0:
            sample_class.append([0.0,1.0])
        elif sample[-1] == 0.0:
            sample_class.append([1.0,0.0])
    final_class_data = np.vstack(sample_class)
    return final_class_data

def getdata(filename):
    data = []
    le = preprocessing.LabelEncoder()
    with open(filename, "rb") as f:
        for line in f:
            features = line.split()
            point = []
            for ft in features:
                try:
                    point.append(float(ft))
                except ValueError:
                    le.fit(["Present", "Absent"])
                    list(le.classes_)
                    point.append(float(le.transform(ft)))
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

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log
    %   likelihood error function with regularization) given the parameters
    %   of Neural Networks, thetraining data, their corresponding training
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, train_label, lambdaval = args
    #print 'input units: ',n_input
    #print 'hidden units: ',n_hidden
    #print 'output units: ',n_class
    #print 'train data shape in nnObj: ',training_data.shape
    #print 'train label shape in nnObj: ',train_label.shape
    #print 'lambdaval in nnObj: ',lambdaval

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    obj_val = 0

    n_iterations = 50000

    #print 'shape of w1 is', str(w1.shape)
    #print 'shape of w2 is', str(w2.shape)

    a = np.array([])
    b = np.array([])
    z = np.array([])
    o = np.ndarray([])

    #print '\n---------------------'
    #feed forward from input to hidden layer
    #making bias vector, put in temp, multiply with w1 to get a
    #bias1 = np.ndarray([1,n_input])
    bias1 = np.ones_like(training_data[:,-1]).reshape(-1, 1)
    #bias1.fill('1.0')
    #print 'bias1 shape is', bias1.shape

    #print 'training data shape: ',training_data.shape
    temp1 = np.hstack((training_data, bias1))
    #print 'temp1 is', temp1.shape
    #print 'w1 transp is', np.transpose(w1).shape

    z = sigmoid(np.dot(temp1, np.transpose(w1)))
    #print 'shape z is', z.shape

    #print '\n---------------------'
    #feed forward from hidden to output layer
    #making bias vector, put in temp, multiply with w2 to get b
    #bias2 = np.ndarray([1, n_hidden])
    #bias2 = np.hstack(z, bias1)
    #bias2.fill('1.0')
    #print 'bias2 shape is', bias2.shape

    temp2 = np.hstack((z, bias1))
    #print 'temp2 is', temp2.shape
    #print 'w2 transp is', np.transpose(w2).shape

    o = sigmoid(np.dot(temp2, np.transpose(w2)))
    #print 'shape o is', o.shape
    #print ' o is ', o

    #print '\n---------------------'
    #calculate objective function
    #print np.log(0.58372467)
    J = (sum(sum(train_label*np.log(o) + ((1-train_label) * np.log(1-o)))))/(-n_iterations)
    #print 'J=', J, ' nad its shape is ', J.shape

    #calculate error gradient
    deltaJO = np.ndarray([])

    deltaL = o-train_label
    #deltaL[np.newaxis,:]
    #+z[np.newaxis,:]
    #print 'deltaL', deltaL.shape

    #deltaJO = np.dot(deltaL[0][:,np.newaxis], z[0][:,np.newaxis].T)
    #for i in range(1,n_iterations):
     #   deltaJO = np.dot(deltaL[i][:,np.newaxis], z[i][:,np.newaxis].T)


    #eq 9
    deltaJO = np.dot(np.transpose(deltaL), temp2)
    #
    #print 'deltaJO shape', deltaJO.shape


    #print '\n---------------------'
    #derivative of error function wrt weight from input to hidden layer ... eq 10-12
    deltaJH = np.ndarray([])

    #w2_del = np.delete(w2, (-1), axis=1)
    #print 'w2 del shape', w2_del.shape

    #zt = np.transpose((1-temp2))
    de = (1-temp2)*temp2
    #print 'zt shape', zt.shape
    #print 'de shape', de.shape
    abc = np.dot(deltaL, w2)
    #print 'abc shape', abc.shape
    #eq12
    #deltaJH = np.dot(de , (np.dot(np.transpose(abc), temp1)))
    deltaJH = np.dot(np.transpose(de*abc),temp1)
    #print 'deltaJH shape', deltaJH.shape
    #print deltaJH[0]


    #regularization
    #eq15
    weights_sum = sum(sum(w1*w1)) + sum(sum(w2*w2))
    #print 'weights_sum ',weights_sum
    Jbar = J + (lambdaval / (2*n_iterations)) * weights_sum
    #print 'Jbar', Jbar

    #eq16
    deltaJObar = (deltaJO + lambdaval*w2) / (n_iterations)
    #print deltaJObar.shape

    #eq17
    deltaJHbar = (np.delete(deltaJH, (-1), axis = 0) + lambdaval*w1) / (n_iterations)
    #print deltaJHbar.shape

    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
    #obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.array([])
    obj_val = Jbar
    grad_w1 = deltaJHbar
    grad_w2 = deltaJObar
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    #print 'obj_grad ' , obj_grad
    #print 'obj_val ' , obj_val
    return (obj_val,obj_grad)

def nnPredict(w1,w2,data):

    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature
    %       vector of a particular image

    % Output:
    % label: a column vector of predicted labels"""

    labels = np.array([])

    bias1 = np.ndarray([data.shape[0],1])
    bias1.fill('1.0')

    temp1 = np.hstack((data, bias1))
    a = np.dot(temp1, np.transpose(w1))
    z = sigmoid(a)

    #feed forward from hidden to output layer
    #making bias vector, put in temp, multiply with w2 to get b
    bias2 = np.ndarray([data.shape[0],1])
    bias2.fill('1.0')

    temp2 = np.hstack((z, bias2))

    b = np.dot(temp2, np.transpose(w2))
    o = sigmoid(b)

    labels = np.argmax(o, axis=1)
    #print 'shape of labels', labels.shape
    return labels




def neural(train, test, valid, data):
    num_columns = len(data[0])

    # training data split into feature labels and class labels
    train_data = featureLabelClassify(train)
    print 'train data shape: ',train_data.shape
    train_label = classLabelClassify(train)
    print 'train label shape: ',train_label.shape

    # test data split into feature labels and class labels
    test_data = featureLabelClassify(test)
    print 'test data shape: ',test_data.shape
    test_label = classLabelClassify(test)
    print 'test label shape: ',test_label.shape

    # validation data split into feature labels and class labels
    validation_data = featureLabelClassify(valid)
    print 'validation data shape: ',validation_data.shape
    validation_label = classLabelClassify(valid)
    print 'validation label shape: ',validation_label.shape

    np.seterr (over = 'ignore')

    #  Train Neural Network predicted_label

    # set the number of nodes in input unit (not including bias unit)
    n_input = (num_columns-1);

    # set the number of nodes in hidden unit (not including bias unit)
    n_hidden = 50;

    # set the number of nodes in output unit
    n_class = 2;

    # initialize the weights into some random matrices
    initial_w1 = initializeWeights(n_input, n_hidden);
    initial_w2 = initializeWeights(n_hidden, n_class);

    # unroll 2 weight matrices into single column vector
    initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

    # set the regularization hyper-parameter
    lambdaval = 0.2;

    args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

    #Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

    opts = {'maxiter' : 50}    # Preferred value.

    nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

    #In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
    #and nnObjGradient. Check documentation for this function before you proceed.
    #nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


    #Reshape nnParams from 1D vector into w1 and w2 matrices
    w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    #Test the computed parameters

    train_predicted_label = nnPredict(w1,w2,train_data)
    print 'Predicted label for train data:' , train_predicted_label.shape
    predicted_label_to_train_list = predictLabelToList(train_predicted_label)
    #print 'predicted label to train list: ',predicted_label_to_train_list

    #find the accuracy on Training Dataset
    train_label = np.argmax(train_label, axis=1)
    print 'Train label :' , train_label.shape
    train_label_to_train_list = predictLabelToList(train_label)
    #print 'train label to list: ',train_label_to_train_list
    print('Training set Accuracy:' + str(100*np.mean((train_predicted_label == train_label).astype(float))) + '%')

    #find the accuracy on Validation Dataset
    valid_predicted_label = nnPredict(w1,w2,validation_data)
    print '\nPredicted label for validation data:', valid_predicted_label.shape

    validation_label = np.argmax(validation_label, axis=1)
    print 'Validation label :' , validation_label.shape
    print('Validation set Accuracy:' + str(100*np.mean((valid_predicted_label == validation_label).astype(float))) + '%')

    #find the accuracy on Test Dataset
    test_predicted_label = nnPredict(w1,w2,test_data)
    predicted_label_to_test_list = predictLabelToList(test_predicted_label)
    #print '\npredicted label to test list: ',predicted_label_to_test_list
    print 'Predicted label for test data:' , test_predicted_label.shape

    test_label = np.argmax(test_label, axis=1)
    test_label_to_test_list = predictLabelToList(test_label)
    #print 'test label to test list: ',test_label_to_test_list
    print 'Test label :' , test_label.shape
    print('Test set Accuracy:' + str(100*np.mean((test_predicted_label == test_label).astype(float))) + '%')

    end = timeit.default_timer()
    print '\nn_input=',n_input
    print 'n_hidden=',n_hidden
    print 'lambdaval=', lambdaval
    print ('Runtime :' + str(end - start))

    test_count = 0
    for l in test_label_to_test_list:
        if (l == 1.0):
            test_count += 1
    pred_count = 0
    for l in predicted_label_to_test_list:
        if (l == 1.0):
            pred_count += 1
    print "Test 1 count: ", test_count
    print "Pred 1 count: ", pred_count
    print predicted_label_to_test_list

    return predicted_label_to_test_list, test_label_to_test_list

def predictLabelToList(predicted_label):
    train_label_list = []
    flag = 0
    for i in predicted_label:
        train_label_list.append(float(i))
        if (float(i) == 1.0):
            flag = 1
    if (flag == 0):
        "NO 1 PREDICTED"
    #print train_label_list
    return train_label_list

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
    print positive0, positive1
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
def neuralcrossvalidation(k, data):
    kfold_data = getkfolds(k, data)
    kperfs = []
    for idx, fold in enumerate(kfold_data):
        ktrain, ktest = mergefolds(kfold_data, idx)
        kpredictions, ktruths = neural(ktrain, ktest, ktest, data)
        kperf = getperformance(kpredictions, ktruths)
        kperfs.append(kperf)
    #for perf in kperfs:
    #    print perf
    #    print "----------------------------------"
    #print "# perfs = ", len(kperfs)
    perfs = mergekperfs(kperfs)
    return perfs
"""**************Neural Network Script Starts here********************************"""

start = timeit.default_timer()

#train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();

filename = os.path.join("D:/Pycharm Workspace/Data Mining/", "project3_dataset2.txt");
print filename
data = getdata(filename)
k = 10
#print len(data)

splits = splitdata(data)

# splits from splitdata divided into training, test and validation
train = splits[0]
test = splits[1]
valid = splits[2]

predictions, truths = neural(train, test, valid, data)
perf = getperformance(predictions, truths)
print perf

k_perfs = neuralcrossvalidation(k, data)
print k_perfs

##Data dump
# print ('Writing to pickle file')
# pickle.dump([n_input, n_hidden, w1, w2, lambdaval], open( "params.pickle", "wb" ))
# print ('Writing to pickle file done')



#with open('params.pickle', 'w') as g:
#    pickle.dump([n_input, n_hidden, w1, w2, lambdaval], g, protocol=pickle.HIGHEST_PROTOCOL)

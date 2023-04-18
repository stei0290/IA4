

#CONSTANTS
import csv
import math as math
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
sp.init_printing(use_unicode=True, use_latex='mathjax')


TRAIN_FILE = 'knn_train.csv'
TEST_FILE = 'knn_test.csv'


#Read Data
def readCSVData(fileName):
    """    
    Function:   readData
    Descripion: Opens and reads text file
    Input:      fileName - name of file to read from
    Output:     dataList - numpy array of data from file being read
    """
    dataList = []
    # with open(fileName, "r") as f:
    #     raw = f.read()
    #     for line in raw.split("\n"):
    #         subLine = line.split()
    #         dataList.append(subLine)

    with open(fileName, 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            # print(row)
            dataList.append(row)
    return dataList

def normalize(train, predict):
    """    
    Function:   normalize
    Descripion: normalizes all data in the given matrix column by column
    Input:      train - the training data of the set
                test - the testing data of the set
    Output:     dataCopy - normalized matrix
    """
    trainCopy = np.copy(train)
    predictCopy = np.copy(predict)

    for col in range(trainCopy.shape[1]):
        min = np.min(trainCopy.T[:][col])
        max = np.max(trainCopy.T[:][col])
        for i in range(len(trainCopy.T[:][col])):
            norm = (trainCopy[i][col] - min)/(max - min)
            trainCopy[i][col] = norm
        for i in range(len(predictCopy.T[:][col])):
            norm = (predictCopy[i][col] - min)/(max - min)
            predictCopy[i][col] = norm

    return trainCopy, predictCopy

def distance(item1, item2):
    """    
    Function:   distance
    Descripion: calculates distance between two points
    Input:      item1 - start location for distance
                item2 - end location for distance
    Output:     np.linalg.norm(item1 - item2)
    """

    return np.linalg.norm(item1 - item2)

def kNearest(k, point, data, labels):
    """    
    Function:   kNearest
    Descripion: calculates the k nearest neigbors to the given point
    Input:      k - number of nearest points 
                point - given point to calculate the k nearest of
                data - the data
                labels - label of each data point
    Output:     minDistances - a list of distances to k nearest points
                minLabels - the classification of the k nearest points
    """
    distanceList = []
    labelCopy = list(labels.T)
    minDistances = []
    minLabels = []

    for i in data:
        distanceList.append(distance(point, i))
    for i in range(k):
        minimum = min(distanceList)
        for i in range(len(distanceList)):
            if distanceList[i] == minimum:
                minDistances.append(distanceList[i])
                minLabels.append(labelCopy[i])
                del distanceList[i]
                del labelCopy[i] # might need to be axis 1
                break

    return minDistances, minLabels

def voting(distances, labels):
    """    
    Function:   voting
    Descripion: decides classification based on returns from k nearest function
    Input:      distances - a list of distances to k nearest points
                labels - the classification of the k nearest points
    Output:     1 or -1 based on voting scheme
    """
    sum = 0
    for i in range(len(distances)):
        #sum = sum + math.exp(-2 * distances[i]) * int(labels[i])
        sum = sum + int(labels[i])
    if sum >= 0:
        return 1
    else:
        return -1

def kNearestDecide(k, point, data, labels):
    """    
    Function:   kNearestDecide
    Descripion: decides classification  for the entered point
    Input:      k - the number of neighbors to classify by
                point - data to classify
                data - dataset to classify by
                labels - the ckass labels of the data
    Output:     1 or -1 based on voting scheme
    """
    kDistances, kLabels = kNearest(k, point, data, labels)
    return voting(kDistances, kLabels)

def Error(Y, Y_predict):
    """    
    Function:   Error
    Descripion: calculates the error of a given prediction set
    Input:      Y - the actual classification data
                Y_predict - the predicted values
    Output:     1 - (correct/len(Y.T)) - Error calculation
    """
    correct = 0
    for num in range(0,len(Y.T)):
        if int(Y[0, num]) == int(Y_predict[num]):
            correct += 1
    
    return 1 - (correct/len(Y.T)) 

def KNN_SSE(k, trainingSet, trainingLabels, testingSet, testingLabels):
    """    
    Function:   KNN_SSE
    Descripion: calculates the SSE of a given KNN scheme
    Input:      k - the number of neighbors to classify by
                trainingSet - the set to train KNN of
                traingingLabels - the class labels of the training data
                testingSet - the set to test KNN on
                testingLabels - the class labels of the testing set
    Output:     Error(testingLabels, calculatedLabels) - error of predicted classification
    """
    calculatedLabels = []
    normTraining, normTesting = normalize (trainingSet, testingSet)
    for i in range(len(testingSet)):
        calculatedLabels.append(kNearestDecide(k, normTraining[i], normTesting, trainingLabels))
    return Error(testingLabels, calculatedLabels)

def KCrossVal(KFolds, kNeigh, data, labels):
    """
    Function:    KCrossVAl
    Description: Preforms K-Fold Cross validation, splitting the data into K
                 indices and calculating the SSE using each set as the
                 valudation set, returning the average of each SSE
    Input:       KFolds - number of indices
                 kNeigh - number of neighbors for knn
                 data - the matrix containing the features
                 labels - the array containing the labels
    Output:      Average of SSE for each fold.
    """
    secSize = math.ceil(len(data)/KFolds)
    errors = []
   
    for i in range(KFolds):
        dataCopy = np.copy(data)
        labelCopy = np.copy(labels)
       
        valDSet = []
        valLSet = []
        for j in range(i*secSize, i*secSize+secSize-1):
            if j < len(dataCopy) - 1:
                valDSet.append(dataCopy[j])
                valLSet.append(labelCopy[j])
           
                dataCopy = np.delete(dataCopy, i*secSize, 0)
                labelCopy = np.delete(labelCopy, i*secSize, 0)
        valDSet = np.array(valDSet)
        valLSet = np.array(valLSet)
        SSE = KNN_SSE(kNeigh, dataCopy, labelCopy.T, valDSet, valLSet.T)
        errors.append(SSE)

    return 1/KFolds * sum(errors)

def driver():
    K_VALUE = 14


    testData = []
    trainData = []
    trainOutput = []
    testOutput = []

    knnSSEArr = []
    crossValArr = []
    trainSSEArr = []
    kValueArr = []

    testLabel_Index = 0
    trainLabel_Index = 0

    testData = readCSVData(TEST_FILE)
    trainData = readCSVData(TRAIN_FILE)

    #Apply outpts to its own array
    for i in range(len(trainData)):
        trainOutput.append(trainData[i][trainLabel_Index])
    for i in range(len(testData)):
        testOutput.append(testData[i][testLabel_Index])

    #Convert outputs to numpy array
    trainOutput = np.array(trainOutput, dtype=float)
    testOutput = np.array(testOutput, dtype=float) 

    #Create input array
    trainFeatures = np.array(trainData, dtype=float)
    testFeatures = np.array(testData,dtype=float)

    numTrainFeatures = len(trainOutput)

        #Delte label from feature array
    trainFeatures = np.delete(trainFeatures,0, axis=1)
    testFeatures = np.delete(testFeatures,0, axis=1)

    #reshape arary
    trainOutput = trainOutput.reshape(numTrainFeatures,1)
    numTestFeatures =  len(testFeatures)
    testOutput = testOutput.reshape(numTestFeatures,1)
    numTrainFeatures = len(trainFeatures)

    for i in range(K_VALUE):
        print(2*i +1, ':')
        knnSSE = KNN_SSE(2*i + 1, trainFeatures, trainOutput.T, trainFeatures, trainOutput.T)
        knnSSEArr.append(knnSSE)
        print(f"KNN SSE: {knnSSE}")
        kCrossVal = KCrossVal(5, 2*i + 1, trainFeatures, trainOutput)
        crossValArr.append(kCrossVal)
        print(f"Cross Validation Error: {kCrossVal}")
        trainSSE = KNN_SSE(2*i + 1, trainFeatures, trainOutput.T, testFeatures, testOutput.T)
        trainSSEArr.append(trainSSE)
        print(f"Train SSE: {trainSSE}")
        kValueArr.append(i*2 + 1)
        print()

    print(KNN_SSE(280, trainFeatures, trainOutput.T, trainFeatures, trainOutput.T))
    print(KNN_SSE(280, trainFeatures, trainOutput.T, testFeatures, testOutput.T))

    # plt.plot(kValueArr, knnSSEArr, 'k-', label="KNN SSE ")
    # plt.plot(kValueArr,crossValArr, 'b-', label="Cross Validation SSE" )
    # plt.plot(kValueArr, trainSSEArr, 'r-', label="Train SSE")
    # plt.xlabel("K-Value")
    # plt.legend(loc='lower right')
    # plt.show()


    figure, axis = plt.subplots(2,2)

    axis[0,0].plot(kValueArr, knnSSEArr, 'k-')
    axis[0,0].set_title("KNN SSE")

    axis[0,1].plot(kValueArr,crossValArr, 'b-')
    axis[0,1].set_title("Cross Validation SSE")

    axis[1,0].plot(kValueArr, trainSSEArr, 'r-', label="Train SSE")
    axis[1,0].set_title("Train SSE")

    plt.show()


# figure, axis = plt.subplots(2, 2)
  
# # For Sine Function
# axis[0, 0].plot(X, Y1)
# axis[0, 0].set_title("Sine Function")
  
# # For Cosine Function
# axis[0, 1].plot(X, Y2)
# axis[0, 1].set_title("Cosine Function")
  
# # For Tangent Function
# axis[1, 0].plot(X, Y3)
# axis[1, 0].set_title("Tangent Function")

    

driver()
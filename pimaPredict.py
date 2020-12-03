import time
import math
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def pre():
    dataset = pd.read_csv('diabetes.csv')

    # referring to the outlier on the png, replace zero values with median
    dataset['Glucose'].replace(0,dataset['Glucose'].median(),inplace=True) 
    dataset['BloodPressure'].replace(0,dataset['BloodPressure'].median(),inplace=True) 
    dataset['SkinThickness'].replace(0,dataset['SkinThickness'].median(),inplace=True) 
    dataset['Insulin'].replace(0,dataset['Insulin'].median(),inplace=True) 
    dataset['BMI'].replace(0,dataset['BMI'].median(),inplace=True)
    # print(dataset.describe())
    
    sc = StandardScaler()
    data =  pd.DataFrame(sc.fit_transform(dataset.drop(["Outcome"],axis = 1),),
        columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age'])
    # data = dataset.drop(columns=['Outcome'])
    target = dataset['Outcome']
    # print(target.shape) 768 items/ 80% = 614 and 20% = about 154
    
    X = data.values
    Y = target.values
    # split the data
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
    
    return x_train, x_test, y_train, y_test
    ### Train data ###
    # TrainData = x_train.reshape(-1,614,8,1)
    # forTrain = data.iloc[:614] # predict variable
    # for i in forTrain.values:
        # print(type(i)) numpy.ndarray
        # TrainData = np.arange(4912).reshape((614,8))
    # age = np.asarray(forTrain['Age'])
    # print(type(age)) <class 'pandas.core.series.Series'>
    # TrainData = [preg, glu, skin, insu, bmi, dpf, age]   

    # global train_target
    # target_1 = target.iloc[:614]
    # train_target = np.asarray(target_1)

    ### Test data ###
    # global TestData
    # forTest = data.iloc[614:]
    # for i in forTest.values:
    #     TestData = np.arange(1232).reshape((154,8))
    # pregT = np.asarray(forTest['Pregnancies'])   
    # TestData = [pregT, gluT, skinT, insuT, bmiT, dpfT, ageT] 

    # global test_target
    # target_2 = target.iloc[614:]
    # test_target = np.asarray(target_2)

def main():

    source = pre()
    trainx = source[0]
    testx = source[1]
    trainy = source[2]
    testy = source[3]

    start_time = time.clock()
    pred_cm = []
    actual_cm = []
    i = 0
    correct = 0
    for item in testx:
        # the second parameter can be 1,2,3. Each number represnts one distance method
        pred = predict(10, 3, trainx, trainy, item)
        pred_cm.append(pred)
        actual_cm.append(testy[i])
        if (pred == testy[i]).any():
            correct += 1

        acc = (correct / (i+1)) * 100
        print('test variable['+str(i)+']', '\tpredict:', pred, '\torigin:', testy[i], '\taccuracy:', str(round(acc, 2))+'%')
        i += 1
    
    print("Training Time: %.2f seconds" % (time.clock() - start_time))

    # True negative, False positive, False negative, True positive
    print("Complete Confusion Matrix: \n", confusion_matrix(actual_cm, pred_cm))
    tn, fp, fn, tp = confusion_matrix(actual_cm, pred_cm).ravel()
    print('True negative:', tn, '\tFalse positive:', fp, '\tFalse negative:', fn,'\tTrue positive:', tp)
    print('Accuracy:', accuracy_score(actual_cm, pred_cm))
    

def Euclidean_distance(v1, v2):
    result = sum((v1 - v2) ** 2)
    return math.sqrt(result)

# a special case of the Minkowski distance where p goes to infinity
def Chebychev_distance(v1, v2):
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    result = distance.chebyshev(v1, v2)
    return result

def Manhattan_distance(val1, val2):
    result = sum(abs(val1 - val2))
    return result

def Majority_Target(target):
    count = defaultdict(int)
    for element in target:
        count[element] += 1
 
    majority = max(count.values())
    for key, value in count.items():
        if value == majority:
            return key 

def predict(k, select, trainData, trainTarget, testData):

    if select == 1:
        distances = [ (Euclidean_distance(testData, data), target) for (data, target) in zip(trainData, trainTarget) ]
    elif select == 2:
        distances = [ (Chebychev_distance(testData, data), target) for (data, target) in zip(trainData, trainTarget) ]
    elif select == 3:
        distances = [ (Manhattan_distance(testData, data), target) for (data, target) in zip(trainData, trainTarget) ]
    
    sorted_distances = sorted(distances, key=lambda distance, : distance)
    k_targets = [target for (_, target) in sorted_distances[:k]]
    return Majority_Target(k_targets)


if __name__ == '__main__':
    main()
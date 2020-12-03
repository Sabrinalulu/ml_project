import time
import math
import itertools
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def main():

    file = pd.read_csv("digit-recognizer/train.csv")
    # divide training data into features and labels
    data = file.drop(columns=['label'])
    label = file['label']
    # a 28*28 pixels image 
    # print(data.iloc[0])
    # list 0-4's image
    # print(data.iloc[:5])
    # 80%: train, 20%: test; total: 42000 -> train: 33600, test: 8400
    # forTrain = data.iloc[:33600]
    # label_1 = label.iloc[:33600]
    # train_label = np.asarray(label_1)
    # print(train_data.values[0]) the first row, label 0
    # for i in forTrain.values:
        # print(type(i)) numpy.ndarray
        # TrainData = np.reshape(i, (28,28))
        # print(type(TrainData))
        # plt.imshow(TrainData, cmap = 'gray')
        # plt.show()

    X = data.values[:5000]
    Y = label.values[:5000]
    # split the data
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
    
    TrainData = x_train.reshape(-1,28,28,1)
    TestData = x_test.reshape(-1,28,28,1)


    # Predict and print the accuracy
    # i: index of test array
    # correct: the count of correct items 
    # TestData: the 28*28 pixels image's collection (item: one row one image)
    # test_label: true label value
    start_time = time.clock()
    pred_cm = []
    actual_cm = []
    i = 0
    correct = 0
    for item in TestData:
        pred = predict(15, TrainData, y_train, item)
        pred_cm.append(pred)
        # print('pred_cm', pred_cm)
        actual_cm.append(y_test[i])
        # print('actual_cm', pred_cm)
        if (pred == y_test[i]).any():
            correct += 1

        acc = (correct / (i+1)) * 100
        print('test image['+str(i)+']', '\tpredict:', pred, '\torigin:', y_test[i], '\taccuracy:', str(round(acc, 2))+'%')
        i += 1
    
    draw = confusion_matrix(actual_cm, pred_cm)
    print("Complete Confusion Matrix: \n", draw)
    print("Training Time: %.2f seconds" % (time.clock() - start_time))
    # Plot Confusion Matrix Data as a Matrix
    plt.matshow(draw)
    plt.title('Confusion Matrix for Test Data')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def Euclidean_distance(p1, p2):
    # Finds the distance between 2 images
    # numpy handles the element-wise computation
    # result = sum((img1 - img2) ** 2)
    # return math.sqrt(result)
    return np.sqrt(np.sum((p1-p2)**2))

def Majority_label(labels):
    # defaultdict(): add new key if the key doesn't exist
    count = defaultdict(int) # dist
    # interate all labels and store them into a dictionary (key:label(element)|value:count) 
    for element in labels:
        count[element] += 1

    # max(): find a lable with a biggest count value  
    majority = max(count.values())
    # return the key (label) having the biggest count value
    for key, value in count.items():
        if value == majority:
            return key # the majority label

# predict the new data label by training trained data's label
# parameter k means k nearest neighbors
# trainImage: 28*28 pixels image from the train dataset (matches with the global variable: TrainData)
# trainLabel: the label list from the train dataset (matches with the global variable: train_label)
def predict(k, trainImage, trainLabel, testImage):
    
    # distances contains tuples of (distance, label)
    # zip(): Join two tuples together
    # a = ("John", "Charles"); b = ("Jenny", "Christy") => zip(a, b) => (('John', 'Jenny'), ('Charles', 'Christy'))
    # image: train's data of one image; train's label of one row  
    # create a tuple with the distance calculated from Euclidean function and train label 
    distances = [ (Euclidean_distance(testImage, image), label) for (image, label) in zip(trainImage, trainLabel) ]
    # sort the distances list by distance
    # distances: list[(distance, label)]
    sorted_distances = sorted(distances, key=lambda x : x[0])
    # print('Sorted:', sorted_distances)
    # pick first k labels (k-nearest) from the sorted list; (_,...): ignores part of the function
    k_labels = [label for (_, label) in sorted_distances[:k]]
    # send the list of k-nearest label to majority function => the majority label we got means the image is categorized to that label
    # print('kLabel: ', k_labels)
    return Majority_label(k_labels)

if __name__ == '__main__':
    global TrainData
    global TestData
    global train_label
    global test_label
    main()
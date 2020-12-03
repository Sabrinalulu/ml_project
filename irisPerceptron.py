import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from matplotlib.colors import ListedColormap

# evaluate an algorithm by using a stratified cross validation split       
def evaluate(X, y, lr, epoch):
    confis = []
    confiv = []
    scores = []
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    for train_index, test_index in kfold.split(X, y):
        train_data, test_data = X[train_index], X[test_index]
        train_target, test_target = y[train_index], y[test_index]
        
        classifier = Perceptron(learning_rate=lr, epochs=epoch)
        classifier.fit(train_data, train_target)
        predicted = classifier.predict(test_data)
        print('predict:', predicted)
        print('actual:',test_target)
        metric = confidence_metric(test_target, predicted)
        print('metric:', metric)
        confis.append(metric[0])
        confiv.append(metric[1])
        accuracy = accuracy_metric(test_target, predicted)
        scores.append(accuracy)
    
    setosa = sum(confis)/len(confis)
    virginica = sum(confiv)/len(confiv) 
    print('setosa-', setosa, 'virginica-', virginica)
    return scores, setosa, virginica

def load_file(data):
    global X,Y
    df = pd.read_csv(data, header=None)
    setosa = df.iloc[:50]
    virginica = df.iloc[100:]
    combine = setosa.append(virginica, ignore_index=True)
    
    tempy = combine.iloc[:,4].values
    Y = np.where(tempy == 'Iris-setosa', 1, -1)
    
    X = combine.iloc[:,[0,1,2,3]].values
    
    # split the data
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

    return x_train, x_test, y_train, y_test
    # # 0:sepal length; 2:petal length
    # plt.scatter(X[:50, 0], X[:50, 2], color='red', marker='o', label='setosa-length')
    # plt.scatter(X[50:100, 0], X[50:100, 2], color='blue', marker='x', label='virginica-length')
    # # 1:sepal width; 3:petal width
    # plt.scatter(X[:50, 1], X[:50, 3], color='red', marker='^', label='setosa-width')
    # plt.scatter(X[50:100, 1], X[50:100, 3], color='blue', marker='D', label='virginica-width')
    # plt.xlabel('sepal')
    # plt.ylabel('petal')   
    # plt.legend(loc='upper left')
    # plt.show()    
    
class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=10):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.errors = []
        self.weights = None
        
    def fit(self, x, y):
        # shape attribute for numpy arrays returns the dimensions of the array => x.shape (3 rows, 4 columns) => x.shape[0]: 3 
        self.weights = np.zeros(1+ x.shape[1])
        for i in range(self.epochs):
            error = 0
            for item, target in zip(x,y):
                update = self.learning_rate * (target - self.predict(item))
                self.weights[1:] += update * item
                self.weights[0] += update
                error += int(update != 0)
            self.errors.append(error)
        return self
                
    # summing the given matrix inputs and their corresponding weights
    def neroun_output(self, x):
        return np.dot(x, self.weights[1:]) + self.weights[0]
    
    # predict method for predicting the classification of data inputs
    def predict(self, x):
        return np.where(self.neroun_output(x) >= 0.0, 1, -1)

def confidence_metric(actual, activation):
    metricS = 0
    metricV = 0
    countS = 0
    countV = 0
    for i in range(len(actual)):
        if actual[i] > 0:
            countS += 1
            metricS += activation[i]*100
        else:
            countV += 1
            metricV += -activation[i]*100

    print('metricS: ', metricS, 'metricV: ', metricV)
    setosa = metricS/countS
    virginica = metricV/countV
    
    return setosa, virginica

def plot_confidence_metric(s, v):
    plt.plot(epo, s, 'o', label = "iris-Setosa")
    plt.plot(epo, v, '^', label = "iris-Virginica")
    plt.title('Confidence Metric')
    plt.xlabel('Training epoch')
    plt.ylabel('Confidence (%)')
    plt.legend()
    plt.xticks(np.arange(0, 21, 1))
    plt.yticks(np.arange(-100, 101, 50))
    
    plt.show()

def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
    alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)

def plot_classification_accuracy(accuracy, epoch):    
    plt.plot(epoch, accuracy[0], '-', label = "learning_rate: 0.00005")
    plt.plot(epoch, accuracy[1], '--', label = "learning_rate: 0.001")
    plt.plot(epoch, accuracy[2], '-.', label = "learning_rate: 0.005")
    plt.title('Classification Accuracy')
    plt.xlabel('Training epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.xticks(np.arange(0, 21, 1))
    plt.yticks(np.arange(0, 101, 10))
    
    plt.show()

def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0
          
if __name__ == '__main__':
    plotMs = []
    plotMv = []
    global classifier
    accuracy = []
    global epo
    epo = list(range(1, 21))
    # lr = [0.00005, 0.001, 0.005]
    lr = [0.001]
    
    source = load_file('iris.data')
    trainx = source[0]
    testx = source[1]
    trainy = source[2]
    testy = source[3]
    
    for rate in lr:
        subacc = []       
        for e in epo:
            classifier = Perceptron(learning_rate=rate, epochs=e)
            classifier.fit(trainx, trainy)
            Target = classifier.predict(testx)            
            acc = accuracy_metric(testy, Target) 
            subacc.append(acc)
            print('learning_rate: ', str(rate), '\tepochs: ', str(len(classifier.errors)), '\taccuracy: ', str(acc))
            scores = evaluate(X, Y, rate, e)
            plotMs.append(scores[1])
            plotMv.append(scores[2])
            print('evaluate:', scores[0])
            print('10-fold-validation Mean Accuracy: %.2f%%' % (sum(scores[0])/float(len(scores[0]))))
            print()
        accuracy.append(subacc)
    
    # plot_classification_accuracy(accuracy, epo)
    plot_confidence_metric(plotMs, plotMv)
    # Classifier = Perceptron(learning_rate=0.001, epochs=20)
    # Classifier.fit(trainx, trainy)
    
    # Showing the final results of the perceptron model.
    # plot_decision_regions(trainx, trainy, classifier=classifier)
    # plt.xlabel('petal length [cm]')
    # plt.ylabel('petal width [cm]')
    # plt.legend(loc='upper left')
    # plt.show()
    
    # plt.plot(range(1, len(classifier.errors) + 1), classifier.errors, marker='o')
    # plt.xlabel('Epochs')
    # plt.ylabel('Number of misclassifications')
    # plt.show()
    
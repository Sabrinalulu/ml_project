import numpy as np
import pandas as pd
import lRpre as pre
import seaborn as sbs
import matplotlib.pyplot as plt
from numpy.linalg import inv
# from sklearn.metrics import mean_squared_error

def main():
    
    df_1 = data[['G1', 'G2', 'failures', 'higher', 'school', 'studytime', 'Fedu', 'Medu', 'Dalc', 'Walc']] 
    df_2 = data[['age', 'Mjob', 'address', 'reason', 'internet', 'sex', 'freetime', 'traveltime', 'health', 'romantic', 'goout']]
    df_3 = data[['famrel', 'famsize', 'Pstatus', 'Fjob', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'absences']]
    # print(type(data)) dataframe
    # print(type(target)) series
    Target = pd.DataFrame(target)
    
    result1 = linear_matrix(df_1, Target)
    result2 = linear_matrix(df_2, Target)
    result3 = linear_matrix(df_3, Target)
    result4 = linear_matrix(data, Target) # alldata
    print('df_1: ', result1, '\tdf_2: ', result2, '\tdf_3: ', result3, '\tall:', result4)
    # initializing inputs and outputs
    # X = fortrain['G2'].values
    # Y = train_target
    # output = linear_regression(X, Y)
    # print('RMSE: ', output)
    # plt.show()
    
def linear_matrix(X, Y):
    # linear least squares, b: coefficient
    b = inv(X.T.dot(X)).dot(X.T).dot(Y)
    xdata = X.values
    ydata = Y.values
    
    # predict using coefficients xβ 
    yhat = xdata.dot(b)
    # e(β) = y − xβ 
    e = ydata - yhat
    
    MSE = 0.0
    MSEtest = 0.0
    MSE = sum(e**2)/649 # n=649
    print(MSE)   
    # MSEtest = mean_squared_error(ydata,yhat) 
    # print(test)
    return MSE
        
    
def linear_regression(X, Y):
    # mean of inputs and outputs
    x_mean = np.mean(X)
    y_mean = np.mean(Y)
    n = len(X)
    
    # using the formula to calculate the b1 and b0
    numerator = 0
    denominator = 0
    for i in range(n):
        numerator += (X[i] - x_mean) * (Y[i] - y_mean)
        denominator += (X[i] - x_mean) ** 2
    b1 = numerator / denominator
    b0 = y_mean - (b1 * x_mean)
    #printing the coefficient
    print(b1, b0)
    
    rmse = 0
    for i in range(n):
        y_pred=  b0 + b1* X[i]
        rmse += (Y[i] - y_pred) ** 2
    
    rmse = np.sqrt(rmse/n)
    
    #plotting values 
    x_max = np.max(X) 
    print(x_max)
    x_min = np.min(X)
    print(x_min)
    #calculating line values of x and y 
    x = np.linspace(x_min, x_max, 500)
    y = b0 + b1 * x
    #plotting line 
    plt.plot(x, y, color='#00ff00', label='Linear Regression')
    #plot the data point
    plt.scatter(X, Y, color='#ef5423', label='Data Point')
    # x-axis label
    plt.xlabel('G2')
    #y-axis label
    plt.ylabel('G3')
    plt.legend()
    
    return rmse
    

if __name__ == '__main__':
    global data
    data = pre.train_cols
    global target
    target = pre.y_col
    main()

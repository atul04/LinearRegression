# @Author: Atul Sahay <atul>
# @Date:   2018-08-07T18:09:32+05:30
# @Email:  atulsahay01@gmail.com
# @Filename: model1.py
# @Last modified by:   atul
# @Last modified time: 2018-08-19T22:45:18+05:30

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
import math

"""
You are allowed to change the name of function arguments as per your convinience,
but it should be meaningful.

Like : y, y_train, y_test, output_var, target, output_label, ... are accepted
but do not keep it like : abc, a, b, etc.

Same applies to variable naming also

"""


# Provide Map for non int data (post_day,basetime_day)
def to_map(data_set):
    data_set = pd.concat([data_set,pd.get_dummies(data_set['post_day'], prefix='post_day')],axis=1)

    # now drop the original
    data_set.drop(['post_day'],axis=1, inplace=True)

    data_set = pd.concat([data_set,pd.get_dummies(data_set['basetime_day'], prefix='basetime_day')],axis=1)

    # now drop the original
    data_set.drop(['basetime_day'],axis=1, inplace=True)

    return data_set

# Feature scaling , here I have used min_max which has a unique property of 0 mean and 1 standard deviation, plot is normal distribution
def to_normalize(data_set):
    global train_mean, train_std
    train_mean = data_set.mean()
    train_std = data_set.std()
    data_set = (data_set - data_set.mean())/data_set.std()
    # data_set = (data_set - data_set.min())/(data_set.max() - data_set.min())
    return data_set

# Split in x and y
def split(data):
    x_train = data.iloc[:,:-1]
    y_train = data.iloc[:,-1]

    return x_train, y_train

# Provide features set and target set
def get_features(file_path):
	# Given a file path , return feature matrix and target labels
    data = pd.read_csv(file_path)
    #data = data.sample(frac=1).reset_index(drop=True)
    return split(data)

# mean square Error
def mean_square_error(x,y_label,theta,lam):
    m = x.shape[0]
    hypothesis = np.dot(x, theta)
    loss = hypothesis - y_label
    cost = np.sum(loss ** 2)
    cost_reg = cost + lam*np.sum(np.power(theta,2))
    return cost_reg/m
############################################# Another Variant of Gradient##################
def gradientDescent(x, y_label, theta, alpha, Iterations,lam,x_valid,y_valid,error):
    m = x.shape[0]
    xTrans = x.transpose()
    cost_valid_old = math.inf
    train_cost_old = math.inf
    # print(x_valid.shape[0])
    best_theta = theta
    for i in range(Iterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y_label
        train_cost_curr = mean_square_error(x,y_label,theta,lam)
        cost_valid_curr = mean_square_error(x_valid,y_valid,theta,lam)
        if(i%100==0):
            print("\nIteration %d | Train_cost_reg %0.18f \nCost_valid_old %0.18f \nCost_valid_curr %0.18f" % (i, train_cost_curr,cost_valid_old,cost_valid_curr))

        # breaking iteration if breaking criteria met
        if(cost_valid_curr>cost_valid_old):
            print("\nStopped due to cost_valid criteria met")
            break
        if((train_cost_old-train_cost_curr)<error):
            print("\nStopped due to train_error criteria met")
            break
        cost_valid_old = cost_valid_curr
        train_cost_old = train_cost_curr
        gradient = np.dot(xTrans, loss)
        # print(gradient)
        gradient = gradient + lam*theta
        theta = theta - 2*alpha*gradient/m
        best_theta = theta
    print("\nIteration %d | Cost_reg %0.18f \nCost_valid_old %0.18f \nCost_valid_curr %0.18f\
    \nlambda: %0.6f \tlearn:%0.6f " % (i, train_cost_curr,cost_valid_old,cost_valid_curr,lam,alpha))
    print("\nDiff:%0.18f"%abs(train_cost_old-cost_valid_old))
    #returning the previous iteration weight
    return best_theta
#######################################################################


def generate_output(phi_test, weights):
# 	# writes a file (output.csv) containing target variables in required format for Kaggle Submission.
    print("Generating the output file:--")
    df = pd.DataFrame(columns=['target'])
    # y_P_list = []
    # idList = [ i for i in range(int(len(phi_test)))]
    for i in range(int(len(phi_test))):
        # print(phi_test[i])
        y_pred = weights.dot(phi_test[i])
        # y_P_list.append(y_pred)
        # print(y_pred)
        if(y_pred<0):
            y_pred = 0
        # y_pred = int(round(y_pred))
        df.loc[i] = np.array([y_pred])
    df.to_csv('output1.csv')
    print("Done")

#################################################### Task -3 #############################

# mean square Error - norm P
def mean_square_errorP(x,y_label,theta,lam,p):
    m = x.shape[0]
    hypothesis = np.dot(x, theta)
    loss = hypothesis - y_label
    cost = np.sum(loss ** 2)
    cost_reg = cost + lam*(np.sum(np.power(theta,p)))**(1/p)
    return cost_reg/m

# Gradient descent for p norm
def gradientDescentP(x, y_label, theta, alpha, Iterations,lam,x_valid,y_valid,error,p):
    m = x.shape[0]
    xTrans = x.transpose()
    cost_valid_old = math.inf
    train_cost_old = math.inf
    # print(x_valid.shape[0])
    best_theta = theta
    for i in range(Iterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y_label
        train_cost_curr = mean_square_errorP(x,y_label,theta,lam,p)
        cost_valid_curr = mean_square_errorP(x_valid,y_valid,theta,lam,p)
        if(i%100==0):
            print("\nIteration %d | Train_cost_reg %0.18f \nCost_valid_old %0.18f \nCost_valid_curr %0.18f" % (i, train_cost_curr,cost_valid_old,cost_valid_curr))

        # breaking iteration if breaking criteria met
        if(cost_valid_curr>cost_valid_old):
            print("\nStopped due to cost_valid criteria met")
            break
        if((train_cost_old-train_cost_curr)<error):
            print("\nStopped due to train_error criteria met")
            break
        cost_valid_old = cost_valid_curr
        train_cost_old = train_cost_curr
        gradient = np.dot(xTrans, loss)
        # print(gradient)
        theta_P_norm = (np.sum(np.power(theta,p)))**((1-p)/p)
        gradient = gradient + lam*theta_P_norm*theta
        theta = theta - 2*alpha*gradient/m
        best_theta = theta
    print("\nIteration %d | Cost_reg %0.18f \nCost_valid_old %0.18f \nCost_valid_curr %0.18f\
    \nlambda: %0.6f \tlearn:%0.6f\t P-Norm = %d" % (i, train_cost_curr,cost_valid_old,cost_valid_curr,lam,alpha,p))
    print("\nDiff:%0.18f"%abs(train_cost_old-cost_valid_old))
    #returning the previous iteration weight
    return best_theta


################################################### Task 3 ends here ######################

################################################## Task 4 starts(To report mse on basis of x) ##########################
###### sigmoidal function ############
def sigmoidal(value):
    # print(value)
    a = math.exp(value) + 1
    return 1/a

##### gaussian function #############
def gaussian(x, mu, sig):
    # print(x,mu,sig)
    return np.exp(-np.power(x - mu, 2) / (2 * np.power(sig, 2)))


################################################# Task 4 Ends ##########################################################
#
#
# def task1(phi, y):
#
#
# 	return w_final
#
#
def generatePlots(x, y, theta):
# 	"""
# 	 generates and saves plots of top three features with target variable.
# 	 Note: Procedure to obtain top features is important
# 	"""
    theta = pd.Series(theta).apply(abs)
    top3 = theta.nlargest(3)
    indices = list(top3.index[:])
    top3_features = [x.columns[i] for i in indices]
    print("Top 3 Features:")
    for i,j in enumerate(top3_features):
        print("%d. %s"%(i+1,j))
    for i in top3_features:
        # plotting points as a scatter plot
        plt.scatter(x[i], y, label= "likes", color= "red",
            marker= "*", s=10)
        # x-axis label
        plt.xlabel(i)
        # frequency label
        plt.ylabel("Target")
        # plot title
        plt.title('Scatter plot!')
        # showing legend
        plt.legend()

        # function to show the plot
        plt.show()
# def task3(phi, y , lamda, p):
#
#
#
# 	return w_final
#
# def task4(phi, y):
# 	#Try out two different basis functions
#
# 	return w_final
#
#
#
# def task5():
# 	# get best weights
# 	# Use any optimization / modification to linear regression you can
#
#
# 	return w_best
#
#

def main():
    """
    Calls functions required to do tasks in sequence
    say :
    	train_file = first_argument
    	test_file = second_argument
    	x_train, y_train = get_features();
    	task1();task2();task3();.....

    """
    global train_mean, train_std
    # pow = 1
    train_file = sys.argv[1]
    # train_file = '/home/atul/college/cs725/Assignment/train.csv'
    test_file = sys.argv[2]
    # test_file = '/home/atul/college/cs725/Assignment/test.csv'
    print("Reading Files...")
    x_test = pd.read_csv(test_file)

    x_train, y_train = get_features(train_file)
    print("Done")
    ################################## Mapping days to one hot vector################
    x_train = to_map(x_train)
    x_test = to_map(x_test)
    ##############################################################################

    ####################### Normalizing the data points##############################
    x_train = to_normalize(x_train)
    #Using basis function on x_train
    #x_train= np.power(x_train,pow)
    #################################
    #print(x_train)
    # x_train = to_normalize(x_train)
    # x_train = x_train.fillna(0)

    # To take validation set out in proportion of 20-80 #############################
    indexes = int(0.80*x_train.shape[0])
    x_train, x_valid = x_train.iloc[:indexes], x_train.iloc[indexes:]
    y_train, y_valid = y_train.iloc[:indexes], y_train.iloc[indexes:]
    # print(x_valid.shape,x_train.shape)
    # print(y_valid.shape,y_train.shape)
    ####################### Normalizing the data points##############################
    x_test = to_normalize(x_test)
    x_test.promotion = x_test.promotion.fillna(0)

    # using basis function on x
    #x_test = np.power(x_test,pow)
    #x_test = to_normalize(x_test)
    #x_test = x_test.fillna(0)
    ####################################################

    #Appending a series of Ones for bias in x_train
    ones = np.ones(x_train.shape[0])
    x_train.insert(loc=x_train.shape[1], column='Ones', value=ones)

    #Appending a series of Ones for bias in x_valid
    ones = np.ones(x_valid.shape[0])
    x_valid.insert(loc=x_valid.shape[1], column='Ones', value=ones)

    #Appending a series of ones for bias in x_test
    ones = np.ones(x_test.shape[0])
    x_test.insert(loc=x_test.shape[1], column='Ones', value=ones)
    ################################################################
    # print(x_test)

    ############################## Weights are intialised####################
    weights = np.ones(x_train.shape[1])

    '''only for intermedaite I have used w8
    #weights = np.asarray([  1.17060638e-04,   2.35004099e-05,  -1.30034190e-04,
            -6.00073925e-07,   1.44797284e-04,  -1.21922294e-04,
            -2.72050584e-04,   5.26221663e-04,   4.09867733e-05,
            -4.16564801e-04,   5.18866714e-05,   1.04937370e-04,
             7.10802144e-05,  -1.04007447e-04,  -1.19510175e-04,
             1.11695455e-04,  -1.79680686e-05,  -2.72377937e-05,
            -7.60613142e-06,  -3.40653639e-05,   1.54812658e-05,
             2.62793430e-06,  -1.78809030e-07,   9.99996409e-01,
             2.75190075e-17])

    weights = np.asarray([-2.4939705517793422, -0.53974573559169647, 1.0491918338698636, 0.16653430461185/
684, 0.45656696338442215, 5.0809184369857272, -0.24554144887005949, 1.71871231073/
11848, -1.2673354445690064, -1.9000018027212746, -1.3791808114877206, 3.123807206/
5368114, 0.24339423293013179, 13.572180704418875, -0.99178304418971552, -3.255300/
5972305851, -0.18079819497701097, -4.2608779675490114, 0.29195264761876505, 3.180/
6330096196036, 0.077131364057355919, 0.72720412498405451, 0.15508153191195595, 0.0010186968442210439, 6.8950000000000253])
'''
#########################################################################


    #hyperparameters used
    learn = 0.001
    iterations = 100000
    lam = 10000
    error = 10**(-18)
    p = 6
    print("::::::::::::::LINEAR REGRESSION::::::::::::::::")
    print("1. L2 regularisation")
    print("2. P-Norm regularisation")
    print("3. L2 regularisation with basis")
    print("4. Optimised one(P-6 with sigmoidal basis)")
    choice = int(input("Your choice: "))

    if(choice==1):
        lam = float(input("lambda value : "))
        learn = float(input("Learn rate :"))
        iterations = int(input("No. of iterations: "))
        ###### Task -1
        weights = gradientDescent(x_train.values, y_train.values, weights, learn, iterations,lam,x_valid,y_valid,error)
    elif(choice==2):
        p = int(input("P-Norm [4,6]: "))
        lam = float(input("lambda value : "))
        learn = float(input("Learn rate :"))
        iterations = int(input("No. of iterations: "))
        ###### TASK -3
        #
        weights = gradientDescentP(x_train.values, y_train.values, weights, learn, iterations,lam,x_valid,y_valid,error,p)
    elif(choice==3):
        print("1. Inverse Sigmoidal")
        print("2. Gaussian Function")
        basis = int(input("Your Choice: "))
        lam = float(input("lambda value : "))
        learn = float(input("Learn rate :"))
        iterations = int(input("No. of iterations: "))

        ###### TASK -4
        if(basis==1):
            print("Inverse Sigmoidal")
            x_train.iloc[:,:-1] = x_train.iloc[:,:-1].applymap(sigmoidal)
            x_valid.iloc[:,:-1] = x_valid.iloc[:,:-1].applymap(sigmoidal)
            x_test.iloc[:,:-1] = x_test.iloc[:,:-1].applymap(sigmoidal)
        elif(basis==2):
            print("Gaussian Function")
            x_tr_m, x_tr_s = x_train.mean(), x_train.std()
            x_v_m, x_v_s = x_valid.mean(), x_valid.std()
            x_te_m, x_te_s = x_test.mean(), x_test.std()
            for i in range(len(x_tr_m)):
                x_train.iloc[:,i] = x_train.iloc[:,i].apply(lambda x: gaussian(x,x_tr_m[i],x_tr_s[i]))
                x_valid.iloc[:,i] = x_valid.iloc[:,i].apply(lambda x: gaussian(x,x_v_m[i],x_v_s[i]))
                x_test.iloc[:,i] = x_test.iloc[:,i].apply(lambda x: gaussian(x,x_te_m[i],x_te_s[i]))
            x_train = x_train.fillna(0)
            x_valid = x_valid.fillna(0)
            x_test = x_test.fillna(0)
        else:
            exit("Error: Invalid Input")
        weights = gradientDescent(x_train.values, y_train.values, weights, learn, iterations,lam,x_valid,y_valid,error)
    elif(choice==4):
        print("Basis: Inverse Sigmoid Norm: P=6")
        lam = float(input("lambda value : "))
        learn = float(input("Learn rate :"))
        iterations = int(input("No. of iterations: "))

        ##### TASK -5
        x_train.iloc[:,:-1] = x_train.iloc[:,:-1].applymap(sigmoidal)
        x_valid.iloc[:,:-1] = x_valid.iloc[:,:-1].applymap(sigmoidal)
        x_test.iloc[:,:-1] = x_test.iloc[:,:-1].applymap(sigmoidal)

        p = 6
        weights = gradientDescentP(x_train.values, y_train.values, weights, learn, iterations,lam,x_valid,y_valid,error,p)
    else:
        exit("Error: Invalid Input")
    ###### Generate target file
    generate_output(x_test.values,weights)
    choice = input("Want to see Top 3 Features: [y/n]")
    if(choice=="y"):
        ###### TASK -2
        generatePlots(x_train, y_train, weights)




#################### Driver Function
if __name__ == '__main__':
    main()

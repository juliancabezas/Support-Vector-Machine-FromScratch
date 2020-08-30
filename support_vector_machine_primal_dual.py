###################################
# Julian Cabezas Pena
# Introduction to Statistical Machine Learning
# University of Adelaide
# Assingment 1
# Support Vector Machine Classifer
####################################


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix

import cvxopt as cvxopt
import cvxopt.solvers as solvers

import os

def svm_train_primal(data_train, label_train , regularisation_para_C):

    # Get the dimensions of the data
    n_samples = np.shape(data_train)[0]
    m_features = np.shape(data_train)[1]

    # First (Top equation) of the optimization problem

    # The vector X will be optimized, this vector corresponds to [weights,intercept,Epsilon(slack variable)]

    # According to the CVXOPT apy the first term is (1/2) * X.T P X
    # P will be a identity matrix in the top and zeroes in the bottom, that way we will get ||weights||
    P = np.zeros((m_features+n_samples+1,m_features+n_samples+1))
    P[:m_features,:m_features] = np.identity(m_features)

    # The q term belong to the second bart of the top equation
    # The first part is zeroes, corresponding to the weights and the intercept, 
    # and the second part to the slack variables, that are multiplied by the cost
    q_zeros = np.zeros((m_features+1,1))
    q_slack = 1/n_samples * regularisation_para_C* np.ones((n_samples,1))
    q = np.vstack([q_zeros,q_slack])

    # Second equation

    # G will be multiplied by  X like this: Gx < h
    # Then I have to make sure the y(w.T x + b) equation is represented in the G array
    G = np.zeros((2*n_samples, m_features+1+n_samples))

    # yX
    G[:n_samples,0:m_features] = label_train @ data_train
    # One extra column for the label alone (it will be multiplied with the intercept b)
    G[:n_samples,m_features] = label_train.T

    # The right part of the matrix will be identity matrices, they will be multiplied by the slack variable
    # (We are passing the slack variables to the left side of the equation)
    G[:n_samples,m_features+1:]  = np.identity(n_samples)
    G[n_samples:,m_features+1:] = np.identity(n_samples)
    
    # The h is the "1" in the second equation
    h = np.zeros((2*n_samples,1))
    h[:n_samples] = 1.0

    # Here I reverse the inequality, making it Gx > h
    G = -G
    h = -h

    # Convert to the CVROPT matrix type
    P = cvxopt.matrix(P)
    q = cvxopt.matrix(q)
    G = cvxopt.matrix(G)
    h = cvxopt.matrix(h)

    # Quadratic solver, in this case we will not use the bottom equation, hence A and b are not needed
    print("Starting quadratic solver!")
    qp_solution = solvers.qp(P, q, G, h)

    # Get the weights
    weights = np.array(qp_solution['x'][:m_features])

    # Get the intercept
    intercept = qp_solution['x'][m_features]


    return np.append(weights,np.array(intercept))




def svn_predict_primal(data_test,label_test,svn_model):

    # Extract the wrights and intercept from the ssvn model output
    weights = svn_model[:(len(svn_model) - 1)]
    intercept = svn_model[len(svn_model) - 1]

    # Calculate the weighted sum
    weighted_sum = np.dot(data_test, weights) + intercept

    #Predict using the sign function, this will be -1 or +1
    prediction = np.sign(weighted_sum)

    # Get the accuracy of the prediction
    acc = accuracy_score(label_test,prediction)

    return acc



def svm_train_dual(data_train, label_train , regularisation_para_C):
    
    # Get the dimensions of the data
    n_samples = np.shape(data_train)[0]
    m_features = np.shape(data_train)[1]

    # Gram matrix
    #Gram = np.zeros((n_samples, n_samples))

    #for i in range(n_samples):
    #    for j in range(n_samples):
    #        Gram[i,j] = np.dot(data_train[i], data_train[j])


    # Fistly I have to calculate the Gram matrix
    Gram = data_train.dot(data_train.T)

    # First equation
    # It is the sum of (y_i*y_j*<x_1,x_j>)

    P = np.outer(label_train,label_train) * Gram

    print(P)

    # q is just a column of -1, ad q.T is multiplied by the alphas in the SVXOPT API
    q = np.ones(n_samples) * -1

    print(q)

    # Second equation

    # G has a diagonal matrix in the top and an identity matrix in the bottom, as only the 
    tmp1 = np.diag(np.ones(n_samples) * -1)
    tmp2 = np.identity(n_samples)
    G = np.vstack((tmp1, tmp2))

    print(G)

    tmp1 = np.zeros(n_samples)
    tmp2 = np.ones(n_samples) * regularisation_para_C * (1/n_samples)
    h = np.hstack((tmp1, tmp2))

    print(h)

    # Third equation
    # Corresponds to the sum of alpha_i*y_i= 0, so A will only by the labels and b a column of zeroes
    A = cvxopt.matrix(label_train, (1,n_samples))
    b = np.array([[0]])

    

    # Convert to the CVROPT matrix type
    P = cvxopt.matrix(P)
    q = cvxopt.matrix(q)
    G = cvxopt.matrix(G)
    h = cvxopt.matrix(h)
    #A = cvxopt.matrix(A)
    b = cvxopt.matrix(0.0)

    # solve QP problem
    print("Starting the solver")
    qp_solution = cvxopt.solvers.qp(P, q, G, h, A, b)

    # Vector with lagrange multipliers, the solution of the quadratic equation
    lagrange = np.ravel(qp_solution['x'])

    # Support vectors have non zero lagrange multipliers
    sup_vect_bool = lagrange > 0.00001
    ind = np.arange(len(lagrange))[sup_vect_bool]
    lagrange = lagrange[sup_vect_bool]
    sup_vect = data_train[sup_vect_bool]
    sup_vect_y = label_train[sup_vect_bool]

    # Weights
    weights = np.zeros(m_features)
    
    # The weights are the sum of the product of alpha_i, y_i and x_i for the support vectors
    for i in range(len(lagrange)):
        sum = lagrange[i] * sup_vect_y[i] * sup_vect[i]
        weights = weights + sum

    # Intercept (b)
    intercept = 0
    for i in range(len(lagrange)):
        intercept = intercept + sup_vect_y[i]
        intercept =  intercept - np.sum(lagrange * sup_vect_y * Gram[ind[i], sup_vect_bool])

    intercept = intercept / len(lagrange)



    print(weights)
    print(intercept)
    print("Ready")

    return np.append(weights,np.array(intercept))

def svn_predict_dual(data_test,label_test,svn_model):


    # Extract the wrights and intercept from the ssvn model output
    weights = svn_model[:(len(svn_model) - 1)]
    intercept = svn_model[len(svn_model) - 1]

    # Calculate the weighted sum
    weighted_sum = np.dot(data_test, weights) + intercept

    #Predict using the sign function, this will be -1 or +1
    prediction = np.sign(weighted_sum)

    # Get the accuracy of the prediction
    acc = accuracy_score(label_test,prediction)

    return acc


# Main function, it will preprocess the data, create train and test data,
# tune the parameters of the Random forest and evaluate the final model
def main():

    #----------------------------------------------
    # Data reading and preprocessing

    # Train data

    # Read the database using pandas
    data = pd.read_csv("Input_Data/train.csv")

    # Recode the output column to get -1 and 1 output values
    data.iloc[:, 0] = np.where(data.iloc[:, 0] == 0, -1, data.iloc[:, 0])

    # Generate lists
    x_train = data.drop(data.columns[0], axis=1).values
    y_train = data.iloc[:, 0].values

    # Test data

    # Read the database using pandas
    data = pd.read_csv("Input_Data/test.csv")

    # Recode the output column to get -1 and 1 output values
    data.iloc[:, 0] = np.where(data.iloc[:, 0] == 0, -1, data.iloc[:, 0])

    # Generate lists of lists
    x_test = data.drop(data.columns[0], axis=1).values
    y_test = data.iloc[:, 0].values



    #svn_model = svm_train_dual(data_train = x_train, label_train= y_train, regularisation_para_C = 10000)

    #test_accuracy = svn_predict_dual(data_test = x_test,label_test = y_test, svn_model = svn_model)

    #print(test_accuracy)




    #svn_model = svm_train_primal(data_train = x_train, label_train= y_train, regularisation_para_C = 10000)

    #test_accuracy = svn_predict_primal(data_test = x_test,label_test = y_test, svn_model = svn_model)

    #print(test_accuracy)

    #------------------------------------------------------------------------
    # Tuning of the primal implementation

    # set up a 5-fold partition of the train data
    k_fold = KFold(n_splits=5, random_state=23,shuffle=True)

    # Test different cost values in each split
    #cost_array = np.arange(start=0.5, stop=10.5, step=0.5)
    cost_array = [1.0, 10.0, 100.0,1000.0,10000.0]

    # Store the partial results in lists
    cost_full = []
    acc_full = []

    if not os.path.exists('Cross_Validation/cost_cv_svm_sklearn.csv'):
    
    # Loop trough the different combinations of step and number of iterations
        for cost in cost_array:

            # Store partial results for accuracy
            acc = []

        
            # Iterate thorgh the folds
            for kfold_train_index, kfold_test_index in k_fold.split(x_train, y_train):
                
                # Get the split into train and test
                kfold_x_train, kfold_x_test = x_train[kfold_train_index][:], x_train[kfold_test_index][:]
                kfold_y_train, kfold_y_test = y_train[kfold_train_index], y_train[kfold_test_index]

                # Train the SVM and get the accuracy
                svn_model = svm_train_primal(data_train = kfold_x_train, label_train= kfold_y_train, regularisation_para_C = cost)
                acc_kfold = svn_predict_primal(data_test = kfold_x_test, label_test = kfold_y_test, svn_model = svn_model)

                # Calculate the indexes and store them
                acc.append(acc_kfold)

            print("Testing the model with cost = ", cost)
            
            # Store the mean of the indexes for the 4 folds
            cost_full.append(cost)
            acc_full.append(np.mean(acc))
            print("Mean Accuracy = ", np.mean(acc))

        # Create pandas dataset
        dic = {'cost':cost_full, 'accuracy':acc_full}
        df_grid_search = pd.DataFrame(dic)
        df_grid_search.to_csv('Cross_Validation/cost_cv_svm_primal.csv')
        print("Tuning Ready!")
    else:
        # In case the parameters were already tuned
        df_grid_search = pd.read_csv('Cross_Validation/cost_cv_svm_primal.csv')
        print("Previous tuning detected")

    print("Tuning Ready!")

    # Search the bigger F1 index in the dataframe
    row_max = df_grid_search['accuracy'].argmax()

    # Get the the better performing step and number of iterations
    cost_max = float(df_grid_search['cost'].values[row_max])

    print("")
    print("The parameter was chosen looking at the maximum accuracy score")
    print("Accuracy:", df_grid_search['accuracy'][row_max])
    print("Using cost = ", cost_max)

    print("Training final model")

    svn_model = svm_train_primal(data_train = x_train, label_train= y_train, regularisation_para_C = cost_max)

    test_accuracy = svn_predict_primal(data_test = x_test,label_test = y_test, svn_model = svn_model)

    print(test_accuracy)

    print("Testing final model")
    print("Accuracy:", test_accuracy)

    print("Ready!")

    # Confusion matrix
    #print("Confusion matrix:")
    #print(confusion_matrix(y_test,predicted_final))


if __name__ == '__main__':
	main()

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

def svm_train_primal(data_train, label_train , regularisation_para_C):


    n_samples = np.shape(data_train)[0]
    m_features = np.shape(data_train)[1]


    P = np.zeros((m_features+n_samples+1,m_features+n_samples+1))

    for i in range(m_features):
        P[i,i] = 1


    #print(P)

    # The q term belong to the second bart of the top equation
    # The first part is zeroes, corresponding to the weights and the intercept, 
    # and the second part to the slack variables, that are multiplied by the cost
    q_zeros = np.zeros((m_features+1,1))
    q_slack = 1/n_samples * regularisation_para_C* np.ones((n_samples,1))
    q = np.vstack([q_zeros,q_slack])


    G = np.zeros((2*n_samples, m_features+1+n_samples))

    G[:n_samples,0:m_features] = label_train @ data_train
    G[:n_samples,m_features] = label_train.T
    G[:n_samples,m_features+1:]  = np.eye(n_samples)
    G[n_samples:,m_features+1:] = np.eye(n_samples)
    G = -G

    #print(G)

    h = np.zeros((2*n_samples,1))
    h[:n_samples] = -1

    #print(h)
    ## E and d are not used in the primal form
    ## convert to array
    ## have to convert everything to cxvopt matrices
    P = cvxopt.matrix(P,P.shape,'d')
    q = cvxopt.matrix(q,q.shape,'d')
    G = cvxopt.matrix(G,G.shape,'d')
    h = cvxopt.matrix(h,h.shape,'d')
    ## set up cvxopt
    ## z (the vector being minimized for) in this case is [w, b, eps].T
    print("Starting quadratic solver!")
    qp_solution = solvers.qp(P, q, G, h)

    weights = np.array(qp_solution['x'][:m_features])
    intercept = qp_solution['x'][m_features]
    slack = np.array(qp_solution['x'][m_features+1:])

    return qp_solution


def svn_predict_dual(data_test,label_test,svn_model):


    n_samples = np.shape(data_test)[0]
    m_features = np.shape(data_test)[1]

    weights = np.array(svn_model['x'][:m_features])
    intercept = svn_model['x'][m_features]

    weighted_sum = np.dot(data_test, weights) + intercept

    prediction = np.sign(weighted_sum)

    acc = accuracy_score(label_test,prediction)

    return acc






# Main function, it will preprocess the data, create train and test data,
# tune the parameters of the Random forest and evaluate the final model
def main():

    # Train data

    # Read the database using pandas
    data = pd.read_csv("Input_Data/train.csv")

    # Recode the output column to get -1 and 1 output values
    data.iloc[:, 0] = np.where(data.iloc[:, 0] == 0, -1, data.iloc[:, 0])

    # Generate lists of lists
    #x_train = data.iloc[:, -1:].values
    x_train = data.drop(data.columns[0], axis=1).values
    y_train = data.iloc[:, 0].values

    #print(x_train)
    #print(y_train)

    # Test data

    # Read the database using pandas
    data = pd.read_csv("Input_Data/test.csv")

    # Recode the output column to get -1 and 1 output values
    data.iloc[:, 0] = np.where(data.iloc[:, 0] == 0, -1, data.iloc[:, 0])

    # Generate lists of lists
    #x_test = data.iloc[:, -1:].values
    x_test = data.drop(data.columns[0], axis=1).values
    y_test = data.iloc[:, 0].values

    #print(x_test)
    #print(y_test)


    svn_model = svm_train_primal(data_train = x_train, label_train= y_train, regularisation_para_C = 10000)

    test_accuracy = svn_predict_dual(data_test = x_test,label_test = y_test, svn_model = svn_model)

    print(test_accuracy)










    # set up a 5-fold partition of the train data
    k_fold = KFold(n_splits=5, random_state=23,shuffle=True)

    # Test different cost values in each split
    #cost_array = np.arange(start=0.5, stop=10.5, step=0.5)
    cost_array = [0.01, 0.5, 1.0, 5.0, 10.0, 100.0]

    # Store the partial results in lists
    cost_full = []
    acc_full = []

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
            acc_kfold = svn_predict_dual(data_test = kfold_x_test, label_test = kfold_y_test, svn_model = svn_model)

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

    test_accuracy = svn_predict_dual(data_test = x_test,label_test = y_test, svn_model = svn_model)

    print(test_accuracy)

    print("Testing final model")
    print("Accuracy:", test_accuracy)

    print("Ready!")

    # Confusion matrix
    #print("Confusion matrix:")
    #print(confusion_matrix(y_test,predicted_final))


if __name__ == '__main__':
	main()

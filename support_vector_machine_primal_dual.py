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


    #n_samples = np.shape(data_train)[0]
    #n_features = np.shape(data_train)[1]

    #n,D = data_train.shape

    #ones = np.ones(n).reshape((n,1))

    #multinomial = np.array([(data_train[:,i]*data_train[:,j]).reshape((n,1)) for i in range(D) for j in range(i,D)]).T[0,:,:]

    #data_train = np.hstack([ones, data_train, multinomial])

    n,m = data_train.shape



    #label_train = label_train.reshape((1,label_train.shape[0]))

    P = np.zeros((m+n+1,m+n+1))

    for i in range(m):
        P[i,i] = 1

    print(P)
    
    q = np.vstack([np.zeros((m+1,1)), regularisation_para_C* np.ones((n,1))])

    #q = -q

    print(q)

    G = np.zeros((2*n, m+1+n))

    G[:n,0:m] = label_train @ data_train
    G[:n,m] = label_train.T
    G[:n,m+1:]  = np.eye(n)
    G[n:,m+1:] = np.eye(n)
    G = -G

    print(G)

    h = np.zeros((2*n,1))
    h[:n] = -1


    print(h)
    ## E and d are not used in the primal form
    ## convert to array
    ## have to convert everything to cxvopt matrices
    P = cvxopt.matrix(P,P.shape,'d')
    q = cvxopt.matrix(q,q.shape,'d')
    G = cvxopt.matrix(G,G.shape,'d')
    h = cvxopt.matrix(h,h.shape,'d')
    ## set up cvxopt
    ## z (the vector being minimized for) in this case is [w, b, eps].T
    sol = solvers.qp(P, q, G, h)

    w = np.array(sol['x'][:m])
    b = sol['x'][m]
    slack = np.array(sol['x'][m+1:])

    return sol


def svn_predict_dual(data_test,label_test,svn_model):


    n,m = data_test.shape

    w = np.array(svn_model['x'][:m])
    b = svn_model['x'][m]

    weighted = np.dot(data_test, w) + b

    prediction = np.sign(weighted)

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


    svn_model = svm_train_primal(data_train = x_train, label_train= y_train, regularisation_para_C = 10)

    test_accuracy = svn_predict_dual(data_test = x_test,label_test = y_test, svn_model = svn_model)

    print(test_accuracy)










    # set up a 5-fold partition of the train data
    k_fold = KFold(n_splits=5, random_state=23,shuffle=True)

    # Test different cost values in each split
    #cost_array = np.arange(start=0.5, stop=10.5, step=0.5)
    cost_array = [0.01, 0.5, 1.0, 5.0, 10.0]

    # Store the partial results in lists
    cost_full = []
    acc_full = []

    # Loop trough the different combinations of step and number of iterations
    for cost in cost_array:

        # Store partial results for accuracy, cohen kappa and F1
        acc = []

        # Initialize the support vector classifier
        svm = SVC(random_state=124, kernel='linear', C = cost)
    
        # Iterate thorgh the folds
        for kfold_train_index, kfold_test_index in k_fold.split(x_train, y_train):
            
            # Get the split into train and test
            kfold_x_train, kfold_x_test = x_train[kfold_train_index][:], x_train[kfold_test_index][:]
            kfold_y_train, kfold_y_test = y_train[kfold_train_index], y_train[kfold_test_index]

            # Train the SVM and get the predicted values
            svm.fit(kfold_x_train,kfold_y_train)
            predicted = svm.predict(kfold_x_test)

            # Calculate the indexes and store them
            acc.append(accuracy_score(kfold_y_test,predicted))

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

    # Training of the final model
    svm = SVC(random_state = 124, kernel='linear', C = cost_max)

    svm.fit(x_train,y_train)
    predicted_final = svm.predict(x_test)

    # Calculate the indexes of the final results
    acc_final = accuracy_score(y_test,predicted_final)

    print("Testing final model")
    print("Accuracy:", acc_final)

    # Confusion matrix
    print("Confusion matrix:")
    print(confusion_matrix(y_test,predicted_final))


if __name__ == '__main__':
	main()

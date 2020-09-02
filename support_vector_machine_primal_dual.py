###################################
# Julian Cabezas Pena
# Introduction to Statistical Machine Learning
# University of Adelaide
# Assingment 1
# Support Vector Machine Classifer using the primal and dual form
####################################

# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix
import cvxopt as cvxopt
import cvxopt.solvers as solvers
import os

# Train a SVM model using the primal form
def svm_train_primal(data_train, label_train , regularisation_para_C):
# data train: numpy array with explanatory features (X)
# label train: numpy array with target variable (y)
# regularisation_para_C: Cost parameter (integer)

    # Get the dimensions of the data
    n_samples = np.shape(data_train)[0]
    m_features = np.shape(data_train)[1]

    # First (Top equation) of the optimization problem

    # The vector X will be optimized, this vector corresponds to [weights,intercept,Epsilon(slack variable)]
    # According to the CVXOPT API the first term is (1/2) X.T P X
    # P will be a identity matrix in the top and zeroes in the bottom, that way we will get ||weights||^2
    P = np.zeros((m_features+n_samples+1,m_features+n_samples+1))
    P[:m_features,:m_features] = np.identity(m_features)

    # The q term belongs to the second part of the top equation
    # The first part of the q matrix is just zeroes, corresponding to the weights and the intercept, 
    # and the second part to the slack variables, that are multiplied by the cost/n
    q_zeros = np.zeros((m_features+1,1))
    q_slack = 1/n_samples * regularisation_para_C* np.ones((n_samples,1))
    q = np.vstack([q_zeros,q_slack])

    # Second equation (constrain)

    # G will be multiplied by X like this: Gx < h
    # Then the y(w.T x + b) equation is represented in the G array
    G = np.zeros((2*n_samples, m_features+1+n_samples))
    for i in range(n_samples):
        for j in range(m_features):
            G[i,j] = data_train[i,j] * label_train[i]

    # One extra column for the label alone (it will be multiplied with the intercept b)
    G[:n_samples,m_features] = label_train.T

    # The right part of the matrix will be identity matrices, they will be multiplied by the slack variable
    G[:n_samples,m_features+1:]  = np.identity(n_samples)
    G[n_samples:,m_features+1:] = np.identity(n_samples)
    
    # The h is the "1" in the second equation
    h = np.zeros((2 * n_samples,1))
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

    # Retuen an array with [w,b]
    return np.append(weights,np.array(intercept))



# Predict new data using a previowsly trained model
def svn_predict_primal(data_test,label_test,svn_model):
    # data test: numpy array with explanatory features (X)
    # label test: numpy array with target variable (y)
    # svn_model: Result of svm_train_primal(), numpy array with [w,b]

    # Extract the weights and intercept from the svn model output
    weights = svn_model[:(len(svn_model) - 1)]
    intercept = svn_model[len(svn_model) - 1]

    # Calculate the weighted sum
    weighted_sum = np.dot(data_test, weights) + intercept

    # Predict using the sign function, this will be -1 or +1
    prediction = np.sign(weighted_sum)

    # Get the accuracy of the prediction and return it
    acc = accuracy_score(label_test,prediction)

    return acc


# Train an SVN model using the dual form of optimization
def svm_train_dual(data_train, label_train , regularisation_para_C):
    # data train: numpy array with explanatory features (X)
    # label train: numpy array with target variable (y)
    # regularisation_para_C: Cost parameter (integer)

    # Get the dimensions of the data
    n_samples = np.shape(data_train)[0]
    m_features = np.shape(data_train)[1]

    # First equation

    # Fistly I have to calculate the Gram matrix to start building the top equation
    Gram = data_train.dot(data_train.T)
    # It is the sum of (y_i*y_j*<x_1,x_j>, we need to use the outer product to generate a n x n matrix
    P = np.outer(label_train,label_train) * Gram

    # q is just a column of -1, ad q.T is multiplied by the alphas in the SVXOPT API
    q = np.ones(n_samples) * -1

    # Second equation (constrain)

    # G has a diagonal matrix in the top and an identity matrix in the bottom, as only the 
    G = np.vstack((np.identity(n_samples) * -1.0, np.identity(n_samples)))
    # h is just zeroes on the top and the cost restrains on the bottom
    cost_restrain = np.ones(n_samples) * regularisation_para_C * (1/n_samples)
    h = np.hstack((np.zeros(n_samples), cost_restrain))

    # Third equation (second constrain)

    # Corresponds to the sum of alpha_i*y_i= 0, so A will only by the labels and b a column of zeroes
    A = np.reshape(label_train, (1,n_samples))
    b = 0.0

    # Convert to the CVXOPT matrix type
    P = cvxopt.matrix(P)
    q = cvxopt.matrix(q)
    G = cvxopt.matrix(G)
    h = cvxopt.matrix(h)
    A = cvxopt.matrix(A)
    b = cvxopt.matrix(b)

    # solve QP problem
    print("Starting the solver")
    qp_solution = cvxopt.solvers.qp(P, q, G, h, A, b)

    # Vector with lagrange multipliers, the solution of the quadratic equation
    lagrange = np.ravel(qp_solution['x'])

    # Support vectors have non zero lagrange multipliers
    sup_vect_bool = lagrange > 0.00001

    # Get the indexes of the support vectors
    sup_vect_index = []
    for i in range(len(lagrange)):
        if lagrange[i] > 0.00001:
            sup_vect_index.append(i)

    sup_vect_index = np.asarray(sup_vect_index)

    # Get the support vectors for x and y
    lagrange = lagrange[sup_vect_bool]
    sup_vect = data_train[sup_vect_bool]
    sup_vect_y = label_train[sup_vect_bool]

    # Weights
    weights = np.zeros(m_features)
    
    # The weights are the sum of the product of alpha_i, y_i and x_i for the support vectors
    for i in range(len(lagrange)):
        sum = lagrange[i] * sup_vect_y[i] * sup_vect[i]
        weights = weights + sum

    # Get the Intercept (b) value
    intercept = sup_vect_y - np.dot(data_train[sup_vect_index], weights)
    intercept = np.mean(intercept)

    return np.append(weights,np.array(intercept))

# Predict new data using a previowsly trained model
def svn_predict_dual(data_test,label_test,svn_model):
    # data test: numpy array with explanatory features (X)
    # label test: numpy array with target variable (y)
    # svn_model: Result of svm_train_primal(), numpy array with [w,b]

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


# Main function, it will preprocess the data, 
# tune the cost parameter for both primal and dual implementations
# and evaluate the final models
def main():

    #----------------------------------------------
    # Step 0: Data reading and preprocessing
    print("Step 0: Data reading and preprocessing")
    print("")

    # Train data

    # Read the database using pandas
    data = pd.read_csv("Input_Data/train.csv", header = None)

    # Recode the output column to get -1 and 1 output values
    data.iloc[:, 0] = np.where(data.iloc[:, 0] == 0, -1, data.iloc[:, 0])

    # Generate arrays
    x_train = data.drop(data.columns[0], axis=1).values
    y_train = data.iloc[:, 0].values

    # Test data

    # Read the database using pandas
    data = pd.read_csv("Input_Data/test.csv", header = None)

    # Recode the output column to get -1 and 1 output values
    data.iloc[:, 0] = np.where(data.iloc[:, 0] == 0, -1, data.iloc[:, 0])

    # Generate arrays
    x_test = data.drop(data.columns[0], axis=1).values
    y_test = data.iloc[:, 0].values

    #------------------------------------------------------------------------
    # Step 1: Tuning of the primal implementation
    print("Step 1: Tuning of the primal form")
    print("")

    # set up a 5-fold partition of the train data
    k_fold = KFold(n_splits=5, random_state=28,shuffle=True)

    # Test different cost values in each split
    cost_array = [1.0, 10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0, 90.0, 100.0]

    # Store the partial results in lists
    cost_full = []
    acc_full = []

    # Check if the tuning was already done
    if not os.path.exists('Cross_Validation/cost_cv_svm_primal.csv'):
    
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

                print("Accuracy of the fold ",acc_kfold)

                # Calculate the indexes and store them
                acc.append(acc_kfold)

            print("Testing the model with cost = ", cost)
            
            # Store the mean of the indexes for the 4 folds
            cost_full.append(cost)
            acc_full.append(np.mean(acc))
            print("Mean Accuracy = ", np.mean(acc))

        # Create pandas dataset and store the results
        dic = {'cost':cost_full, 'accuracy':acc_full}
        df_grid_search = pd.DataFrame(dic)
        df_grid_search.to_csv('Cross_Validation/cost_cv_svm_primal.csv')
        print("Tuning Ready!")
    else:
        # In case the parameters were already tuned, read the results
        df_grid_search = pd.read_csv('Cross_Validation/cost_cv_svm_primal.csv')
        print("Previous tuning detected")

    #------------------------------------------------------------------------
    # Step 2: Training and testing of primal implementation
    print("")
    print("Step 2: Training and testing of primal implementation")
    print("")

    # Search the bigger F1 index in the dataframe
    row_max = df_grid_search['accuracy'].argmax()

    # Get the the better performing step and number of iterations
    cost_max = float(df_grid_search['cost'].values[row_max])

    print("The parameter was chosen looking at the maximum accuracy score")
    print("Using cost = ", cost_max)

    print("Training final model")

    # Final model training using the cost we obtained using cross validation
    svn_model = svm_train_primal(data_train = x_train, label_train= y_train, regularisation_para_C = cost_max)

    # Save the model parameters
    dic = {'w_b':svn_model}
    df_parameters = pd.DataFrame(dic)
    df_parameters.to_csv('Results/model_parameters_primal.csv')

    # Getting the accuracy of the model on the train data
    train_accuracy = svn_predict_primal(data_test = x_train,label_test = y_train, svn_model = svn_model)
    print("Train Accuracy:", train_accuracy)

    # Get the accuracy of the final model on the test data
    test_accuracy = svn_predict_primal(data_test = x_test,label_test = y_test, svn_model = svn_model)

    print("Testing final model - primal implementation")
    print("Test Accuracy:", test_accuracy)

    print("Ready!")

    #------------------------------------------------------------------------
    # Step 3: Tuning of the primal implementation
    print("Step 1: Tuning of the dual form")
    print("")

    # set up a 5-fold partition of the train data
    k_fold = KFold(n_splits=5, random_state=28,shuffle=True)

    # Test different cost values in each split
    cost_array = [1.0, 10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0, 90.0, 100.0]

    # Store the partial results in lists
    cost_full = []
    acc_full = []

    # Check if the tuning was already done
    if not os.path.exists('Cross_Validation/cost_cv_svm_dual.csv'):
    
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
                svn_model = svm_train_dual(data_train = kfold_x_train, label_train= kfold_y_train, regularisation_para_C = cost)
                acc_kfold = svn_predict_dual(data_test = kfold_x_test, label_test = kfold_y_test, svn_model = svn_model)

                # Calculate the indexes and store them
                acc.append(acc_kfold)

            print("Testing the model with cost = ", cost)
            
            # Store the mean of the indexes for the 4 folds
            cost_full.append(cost)
            acc_full.append(np.mean(acc))
            print("Mean Accuracy = ", np.mean(acc))

        # Create pandas dataset with results and store them
        dic = {'cost':cost_full, 'accuracy':acc_full}
        df_grid_search = pd.DataFrame(dic)
        df_grid_search.to_csv('Cross_Validation/cost_cv_svm_dual.csv')
        print("Tuning Ready!")
    else:
        # In case the parameters were already tuned read the results
        df_grid_search = pd.read_csv('Cross_Validation/cost_cv_svm_dual.csv')
        print("Previous tuning detected")


    #------------------------------------------------------------------------
    # Step 3: Tuning of the primal implementation
    print("Step 4: Training and testing of the dual form")
    print("")

    # Search the bigger accuracy index in the dataframe
    row_max = df_grid_search['accuracy'].argmax()

    # Get the the better performing cost
    cost_max = float(df_grid_search['cost'].values[row_max])

    print("")
    print("The parameter was chosen looking at the maximum accuracy score")
    print("Using cost = ", cost_max)

    print("Training final model")

    # Final model training using the cost we obtained using cross validation
    svn_model = svm_train_dual(data_train = x_train, label_train= y_train, regularisation_para_C = cost_max)

    # Save the model parameters
    dic = {'w_b':svn_model}
    df_parameters = pd.DataFrame(dic)
    df_parameters.to_csv('Results/model_parameters_dual.csv')

    # Getting the accuracy of the model on the train data
    train_accuracy = svn_predict_dual(data_test = x_train,label_test = y_train, svn_model = svn_model)
    print("Train Accuracy:", train_accuracy)

    # Getting the accuracy of the model on the test data
    test_accuracy = svn_predict_dual(data_test = x_test,label_test = y_test, svn_model = svn_model)

    print("Testing final model - dual implementation")
    print("Test Accuracy:", test_accuracy)

    print("Ready!")



if __name__ == '__main__':
	main()

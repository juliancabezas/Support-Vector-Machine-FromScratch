###################################
# Julian Cabezas Pena
# Introduction to Statistical Machine Learning
# University of Adelaide
# Assingment 1
# Support Vector Machine Classifer using scikit-learn
####################################

# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
import os


# Main function, it will read the data, tune the C parameters and evaluate the final model
def main():

    # Train data

    # Read the database using pandas
    data = pd.read_csv("Input_Data/train.csv", header=None)

    # Recode the output column to get -1 and 1 output values
    data.iloc[:, 0] = np.where(data.iloc[:, 0] == 0, -1, data.iloc[:, 0])

    # Generate numpy arrays with explanatory features (x) and target feature (y)
    x_train = data.drop(data.columns[0], axis=1).values
    y_train = data.iloc[:, 0].values

    # Test data

    # Read the database using pandas
    data = pd.read_csv("Input_Data/test.csv", header=None)

    # Recode the output column to get -1 and 1 output values
    data.iloc[:, 0] = np.where(data.iloc[:, 0] == 0, -1, data.iloc[:, 0])

    # Generate numpy arrays
    x_test = data.drop(data.columns[0], axis=1).values
    y_test = data.iloc[:, 0].values


    # set up a 5-fold partition of the train data
    k_fold = KFold(n_splits=5, random_state=28, shuffle=True)

    # Test different cost values in each split
    cost_array = [1.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]

    # Store the partial results in lists
    cost_full = []
    acc_full = []

    # Check if the tuning was already done
    if not os.path.exists('Cross_Validation/cost_cv_svm_sklearn.csv'):

        # Loop trough the different combinations of step and number of iterations
        for cost in cost_array:

            # Store partial results for accuracy, cohen kappa and F1
            acc = []

            # Correction factor for the cost to be able to compare it with the primal and dual implementation (C/n)
            cost_n = cost / (8500*0.8)

            # Initialize the support vector classifier
            svm = SVC(kernel = 'linear', C = cost_n)

            # Iterate thorgh the folds
            for kfold_train_index, kfold_test_index in k_fold.split(x_train, y_train):

                # Get the split into train and test
                kfold_x_train, kfold_x_test = x_train[kfold_train_index][:], x_train[kfold_test_index][:]
                kfold_y_train, kfold_y_test = y_train[kfold_train_index], y_train[kfold_test_index]

                # Train the SVM and get the predicted values
                svm.fit(kfold_x_train, kfold_y_train)
                predicted = svm.predict(kfold_x_test)

                # Calculate the indexes and store them
                acc.append(accuracy_score(kfold_y_test, predicted))

            print("Testing the model with cost = ", cost)

            # Store the mean of the indexes for the 5 folds
            cost_full.append(cost)
            acc_full.append(np.mean(acc))
            print("Mean Accuracy = ", np.mean(acc))

        # Create pandas dataset with the results and write to csv
        dic = {'cost': cost_full, 'accuracy': acc_full}
        df_grid_search = pd.DataFrame(dic)
        df_grid_search.to_csv('Cross_Validation/cost_cv_svm_sklearn.csv')
        print("Tuning Ready!")
    else:
        # In case the parameters were already tuned, read the file with the results
        df_grid_search = pd.read_csv('Cross_Validation/cost_cv_svm_sklearn.csv')
        print("Previous tuning detected")

    print("Tuning Ready!")

    # Search the bigger F1 index in the dataframe
    row_max = df_grid_search['accuracy'].argmax()

    # Get the the better performing step and number of iterations
    cost_max = float(df_grid_search['cost'].values[row_max])

    print("")
    print("The parameter was chosen looking at the maximum accuracy score")
    print("Final cost parameter = ", cost_max/8500)

    print("Training final model")

    # Training of the final model
    svm = SVC(kernel='linear', C=cost_max/8500)

    svm.fit(x_train, y_train)

    # Calculate the training accuracy
    predicted_train = svm.predict(x_train)
    acc_train = accuracy_score(y_train, predicted_train)
    print("Train Accuracy:", acc_train)

    # Predict in the test dada
    predicted_final = svm.predict(x_test)

    # Calculate the indexes of the final results
    acc_final = accuracy_score(y_test, predicted_final)

    print("Testing final model")
    print("Test Accuracy:", acc_final)

    # Store w and b
    dic = {'w_b': np.append(svm.coef_, np.array(svm.intercept_))}
    df_parameters = pd.DataFrame(dic)
    df_parameters.to_csv('Results/model_parameters_sklearn.csv')


if __name__ == '__main__':
    main()

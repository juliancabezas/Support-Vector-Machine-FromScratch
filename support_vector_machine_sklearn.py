###################################
# Julian Cabezas Pena
# Deep Learning Fundamentals
# University of Adelaide
# Assingment 1
# Support Vector Machine Classifer testiong in the PIMA diabetes data
####################################


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC


# Main function, it will preprocess the data, create train and test data,
# tune the parameters of the Random forest and evaluate the final model
def main():

    # Train data

    # Read the database using pandas
    data = pd.read_csv("Input_Data/train.csv")

    # Recode the output column to get -1 and 1 output values
    #data.iloc[:, 0] = np.where(data.iloc[:, 0] == 0, -1, data.iloc[:, 0])

    # Generate lists of lists
    #x_train = data.iloc[:, -1:].values
    x_train = data.drop(data.columns[0], axis=1).values
    y_train = data.iloc[:, 0].values

    print(x_train)
    print(y_train)

    # Test data

    # Read the database using pandas
    data = pd.read_csv("Input_Data/test.csv")

    # Recode the output column to get -1 and 1 output values
    #data.iloc[:, 0] = np.where(data.iloc[:, 0] == 0, -1, data.iloc[:, 0])

    # Generate lists of lists
    #x_test = data.iloc[:, -1:].values
    x_test = data.drop(data.columns[0], axis=1).values
    y_test = data.iloc[:, 0].values

    print(x_test)
    print(y_test)

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

    print("Parameters of the model:")
    print('w = ',svm.coef_)
    print('b = ',svm.intercept_)
    print('Number of support vectors = ', svm.n_support_)
    print('Support vectors (indices) = ', svm.support_)
    print('Support vectors (data) = ', svm.support_vectors_)
    print('Coefficients of the support vector= ', np.abs(svm.dual_coef_))


if __name__ == '__main__':
	main()

# Support Vector Machine implementation

This code implements the Support Vector Algorithm to classify a binary class target variable usign the CVXOPT to solve the optimization problem.

The primal and dual forms of the optimization algorithm were implemented, and the results are compared with the Scikit-learn implementation

## Environment

This code was tested under a Linux 64 bit OS (Ubuntu 18.04 LTS), using Python 3.7.7

## How to run this code:

In order to use this code:

1. Install Miniconda or Anaconda
2. Add conda forge to your list of channels

In the terminal run:
    ```
    conda config --add channels conda-forge
    ```
3. Create a environment using the requirements.yml file included in this .zip:

Open a terminal in the folder were the requirements.yml file is (Assign1-code) and run:

    conda env create -f requirements.yml --name svm-env


4. Make sure the folder structure of the project is as follows
    ```
    Assign1-code
    ├── Input_Data
    ├── Cross_Validation
    ├── Results
    ├── support_vector_machine_primal_dual.py
    ├── support_vector_machine_sklearn.py
    └── ...
    ```
If there are .csv files in the Intermediate_Results the code will read them to avoid the delay of the RFE and Gridsearch and go straigh to fitting the models

5.  Run the code in the conda environment: Open a terminal in the Assign1-code folder  and run 
	```
	conda activate svm-env
	python support_vector_machine_primal_dual.py
    ```

6. For comparison, run the code of the SVM implementation in Scikit-learn
    ```
    python support_vector_machine_sklearn.py
    ```
Alternatevely, run the .py codesin your IDE of preference, (VS Code with the Python extension is recommended), using the root folder of the directory (Assign1-code) as working directory to make the relative paths work.

Note: Alternatevely, for 2 and 3 you can build your own environment following the package version contained in requirements.yml file

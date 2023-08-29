# 2 Machine Learning with Scikit Learn

## 2.1 Load Data
```py
import pandas as pd
dataset = pd.read_csv("Car details.csv")
print("dataset length:", len(dataset))
dataset.head()
```
![image](https://github.com/JefoGao/Resource_Machine_Learning_in_Python/assets/19381768/2f7c0c4c-2b58-42b7-b042-996e6aeafe38)

## 2.2 Data Preprocessing
### 2.2.1 Handle Missing & Duplicated Data
```py
# drop columns
dataset.drop(['name'], axis=1, inplace=True)
# drop column name: 'name'
# axis=0 drop column, axis=0 drop row
# inplace=True modifies the dataframe directly, rather than return a copied one
```
![image](https://github.com/JefoGao/Resource_Machine_Learning_in_Python/assets/19381768/a7f27eab-2079-4c79-a71c-480ccf0e6426)

```py
# check missing value
dataset.isna().sum()

# year              0
# selling_price     0
# km_driven         0
# fuel              0
# seller_type       0
# transmission      0
# owner             0
# mileage          19
# engine           19
# max_power        13
# seats            19
# dtype: int64
```
```py
# drop the rows with missing values
dataset = dataset.dropna()
print("dataset length:", len(dataset))

# dataset length: 7742
```
```py
# check duplicated rows
dataset.duplicated().any()

# True
```
```py
# remove duplicate
dataset = dataset.drop_duplicates()
print("dataset length:", len(dataset))

# dataset length: 6539
```
### 2.2.2 Handle Numeric Data
```py
# remove units strings from column: mileage, engine and max_power
# and then transform the column type to float

def remove_unit(df,colum_name) :
    t = []
    for i in df[colum_name]:
        number = str(i).split(' ')[0]
        t.append(number)
    return t

dataset['mileage'] = remove_unit(dataset,'mileage')
dataset['engine'] = remove_unit(dataset,'engine')
dataset['max_power'] = remove_unit(dataset,'max_power')

# transform the column type to float
dataset['mileage'] = pd.to_numeric(dataset['mileage'])
dataset['engine'] = pd.to_numeric(dataset['engine'])
dataset['max_power'] = pd.to_numeric(dataset['max_power'])
dataset.head()
```
![image](https://github.com/JefoGao/Resource_Machine_Learning_in_Python/assets/19381768/a60c50f7-2a74-4240-a2c1-cca5a884d20f)
```py
# adding 'age' feature to know how old the car is
# and dropping 'year' feature as it is useless now
dataset['age'] = 2023 - dataset['year']

# drop the year column by the function that we used before (hints drop function)
dataset.drop(['year'], axis = 1, inplace = True)

# take a look at the dataset afterwards
dataset.head()
```
![image](https://github.com/JefoGao/Resource_Machine_Learning_in_Python/assets/19381768/39d04be2-6d6a-4b99-90c8-bc597728ec4f)
```py
# get a summary of numerical columns
dataset.describe()
```
![image](https://github.com/JefoGao/Resource_Machine_Learning_in_Python/assets/19381768/dfdc61a0-525b-4290-9085-630d55eeffb0)
### 2.2.3 Handle Categorical Data
```py
# check value count for the categorical variables
print(dataset.fuel.value_counts(),"\n")
print(dataset.seller_type.value_counts(),"\n")
print(dataset.transmission.value_counts(), "\n")
print(dataset.owner.value_counts())
```
![image](https://github.com/JefoGao/Resource_Machine_Learning_in_Python/assets/19381768/b56d85f3-d985-4771-916a-618108560746)
```py
# ordinal encoding
dataset['owner'] = dataset['owner'].replace({'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3})
dataset.head()
```
![image](https://github.com/JefoGao/Resource_Machine_Learning_in_Python/assets/19381768/bd762eab-a2f8-49fd-8a1c-4bb38c2c5a00)
```py
# nominal variable
dataset = pd.get_dummies(dataset, columns=['fuel', 'seller_type', 'transmission'])
dataset.head() 
```
![image](https://github.com/JefoGao/Resource_Machine_Learning_in_Python/assets/19381768/c39907ff-be74-4100-b574-a6df1842abe8)
```py
# check dataset shape
dataset.shape # (6539, 17)

# define the input variables and the target variable
# target variable is the selling_price, and input variables are the rest of the columns
array = dataset.values
X = array[:,1:17]
y = array[:,0]
```
## 2.3 Split and Normalize data
```py
# split the training and testing dataset
# randomly sample the dataset with a random state of 123
# use 90% for training and 10% for testing use

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)
```
```py
# apply normalization on both train and testing dataset

from sklearn.preprocessing import MinMaxScaler
norm = MinMaxScaler().fit(X_train) # fit scaler on training data
X_train_norm = norm.transform(X_train) # transform training data
X_test_norm = norm.transform(X_test) # transform testing data
```
## 2.4 Model Training
Approach 1: Train the model based on entire training dataset and then evaluate the model based on testing dataset
Example of how to build a Linear Regression (LR) model
```py
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train_norm, y_train)
test_score = model.score(X_test_norm, y_test)
print("R2 of LR:", test_score)

# R2 of LR: 0.5917399630409528
```
Approach 2: Train the model based on training dataset with cross validation and then evaluate the model based on testing dataset
1) Define a 10 fold cross validation with data shufflling and set the random state with 123
benefits of cross validation: the model can be more generalized, and less prone to be over-fiited. Normally value of k is 5 or 10
```py
from sklearn.model_selection import KFold
kfold = KFold(n_splits=10, shuffle=True, random_state=123)
# set 10-fold cross validation after shuffle the dataset with random seed 123
```
2) Run 10-fold cross validation and print the average r-squared score based on the cross validation results
For a regression task, the default evaluation metrics is r squared.
```py
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

# basic training of the linear regression model
# define a LR model with default parameter setting
lr = LinearRegression()
# run the previously defined 10-fold validation on the dataset
results = cross_val_score(lr, X_train_norm, y_train, cv=kfold)
# print the averae r squared scores
print("Average R2 of LR:",results.mean())

# Average R2 of LR: 0.6301286590968103
```
Optimize the LR Models with Cross Validation
```py
# fine tune parameters for lr model
from sklearn.model_selection import GridSearchCV

grid_params_lr = {
    'fit_intercept': [True, False],
    'normalize': [True, False]
}

lr = LinearRegression()
gs_lr_result = GridSearchCV(lr, grid_params_lr, cv=kfold).fit(X_train_norm, y_train)
print(gs_lr_result.best_score_)

# 0.6301286590968103
```
# 2.4 Evaluate, Predict & Save
Evaluate a trained model using testing dataset
```py
# use the best model and evaluate on testing set
test_R2 = gs_lr_result.best_estimator_.score(X_test_norm, y_test)
print("R2 in testing:", test_R2)
# R2 in testing: 0.5917399630409526

# check the parameter setting for the best selected model
gs_lr_result.best_params_
# {'fit_intercept': True, 'normalize': True}
```
Predict with a trained model
```py
# predict with the first 5 data points
y_predict = gs_lr_result.best_estimator_.predict(X_test_norm[:5]) 
print(y_predict)

# [1214499.9736509   337037.49493902  551382.04202298  143190.56974847 498800.41080653]
```
Save and load a trained model
```py
import pickle

# Save to file in the current working directory
pkl_filename = "lr_model.pkl"  
with open(pkl_filename, 'wb') as file:  
    pickle.dump(gs_lr_result.best_estimator_, file)

# Load from file
with open(pkl_filename, 'rb') as file:  
    pickle_model = pickle.load(file)

# Calculate the accuracy score and predict target values
score = pickle_model.score(X_test_norm, y_test)  
print("R2 score:", score)

# R2 score: 0.5917399630409526
```

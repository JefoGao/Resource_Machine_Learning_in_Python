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

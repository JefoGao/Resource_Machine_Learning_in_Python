# 1* READING & WRITING FILES
# 1.8 Read and Write TXT Files
```py
f = open("files/Python.txt", "r") #opens file with name of "Python.txt"
print(f.read()) # read and print the entire file
f.close() # remember to colse the file

# Life is short,
# Use Python!
```
```py
f = open("files/Python.txt", "r") #opens file with name of "Python.txt"
print(f.readline()) # read the 1st line
print(f.readline()) # read the next line
f.close()

# Life is short,
#
# Use Python!
```
```py
f = open("files/Python.txt", "r") 
myList = []
for line in f:
    myList.append(line)
f.close()
print(myList)

# ['Life is short,\n', 'Use Python!']
```
```py
# Write file with name of "test.txt"
f = open("files/test.txt","w")  
f.write("I love Python.\n")
f.write("I will be a Python master.\n")
f.write("I need to keep learning!")
f.close()

# read and see the test.txt file
f = open("files/test.txt","r") 
print(f.read())
f.close()

# I love Python.
# I will be a Python master.
# I need to keep learning!
```
# 1.9 Read and Write CSV Files
```py
import csv
csvFile = open("files/test.csv", "r") 
reader = csv.reader(csvFile, delimiter=',')
# load the data in a dictionary 
result = {}
for item in reader:
    # ignore the first line
    if reader.line_num == 1:
        continue    
    result[item[0]] = item[1]
csvFile.close()
print(result)

# {'Ali': '25', 'Bob': '24', 'Chirs': '29'}
```
```py
import csv

fileHeader = ["name", "age"]
d1 = ["Chris", "27"]
d2 = ["Ming", "26"]

csvFile = open("files/write.csv", "w")
writer = csv.writer(csvFile)

# write the head and data
writer.writerow(fileHeader)
writer.writerow(d1)
writer.writerow(d2)

# Here is another command 
# writer.writerows([fileHeader, d1, d2])

csvFile.close()
```
# 1.10 Pandas
```py
# read csv files
import pandas as pd
import numpy as np

data = pd.read_csv("files/test.csv")
print(data) # data is pandas dataframe

#     Name  Age
# 0    Ali   25
# 1    Bob   24
# 2  Chirs   29
```
```py
# extract the age data
Age = np.array(data.Age, dtype = 'double')
print(Age)

# [25. 24. 29.]

# reshape this age vector
# NumPy allows you to ese either tuples or lists to specify the new shape
Age = np.reshape(Age, [3,1])
# or Age = np.reshape(Age, (3,1))
print(Age)

# [[25.]
#  [24.]
#  [29.]]
```
```py
# creating a pandas dataframe from data
df = pd.DataFrame({'Yes': [50, 21], 'No': [131, 2]})
print(df)

#    Yes   No
# 0   50  131
# 1   21    2
```
```py
# writing a pandas dataframe to csv file
df.to_csv('files/out.csv', index=False)
```
# 1.11 Pickle
The Pickle pacakge is used for serializing and de-serializing a Python object structure. Any object in python can be pickled so that it can be saved on disk and loaded back to continue the work.
```py
import numpy as np
import pickle
X = np.eye(5)

# Save the matirx X
with open('files/X.pickle', 'wb') as f:
    pickle.dump(X, f)
# Change the value of the original X    
X =  X + 4
print(X)

# [[5. 4. 4. 4. 4.]
#  [4. 5. 4. 4. 4.]
#  [4. 4. 5. 4. 4.]
#  [4. 4. 4. 5. 4.]
#  [4. 4. 4. 4. 5.]]
```
```py
# load the matrix 
with open('files/X.pickle', 'rb') as f:
    X = pickle.load(f)
print(X)

# [[1. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0.]
#  [0. 0. 1. 0. 0.]
#  [0. 0. 0. 1. 0.]
#  [0. 0. 0. 0. 1.]]
```

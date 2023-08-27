# 1. NUMPY

## 1.1 Arrays
```py
import numpy as np

# create an array
np.array(1) # rank 0 array - scalar
np.array([1,2,3]) # rank 1 array - series
np.array([[1,2,3],[1,2,3]]) # rank 2 array - matrix
np.array([[[1,2,3],[1,2,3]], [[1,2,3],[1,2,3]]]) # rank 3 array - tensor
```
```py
# the dimension of array
np.array([1,2,3]).shape 
```
```py
np.zeros((2,2)) # all zeros, where (2,2) is the shape
np.ones((1,2), dtype=np.int16) # all ones with specified type
np.full((2,2),7) # all constants
np.eye(2) # 2x2 identity matrix
np.random.random((2,2)) # random values from 0 - 1
```
```py
# create sequence
np.arrange(10, 30, 5)
np.arrange(0, 2, 0.3)
```

## 1.2 Array Attributes
```py
# the class of numpy array is ndarray
ndarray.ndim # number of dimensions (axes)
ndarray.shape # the dimension
ndarray.size # number of elements
ndarray.dtype # the type of elements
```
```py
np.arrange(15).reshape(3,5) # generate 3x5 matrix with value 0 - 14, dtype=int64
```

## 1.3 Array Indexing
```py
# slicing
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
b = a[:2, 1:3]
# [[2 3]
#  [6 7]]
```
```py
# mixing integer indexing with slicing
row_r1 = a[1, :]    # Rank 1 view of the second row of a
row_r2 = a[1:2, :]  # Rank 2 view of the second row of a
print(row_r1, row_r1.shape)  # Prints "[5 6 7 8] (4,)"
print(row_r2, row_r2.shape)  # Prints "[[5 6 7 8]] (1, 4)"

col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
print(col_r1, col_r1.shape)  # Prints "[ 2  6 10] (3,)"
print(col_r2, col_r2.shape)  # Prints "[[ 2]
                             #          [ 6]
                             #          [10]] (3, 1)"
```
```py
# integer array indexing
a = np.array([[1,2], [3, 4], [5, 6]])

# access elements (0,0), (1,1), (2,0)
print(a[[0, 1, 2], [0, 1, 0]])  # Prints "[1 4 5]"
# The above example of integer array indexing is equivalent to this:
print(np.array([a[0, 0], a[1, 1], a[2, 0]]))

# you can reuse the same element
print(a[[0, 0], [1, 1]])  # Prints "[2 2]"

# mutate one element from each row
a[np.arange(4), np.array([0,2,0,1])] += 10
```
```py
# boolean array indexing
a = np.array([[1,2], [3, 4], [5, 6]])
bool_idx = (a > 2)
print(bool_idx)      # Prints "[[False False]
                     #          [ True  True]
                     #          [ True  True]]"

print(a[bool_idx])  # Prints "[3 4 5 6]"
print(a[a > 2])     # Prints "[3 4 5 6]"
```
```py
# one more example
a = np.arange(10)**3 # [  0   1   8  27  64 125 216 343 512 729]
a[2] # 8
a[2:5] # array([ 8, 27, 64], dtype=int32)
a[:6:2] = -1000    # equivalent to a[0:6:2] = -1000;
# [-1000     1 -1000    27 -1000   125   216   343   512   729]
a[::-1] # reversed a
# array([729, 512, 343, 216, 125,  64,  27,   8,   1,   0])
```
## 1.4 Array Math


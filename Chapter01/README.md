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
```
```py
# mixing integer indexing with slicing
row_r1 = a[1, :]    # Rank 1 view of the second row of a
row_r2 = a[1:2, :]  # Rank 2 view of the second row of a
print(row_r1, row_r1.shape)  # Prints "[5 6 7 8] (4,)"
print(row_r2, row_r2.shape)  # Prints "[[5 6 7 8]] (1, 4)"
```

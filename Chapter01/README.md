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
```py
arr = np.arange(9).reshape(3,3)
new_arr = arr[:, [1,0,2]] # keep all row (order), swap the col order
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
```py
x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)

# addition, elementwise
print(x + y)
print(np.add(x, y))
# subtraction, elementwise
print(x - y)
print(np.subtract(x, y))
# product, elementwise
print(x * y)
print(np.multiply(x, y))
# division, elementwise
print(x / y)
print(np.divide(x, y))
# square root, elementwise
print(np.sqrt(x))

# inner product of vectors
v = np.array([9,10])
w = np.array([11, 12])
# both product 219
print(v.dot(w))
print(np.dot(v, w))

# matrix / vector product; both produce the rank 1 array [29 67]
print(x.dot(v))
print(np.dot(x, v))

# matrix / matrix product; both produce the rank 2 array
# [[19 22]
#  [43 50]]
print(x.dot(y))
print(np.dot(x, y))
```
```py
# numpy sum
x = np.array([[1,2],[3,4]])

print(np.sum(x))  # Compute sum of all elements; prints "10"
print(np.sum(x, axis=0))  # Compute sum of 1st dim; prints "[4 6]"
print(np.sum(x, axis=1))  # Compute sum of 2nd dim; prints "[3 7]"
```
```py
# transpose
x = np.array([[1,2], [3,4]])
print(x)    # Prints "[[1 2]
            #          [3 4]]"
print(x.T)  # Prints "[[1 3]
            #          [2 4]]"

# Note that taking the transpose of a rank 1 array does nothing:
v = np.array([1,2,3])
print(v)    # Prints "[1 2 3]"
print(v.T)  # Prints "[1 2 3]"
```
NumPy provides familiar mathematical functions such as sin, cos, and exp. In NumPy, these are called “universal functions”(ufunc). Within NumPy, these functions operate elementwise on an array, producing an array as output.
## 1.5 Reshape an Array
```py
a = np.floor(10*np.random.random((3,4))) #np.random.random() Return random floats in [0.0, 1.0).
# array([[7., 1., 7., 7.],
#        [0., 2., 8., 4.],
#        [6., 8., 2., 2.]])
a.ravel()  # returns the array, flattened
# array([7., 1., 7., 7., 0., 2., 8., 4., 6., 8., 2., 2.])
a.reshape(6,2)  # returns the array with a modified shape
a.T  # returns the array, transposed
```
## 1.6 Stacking Arrays
```py
a = np.floor(10*np.random.random((2,2)))
b = np.floor(10*np.random.random((2,2)))

np.vstack((a,b)) # vertical stacking
np.hstack((a,b)) # horizontal stacking
```
## 1.7 Splitting an Array
```py
a = np.floor(10*np.random.random((2,12)))
# array([[1., 7., 5., 7., 6., 0., 9., 1., 0., 9., 8., 0.],
#        [3., 6., 6., 5., 7., 8., 8., 2., 7., 0., 8., 4.]])

np.hsplit(a,3)   # Split a into 3
# [array([[1., 7., 5., 7.],
#         [3., 6., 6., 5.]]),
#  array([[6., 0., 9., 1.],
#         [7., 8., 8., 2.]]),
#  array([[0., 9., 8., 0.],
#         [7., 0., 8., 4.]])]

np.hsplit(a,(3,4))   # Split a after the third and the fourth column
# [array([[3., 6., 1.],
#         [6., 2., 8.]]),
#  array([[3.],
#         [4.]]),
#  array([[7., 0., 5., 7., 4., 1., 9., 0.],
#         [5., 5., 4., 1., 0., 2., 6., 4.]])]
```

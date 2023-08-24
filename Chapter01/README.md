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

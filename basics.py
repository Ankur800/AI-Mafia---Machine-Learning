import numpy as np
arr = np.array([[1, 2, 3], [4, 2, 5]])
print(type(arr))    # type of the array
print("Dimensions are", arr.ndim)     # dimensions of array
print("Shape is", arr.shape)    # rows and columns of the array
my_1D_array = np.array([1, 2, 3])
print("Shape of 1-d array is", my_1D_array.shape)    # shape for 1-D array

# creating array from list with type float
a = np.array([[1, 2, 4], [5, 8, 7]], dtype='float')
print("Array created using passed list:\n", a)
b = np.zeros((3, 4))
print("The zero matrix is:\n", b)
c = np.ones((3, 4))
print("The ones matrix is:\n", c)
d = np.eye(3)
print("The identity matrix is:\n", d)

# create a sequence of integers from 0 to 30 with steps of 5
e = np.arange(0, 30, 5)     # will get integers
print("Arrange 0 to 30 with step-size 5: ", e)

# create a sequence of 10 values in range 0 to 5
f = np.linspace(0, 5, 10)   # will get float
print("0 to 5 with 10 values using linspace fun: ", f)

arr = np.array([[1, 2, 3, 4], [5, 2, 4, 2], [1, 2, 0, 1]])
new_arr = arr.reshape(4, 3)
print("New shaped matrix is:\n", new_arr)

arr = np.array([[-1, 2, 0, 4], [4, -0.5, 6, 0], [2.6, 0, 7, 8], [3, -7, 4, 2]])
new_arr = arr[1:, :3]       # can also pass step size e.g. arr[1::2, :3:2]; here 2 is step size for row and col
print("The grabbed array is:\n", new_arr)

# BROADCASTING FEATURE OF NUMPY
a = np.array([1, 2, 5, 3])
# increment every element by 1
new_arr = a + 1
print(new_arr)
new_arr = a * 3
print(new_arr)
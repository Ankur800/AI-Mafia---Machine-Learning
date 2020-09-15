# 1. Import the numpy package under the name as np
import numpy as np

# 2. Print the numpy version and the configuration
print(np.__version__)
print(np.show_config())

# 3. Create a null vector of size 10
my_vector = np.zeros(10)

# 4. How to find the memory size of any array
print("Memory size of my_vector is {}".format(my_vector.size * my_vector.itemsize))

# 5. How to get the documentation of the numpy add function from the command line?
# python -c "import numpy; numpy.info(numpy.add)"
print(np.info())


# 6. Create a null vector of size 10 but the fifth value which is 1
my_vector = np.zeros(10)
my_vector[4] = 1
print(my_vector)

# 7. Create a vector with values ranging from 10 to 49
my_vector = np.arange(10, 50, 1)
print(my_vector)

# 8. Reverse a vector(first element becomes last)
my_vector = my_vector[::-1]
print(my_vector)

# 9. Create a 3x3 matrix with values ranging from 0 to 8
my_vector = np.arange(0, 9).reshape(3, 3)
print(my_vector)

# 10. Find indices of non - zero elements from [1, 2, 0, 0, 4, 0]
my_vector = np.nonzero([1, 2, 0, 0, 4, 0])
print(my_vector)

# 11. Create a 3x3 identity matrix
my_vector = np.eye(3)
print(my_vector)

# 12. Create a 3x3x3 array with random values
my_vector = np.random.random((3, 3, 3))
print(my_vector)

# 13. Create a 10x10 array with random values and find the minimum and maximum values
my_vector = np.random.random((10, 10))
print(my_vector.min(), my_vector.max())

# 14. Create a random vector of size 30 and find the mean value
my_vector = np.random.random(30)
print(my_vector.mean())

# 15. Create a 2d array with 1 on the border and 0 inside
my_vector = np.ones((4, 4))
my_vector[1:-1, 1:-1] = 0
print(my_vector)

# 16. How to add a border(filled with 0's) around an existing array?
my_vector = np.ones((4, 4))
my_vector = np.pad(my_vector, pad_width=1, mode='constant', constant_values=0)
print(my_vector)

# 17. What is the result of the following expression?
                    # 0 * np.nan
                    # np.nan == np.nan
                    # np.inf > np.nan
                    # np.nan - np.nan
                    # np.nan in set([np.nan])
                    # 0.3 == 3 * 0.1
######################## OUTPUT#########################
                        # nan
                        # False
                        # False
                        # nan
                        # True
                        # False

# 18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal
my_vector = np.diag(1+np.arange(4), k=-1)
print(my_vector)

# 19. Create a 8x8 matrix and fill it with a checkerboard pattern
my_vector = np.zeros((8, 8))
my_vector[1::2, ::2] = 1
my_vector[::2, 1::2] = 1
print(my_vector)

# 20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element?
print(np.unravel_index(100, (6, 7, 8)))
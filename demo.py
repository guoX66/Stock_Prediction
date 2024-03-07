import numpy as np

list1 = [[1, 2, 3, 4, 5], [4, 5, 6, 8, 4], [7, 8, 5, 6, 5]]
list1 = np.array(list1)
print(list1)
m, n = np.mean(list1[0]), np.std(list1[0])
print(m,n)

import numpy as np

test = np.array([[i for i in range(j*3,(j+1)*3)] for j in range(3)])
print(test.reshape(-1))
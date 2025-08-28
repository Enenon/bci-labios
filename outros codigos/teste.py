import numpy as np
from random import random as rnd

ma = np.array([[[rnd() for i in range(3)] for j in range(3)] for h in range(6)])
l = np.array([[i+h for i in range(3)] for h in range(6)])

t = np.concatenate((ma,l.T),-1)
print(t)
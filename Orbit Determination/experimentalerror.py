import random as rd
import numpy as np

data = []
for i in range(10000000):
    data.append(rd.normalvariate(2,0.5))

print np.std(data)

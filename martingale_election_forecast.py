import os
import numpy as np
import math
import pandas as pd
import random
from scipy.special import logit

os.chdir("C:/Users/Éva/Desktop/doksik/Data/output")

iterations = 1000
walks = 100
L = 1
k = 1
x0 = 0
d = 100

starting_value = 0
runs = 100
r = 0

x = np.zeros([iterations, runs])
while r < runs:
    z = np.zeros([iterations, 3])
    sigma = ((iterations - 0)/d) ** (1 / 2)
    z[0, 0] = iterations
    z[0, 1] = starting_value
    z[0, 2] = 1 / 2 * (1 + math.erf((logit(0.5) - z[0, 1]) / (2 * (sigma ** 2) ** (1 / 2))))
    x[0, 0] = z[0, 2]

    for i in np.arange(1, iterations):
        sigma = ((iterations - i) / d) ** (1 / 2)
        z[i, 0] = iterations - i
        z[i, 1] = z[i - 1, 1] + random.choice([-1, 1]) / d
        z[i, 2] = 1 / 2 * (1 + math.erf((logit(0.5) - z[i, 1]) / (2 * (sigma ** 2) ** (1 / 2))))
        x[i, r] = z[i, 2]
    r += 1

x = pd.DataFrame(x, index = np.arange(iterations), columns= np.arange(runs))
x = x.stack()
print(x)
x.to_csv("q_randomwalk.csv")


import numpy as np
from scipy.sparse import random
import pandas as pd
import math

def norm(x):
    return np.dot(x.T, x) ** 0.5

def randomMatrix(n,m, density):
    return random(n, m, density).A

def ResultsMatrix(results):
    max_len = len(max(results, key=len))
    for i in range(len(results)):
        last = results[i][len(results[i]) - 1]
        arr = np.array([last for j in range(max_len - len(results[i]))])
        results[i] = np.concatenate([results[i], arr])
    return results
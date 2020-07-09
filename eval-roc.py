
import numpy as np
import csv
import util

roc_samples = np.loadtxt('roc-val.csv', delimiter=',')
util.roc(roc_samples, match_criteria=[1, 2, 4, 6])